import copy
import functools
import os

import blobfile as bf
import torch as th
# import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from layout_diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
import torch
import numpy as np
from layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
from diffusers.models import AutoencoderKL

class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            micro_batch_size,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            find_unused_parameters=False,
            only_update_parameters_that_require_grad=False,
            classifier_free=False,
            classifier_free_dropout=0.0,
            pretrained_model_path='',
            log_dir="",
            latent_diffusion=False,
            vae_root_dir="",
            scale_factor=0.18215
    ):
        self.log_dir =log_dir
        logger.configure(dir=log_dir)
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        if pretrained_model_path:
            logger.log("loading model from {}".format(pretrained_model_path))
            try:
                model.load_state_dict(
                    dist_util.load_state_dict(pretrained_model_path, map_location="cpu"),strict=True
                )
            except:
                print('not successfully load the entire model, try to load part of model')
                model.load_state_dict(
                    dist_util.load_state_dict(pretrained_model_path, map_location="cpu"),strict=False
                )

        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size if micro_batch_size > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size #* dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            only_update_parameters_that_require_grad=only_update_parameters_that_require_grad
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available() and th.cuda.device_count() > 1:
            self.use_ddp = True
            self.find_unused_parameters = find_unused_parameters
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=self.find_unused_parameters,
            )
        else:
            # if dist.get_world_size() > 1:
            #     logger.warn(
            #         "Distributed training requires CUDA. "
            #         "Gradients will not be synchronized properly!"
            #     )
            self.use_ddp = False
            self.ddp_model = self.model

        self.classifier_free = classifier_free
        self.classifier_free_dropout = classifier_free_dropout
        self.dropout_condition = False

        self.scale_factor=scale_factor
        self.vae_root_dir = vae_root_dir
        self.latent_diffusion = latent_diffusion
        if self.latent_diffusion:
            self.instantiate_first_stage()

    def instantiate_first_stage(self):
        model = AutoencoderKL.from_pretrained(self.vae_root_dir).to(dist_util.dev())
        self.first_stage_model = model.eval()
        self.first_stage_model.train = False
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    # https://github.com/huggingface/diffusers/blob/29b2c93c9005c87f8f04b1f0835babbcea736204/src/diffusers/models/autoencoder_kl.py
    @th.no_grad()
    def get_first_stage_encoding(self, x):
        with th.no_grad():
            encoder_posterior = self.first_stage_model.encode(x, return_dict=True)[0]
            z = encoder_posterior.sample()
            return z.to(dist_util.dev()) * self.scale_factor

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"resume step = {self.resume_step}...")
            # if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:07}.pt"
        )
        logger.log(f"try to load optimizer state from checkpoint: {opt_checkpoint}")
        if bf.exists(opt_checkpoint):
            logger.log(f"successfully loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        def run_loop_generator():
            while (
                    not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps
            ):
                yield

        for _ in tqdm(run_loop_generator()):
            batch, cond = next(self.data)
            if self.classifier_free and self.classifier_free_dropout > 0.0:
                p = np.random.rand()
                self.dropout_condition = False
                if p < self.classifier_free_dropout:
                    self.dropout_condition = True

                    if isinstance(self.model, LayoutDiffusionUNetModel):
                        if 'obj_class' in self.model.layout_encoder.used_condition_types:
                            cond['obj_class'] = torch.ones_like(cond['obj_class']).fill_(self.model.layout_encoder.num_classes_for_layout_object - 1)
                            cond['obj_class'][:, 0] = 0
                        if 'obj_bbox' in self.model.layout_encoder.used_condition_types:
                            cond['obj_bbox'] = torch.zeros_like(cond['obj_bbox'])
                            cond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 1, 1])
                        if 'obj_mask' in self.model.layout_encoder.used_condition_types:
                            cond['obj_mask'] = torch.zeros_like(cond['obj_mask'])
                            cond['obj_mask'][:, 0] = torch.ones(cond['obj_mask'].shape[-2:])
                        cond['is_valid_obj'] = torch.zeros_like(cond['is_valid_obj'])
                        cond['is_valid_obj'][:, 0] = 1.0

            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
                
                # if (self.step + self.resume_step) >= 100000:
                #     return

            self.step += 1
            # torch.cuda.empty_cache()

            if (self.step - 1) > 50000: #train only 50,000 iterations
                break

        # # Save the last checkpoint if it wasn't already saved.
        # if (self.step - 1) % self.save_interval != 0:
        #     self.save()

        

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.micro_batch_size):
            micro = batch[i: i + self.micro_batch_size].to(dist_util.dev())
            if self.latent_diffusion:
                micro = self.get_first_stage_encoding(micro).detach()
            micro_cond = {
                k: v[i: i + self.micro_batch_size].to(dist_util.dev())
                for k, v in cond.items() if k in self.model.layout_encoder.used_condition_types
            }

            micro_cond['x_start'] = micro            #change

            last_batch = (i + self.micro_batch_size) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # if dist.get_rank() == 0:
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step + self.resume_step):07d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):07d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0:
        with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):07d}.pt"),
                "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

        # dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):07d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
