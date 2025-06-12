"""
Train a diffusion model on images.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse

# import torch.distributed as dist
from omegaconf import OmegaConf
import torch

from layout_diffusion import logger, dist_util
from layout_diffusion.train_util import TrainLoop
from layout_diffusion.util import loopy
from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.resample import build_schedule_sampler
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.respace import build_diffusion





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./configs/Fire_256x256/LayoutDiffusion_large.yaml')
    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)
    print(OmegaConf.to_yaml(cfg))

    # dist_util.setup_dist(local_rank=cfg.local_rank)
    logger.configure(dir=cfg.train.log_dir)

    logger.log("creating model...")
    model = build_model(cfg)

    if cfg.model.pretrained_model_path:
        print("loading model from {}".format(cfg.model.pretrained_model_path))
        checkpoint = torch.load(cfg.model.pretrained_model_path, map_location="cpu")

        try:
            model.load_state_dict(checkpoint, strict=True)
            print('successfully load the entire model')
        except:
            print('not successfully load the entire model, try to load part of model')
            model.load_state_dict(checkpoint, strict=False)
    model.to(dist_util.dev())
    # print(model)

    logger.log("creating diffusion...")
    diffusion = build_diffusion(cfg)

    logger.log("creating schedule sampler...")
    schedule_sampler = build_schedule_sampler(cfg, diffusion)

    logger.log("creating data loader...")
    train_loader = build_loaders(cfg, mode='train')

    logger.log("training...")
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        schedule_sampler=schedule_sampler,
        data=loopy(train_loader),
        batch_size=cfg.data.parameters.train.batch_size,
        **cfg.train
    )
    trainer.run_loop()


if __name__ == "__main__":
    main()
