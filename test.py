import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import argparse
import functools

import torch
import torch as th
from omegaconf import OmegaConf
import tqdm
import cv2
import colorsys

from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.util import fix_seed
from repositories.dpm_solver.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from layout_diffusion.dataset.data_loader import build_loaders
from scripts.get_gradio_demo import get_demo
from layout_diffusion.dataset.util import image_unnormalize_batch
import numpy as np

from layout_diffusion.dataset.data_loader import build_loaders

from layout_diffusion.gaussian_diffusion import get_named_beta_schedule
from layout_diffusion.resample import UniformSampler

from layout_diffusion.respace import build_diffusion
from layout_diffusion import dist_util

# object_name_to_idx = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14,
#     'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31,
#     'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42,
#     'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56,
#     'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73,
#     'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87,
#     'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90, 'banner': 92, 'blanket': 93, 'branch': 94, 'bridge': 95, 'building-other': 96, 'bush': 97, 'cabinet': 98, 'cage': 99, 'cardboard': 100,
#     'carpet': 101, 'ceiling-other': 102, 'ceiling-tile': 103, 'cloth': 104, 'clothes': 105, 'clouds': 106, 'counter': 107, 'cupboard': 108, 'curtain': 109, 'desk-stuff': 110, 'dirt': 111,
#     'door-stuff': 112, 'fence': 113, 'floor-marble': 114, 'floor-other': 115, 'floor-stone': 116, 'floor-tile': 117, 'floor-wood': 118, 'flower': 119, 'fog': 120, 'food-other': 121, 'fruit': 122,
#     'furniture-other': 123, 'grass': 124, 'gravel': 125, 'ground-other': 126, 'hill': 127, 'house': 128, 'leaves': 129, 'light': 130, 'mat': 131, 'metal': 132, 'mirror-stuff': 133, 'moss': 134,
#     'mountain': 135, 'mud': 136, 'napkin': 137, 'net': 138, 'paper': 139, 'pavement': 140, 'pillow': 141, 'plant-other': 142, 'plastic': 143, 'platform': 144, 'playingfield': 145, 'railing': 146,
#     'railroad': 147, 'river': 148, 'road': 149, 'rock': 150, 'roof': 151, 'rug': 152, 'salad': 153, 'sand': 154, 'sea': 155, 'shelf': 156, 'sky-other': 157, 'skyscraper': 158, 'snow': 159,
#     'solid-other': 160, 'stairs': 161, 'stone': 162, 'straw': 163, 'structural-other': 164, 'table': 165, 'tent': 166, 'textile-other': 167, 'towel': 168, 'tree': 169, 'vegetable': 170,
#     'wall-brick': 171, 'wall-concrete': 172, 'wall-other': 173, 'wall-panel': 174, 'wall-stone': 175, 'wall-tile': 176, 'wall-wood': 177, 'water-other': 178, 'waterdrops': 179,
#     'window-blind': 180, 'window-other': 181, 'wood': 182, 'other': 183, '__image__': 0, '__null__': 184}

object_name_to_idx = {'__image__': 0, 'fire':1, 'smoke': 2, '__null__': 3}



@torch.no_grad()
def layout_to_image_generation(cfg, model_fn, noise_schedule, custom_layout_dict, x_T=None):

    cfg.sample.classifier_free_scale = 1.0
    cfg.sample.timestep_respacing[0] = str(20)
    cfg.sample.sample_method = 'dpm_solver'


    layout_length = cfg.data.parameters.layout_length

    model_kwargs = {
        'obj_bbox': torch.zeros([1, layout_length, 4]),
        'obj_class': torch.zeros([1, layout_length]).long().fill_(object_name_to_idx['__null__']),
        'is_valid_obj': torch.zeros([1, layout_length]),
        'x_start': torch.zeros([1, 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size]),
        'bbox_hard_mask': torch.zeros([1, cfg.data.parameters.image_size, cfg.data.parameters.image_size])
    }
    model_kwargs['obj_class'][0][0] = object_name_to_idx['__image__']
    model_kwargs['obj_bbox'][0][0] = torch.FloatTensor([0, 0, 1, 1])
    model_kwargs['is_valid_obj'][0][0] = 1.0

    for obj_id in range(1, len(custom_layout_dict['obj_bbox'][0])-1):
        obj_bbox = custom_layout_dict['obj_bbox'][0][obj_id]
        obj_class = custom_layout_dict['obj_class_name'][0][obj_id]
    
        if obj_class == 'pad':
            obj_class = '__null__'

        model_kwargs['obj_bbox'][0][obj_id] = torch.FloatTensor(obj_bbox)
        model_kwargs['obj_class'][0][obj_id] = object_name_to_idx[obj_class]
        model_kwargs['is_valid_obj'][0][obj_id] = 1
        
    x_start = custom_layout_dict['x_start'][0].cuda()
    bbox_hard_mask = custom_layout_dict['bbox_hard_mask'][0].cuda()
    model_kwargs['x_start'][0] = x_start
    model_kwargs['bbox_hard_mask'][0] = bbox_hard_mask

    print(model_kwargs)


    wrappered_model_fn = model_wrapper(
        model_fn,
        noise_schedule,
        is_cond_classifier=False,
        total_N=1000,
        model_kwargs=model_kwargs
    )
    for key in model_kwargs.keys():
        model_kwargs[key] = model_kwargs[key].cuda()

    dpm_solver = DPM_Solver(wrappered_model_fn, noise_schedule)

    # x_T = th.randn((1, 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size)).cuda() if x_T is None else x_T.cuda()
    x_T = th.randn((1, 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size)).cuda()
    x_T = x_start*(1-bbox_hard_mask) + x_T*bbox_hard_mask

    sample = dpm_solver.sample(
        x_T,
        steps=int(cfg.sample.timestep_respacing[0]),
        eps=float(cfg.sample.eps),
        adaptive_step_size=cfg.sample.adaptive_step_size,
        fast_version=cfg.sample.fast_version,
        clip_denoised=False,
        rtol=cfg.sample.rtol,
        x_start=x_start,
        bbox_hard_mask=bbox_hard_mask
    )  # (B, 3, H, W), B=1

    sample = sample.clamp(-1, 1)

    generate_img = np.array(sample[0].cpu().permute(1,2,0) * 127.5 + 127.5, dtype=np.uint8)
    # generate_img = np.transpose(generate_img, (1,0,2))
    print(generate_img.shape)


    print("sampling complete")

    return generate_img





def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)



@torch.no_grad()
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
        
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )



def forward_sample(cfg, x_start, diffusion):
    betas = get_named_beta_schedule(cfg.diffusion.parameters.noise_schedule, cfg.diffusion.parameters.diffusion_steps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    schedule_sampler = UniformSampler(diffusion)
    t, weights = schedule_sampler.sample(x_start.shape[0], dist_util.dev())

    t = th.tensor(999, device=x_start.device)

    noise = th.randn_like(x_start)
    x_T = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)

    return x_T






def draw_layout(label, bbox, size, input_img=None, D_class_score=None, topleft_name=None):
    if input_img is None:
        temp_img = np.zeros([size[0]+50,size[1]+50,3]) + 255
    else:
        try:
            num_c = input_img.shape[2]
        except:
            num_c = 1
        temp_img = np.zeros([size[0]+50,size[1]+50,num_c]) + 127
        input_img = np.expand_dims(cv2.resize(input_img, size), axis=-1) if num_c==1 else cv2.resize(input_img, size)
        temp_img[25:25+size[0], 25:25+size[1],:] = input_img
        temp_img = np.repeat(temp_img, repeats=3, axis=2) if num_c==1 else temp_img
     
    bbox = (bbox[0]*(size[0])).numpy()
    label = label[0]
    num_classes = len(object_name_to_idx)

    rectangle_hsv_tuples     = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    label_hsv_tuples         = [(1.0 * x / num_classes, 1., 1.) for x in range(int(num_classes/2), num_classes)] 
    label_hsv_tuples        += [(1.0 * x / num_classes, 1., 1.) for x in range(0, int(num_classes/2))]                                
    rand_rectangle_colors    = list(map(lambda x: colorsys.hsv_to_rgb(*x), rectangle_hsv_tuples))
    rand_rectangle_colors    = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_rectangle_colors))
    rand_text_colors         = list(map(lambda x: colorsys.hsv_to_rgb(*x), label_hsv_tuples))
    rand_text_colors         = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_text_colors))
    
    for i in range(len(bbox)):
        label_color = rand_text_colors[object_name_to_idx[label[i]]]
        if num_classes < 5:
            if label[i] == 1:
                label_color = (255, 0, 0)
            elif label[i] == 2:
                label_color = (0, 0, 255)
            else:
                label_color = (0, 255, 0)
        
        xmin, ymin, xmax, ymax = bbox[i]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        x, y, width, height = int(xmin), int(ymin), int(xmax - xmin +1), int(ymax - ymin +1)

        x,y = x+25, y+25
        class_name = label[i]
        cv2.rectangle(temp_img, (x, y), (x + width, y + height), label_color, 1)  # (0, 255, 0) is the color (green), 2 is the thickness
        cv2.putText(temp_img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

    if D_class_score is not None:
        if D_class_score>=0:
            D_class_text = "Real: {:.2f}%".format(D_class_score*100)
        else:
            D_class_text = "Fake: {:.2f}%".format(-D_class_score*100)
        cv2.putText(temp_img, D_class_text, (25,size[1]+50 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    cv2.rectangle(temp_img, (25, 25), (25 + size[1], 25 + size[0]), (255,255,255), 1)
        
    if topleft_name is not None:
        cv2.putText(temp_img,"| "+topleft_name, (int(size[0]/2)+25, size[1]+50 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    return temp_img


















def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./configs/Fire_256x256/LayoutDiffusion_large.yaml')
    parser.add_argument("--share", action='store_true')

    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)


    print("creating model...")
    model = build_model(cfg)
    model.cuda()
    # print(model)

    diffusion = build_diffusion(cfg)

    if cfg.sample.pretrained_model_path:
        print("loading model from {}".format(cfg.sample.pretrained_model_path))
        checkpoint = torch.load(cfg.sample.pretrained_model_path, map_location="cpu")

        try:
            model.load_state_dict(checkpoint, strict=True)
            print('successfully load the entire model')
        except:
            print('not successfully load the entire model, try to load part of model')

            model.load_state_dict(checkpoint, strict=False)

    model.cuda()
    if cfg.sample.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, **kwargs):
        assert obj_class is not None
        assert obj_bbox is not None

        cond_image, cond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj, x_start=kwargs['x_start'], bbox_hard_mask=kwargs['bbox_hard_mask']
        )
        cond_mean, cond_variance = th.chunk(cond_image, 2, dim=1)

        obj_class = th.ones_like(obj_class).fill_(model.layout_encoder.num_classes_for_layout_object - 1)
        obj_class[:, 0] = 0

        obj_bbox = th.zeros_like(obj_bbox)
        obj_bbox[:, 0] = th.FloatTensor([0, 0, 1, 1])

        is_valid_obj = th.zeros_like(obj_class)
        is_valid_obj[:, 0] = 1.0

        if obj_mask is not None:
            obj_mask = th.zeros_like(obj_mask)
            obj_mask[:, 0] = th.ones(obj_mask.shape[-2:])

        uncond_image, uncond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj, x_start=kwargs['x_start'], bbox_hard_mask=kwargs['bbox_hard_mask']
        )
        uncond_mean, uncond_variance = th.chunk(uncond_image, 2, dim=1)

        mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

        return mean

        if cfg.sample.sample_method in ['ddpm', 'ddim']:
            return [th.cat([mean, cond_variance], dim=1), cond_extra_outputs]
        else:
            return mean


    print("creating diffusion...")

    noise_schedule = NoiseScheduleVP(schedule='linear')

    print('sample method = {}'.format(cfg.sample.sample_method))
    print("sampling...")


    test_loader = build_loaders(cfg, mode='val')

    for i, [batch, cond] in enumerate(test_loader):

        cond['x_start'] = cond['non_fire_image']

        obj_class = cond['obj_class_name']
        # if 'fire hydrant' not in obj_class:
        #     continue


        x_T = forward_sample(cfg, batch.cuda(), diffusion)


        fake_fire_img = layout_to_image_generation(cfg=cfg, model_fn=model_fn, noise_schedule=noise_schedule, custom_layout_dict=dict(cond), x_T=x_T)

        real_fire_img = np.array(batch[0].cpu().permute(1,2,0) * 127.5 + 127.5, dtype=np.uint8)
        non_fire_img = np.array(cond['non_fire_image'][0].cpu().permute(1,2,0) * 127.5 + 127.5, dtype=np.uint8)

        obj_bbox = cond['obj_bbox']
        
        
        layout = np.zeros((256, 256, 3), np.uint8) + 150
        layout = draw_layout(obj_class, obj_bbox, [256,256], layout)

        cv2.imwrite("./outputs/images/{}_real_fire_img.png".format(i), cv2.cvtColor(real_fire_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./outputs/images/{}_fake_fire_img.png".format(i), cv2.cvtColor(fake_fire_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./outputs/images/{}_non_fire_img.png".format(i), cv2.cvtColor(non_fire_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./outputs/images/{}_layout.png".format(i), cv2.cvtColor(layout.astype(np.uint8), cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()