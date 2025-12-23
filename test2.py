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
from layout_diffusion.dataset.data_loader import build_loaders
from scripts.get_gradio_demo import get_demo
from layout_diffusion.dataset.util import image_unnormalize_batch
import numpy as np

from layout_diffusion.dataset.data_loader import build_loaders

from layout_diffusion.gaussian_diffusion import get_named_beta_schedule
from layout_diffusion.resample import build_schedule_sampler
from layout_diffusion.respace import build_diffusion, build_diffusion_ddim
from layout_diffusion import dist_util

object_name_to_idx = {'__image__': 0, 'fire':1, 'smoke': 2, '__null__': 3}



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

    #diffusion
    print("creating diffusion...")
    diffusion = build_diffusion_ddim(cfg)

    #dataloader
    test_loader = build_loaders(cfg, mode='val')

    #output folder
    if not os.path.exists("./outputs/images"):
        os.makedirs("./outputs/images")
    elif os.path.exists("./outputs/images") and len(os.listdir("./outputs/images")) == 0:
        pass
    else:
        max_id = max([int(x.split("_")[-1]) for x in os.listdir("./outputs") if "_" in x] if len(os.listdir("./outputs")) > 1 else [0] if len(os.listdir("./outputs"))==1 else [-1])
        os.rename("./outputs/images", "./outputs/images_{}".format(max_id+1))
        os.makedirs("./outputs/images")


    for i, [batch, cond] in enumerate(test_loader):
        cond['x_start'] = batch
        cond['x_cond'] = cond['non_fire_image']
        
        cond = {k:v.cuda() for k,v in cond.items() if k in model.layout_encoder.used_condition_types}
        noise = torch.randn_like(batch).cuda()


        pred_data = diffusion.ddim_sample_loop(
            model,
            shape=(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]),
            noise=noise,
            clip_denoised=cfg.sample.clip_denoised,
            model_kwargs=cond,
            progress=False,
        )

        fake_fire_img = np.array(pred_data[0]['sample'][0].cpu().permute(1,2,0) * 127.5 + 127.5, dtype=np.uint8)

        real_fire_img = np.array(batch[0].cpu().permute(1,2,0) * 127.5 + 127.5, dtype=np.uint8)
        non_fire_img = np.array(cond['non_fire_image'][0].cpu().permute(1,2,0) * 127.5 + 127.5, dtype=np.uint8)

        obj_bbox = cond['obj_bbox']
        
        obj_class = cond['obj_class_name']
        layout = np.zeros((256, 256, 3), np.uint8) + 150
        layout = draw_layout(obj_class, obj_bbox, [256,256], layout)

        cv2.imwrite("./outputs/images/{}_real_fire_img.png".format(i), cv2.cvtColor(real_fire_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./outputs/images/{}_fake_fire_img.png".format(i), cv2.cvtColor(fake_fire_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./outputs/images/{}_non_fire_img.png".format(i), cv2.cvtColor(non_fire_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./outputs/images/{}_layout.png".format(i), cv2.cvtColor(layout.astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()