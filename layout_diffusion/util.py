import importlib
# import torch.distributed as dist
import numpy as np
import torch
import os
import random

import cv2
import colorsys


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)


def fix_seed(seed=23333):

    # if dist.is_initialized():
    #     seed = seed + dist.get_rank()

    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机种子确定
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)  # 所有的GPU设置种子

    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法

    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)


def loopy(data_loader):
    is_distributed=False
    # if dist.is_initialized() and dist.get_world_size() > 1:
    #     is_distributed=True
    epoch = 0
    while True:
        if is_distributed:
            epoch += 1
            data_loader.sampler.set_epoch(epoch)
        for x in iter(data_loader):
            yield x




def draw_layout(label, bbox, size, input_img=None, D_class_score=None, topleft_name=None):
    object_name_to_idx = {'__none__': 0, 'fire':1, 'smoke': 2, '__image__': 3}
    object_idx_to_name = {x2:x1 for (x1,x2) in zip(object_name_to_idx.keys(), object_name_to_idx.values())}

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
