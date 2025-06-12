import sys
sys.path.insert(0, "./")


import os
import json
import random
import PIL
import cv2
import glob
import argparse
from omegaconf import OmegaConf

import numpy as np
from collections import defaultdict

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from layout_diffusion.dataset.util import image_normalize
from layout_diffusion.dataset.augmentations import RandomSampleCrop, RandomMirror
from layout_diffusion.dataset.util import draw_layout


class FireDataset(Dataset):
    def __init__(self, image_dir, instances_json, non_fire_image_dir, image_size=(64, 64), mask_size=16,
                 max_num_samples=None, min_object_size=0.02, max_object_size = 0.8,
                 min_objects_per_image=3, max_objects_per_image=8, 
                 instance_whitelist=None, left_right_flip=False,include_other=False, 
                 filter_mode='LostGAN', use_MinIoURandomCrop=False, specific_image_ids=None,
                 mode='train'):

        super(Dataset, self).__init__()

        self.mode = mode
        self.max_objects_per_image = max_objects_per_image
        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_num_samples = max_num_samples
        self.filter_mode = filter_mode
        self.image_size = image_size
        self.min_object_size = min_object_size
        self.max_object_size = max_object_size
        self.left_right_flip = left_right_flip
        if left_right_flip:
            self.random_flip = RandomMirror()

        self.use_MinIoURandomCrop = use_MinIoURandomCrop
        if use_MinIoURandomCrop:
            self.MinIoURandomCrop = RandomSampleCrop()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=image_size, antialias=True),
            image_normalize()
        ])


        self.non_fire_image_dir = non_fire_image_dir
        self.non_fire_image_list = glob.glob(os.path.join(self.non_fire_image_dir, '*'))

        self.total_num_bbox = 0
        self.total_num_invalid_bbox = 0

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        #relabeling fire,smoke into 0,1
        instances_data['categories'][0]['id'] = 1
        instances_data['categories'][1]['id'] = 2


        #process "images"
        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        #process "categories"
        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {}
        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        
        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        category_whitelist = set(instance_whitelist)


        # Add object data from instances + filtering:
        self.image_id_to_objects = defaultdict(list)
        self.flag_first_fire_and_smoke = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            size_ok = W>=256 and H>=256
            box_ok = box_area > min_object_size and box_area < max_object_size
            object_name = object_idx_to_name[object_data['category_id']-183]            #fix later
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other

            if self.filter_mode == 'LostGAN':
                if size_ok and box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1) \
                    and (object_idx_to_name[object_data['category_id']-183] not in self.flag_first_fire_and_smoke[image_id]):
                    
                    object_data['category_id'] = object_data['category_id']-183
                    self.flag_first_fire_and_smoke[image_id].append(object_idx_to_name[object_data['category_id']])

                    self.image_id_to_objects[image_id].append(object_data)
            else:
                raise NotImplementedError

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0

        # None for 3
        self.vocab['object_name_to_idx']['__null__'] = 3

        self.object_idx_to_name = {x[0]:x[1] for x in zip(self.vocab['object_name_to_idx'].values(), self.vocab['object_name_to_idx'].keys())}


        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['__null__'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # Prune images that have too few or too many objects
        new_image_ids = []
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])


            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        
        
        self.image_ids = new_image_ids


        # get specific image ids or get specific number of images
        self.specific_image_ids = specific_image_ids
        if self.specific_image_ids:
            new_image_ids = []
            for image_id in self.specific_image_ids:
                if int(image_id) in self.image_ids:
                    new_image_ids.append(image_id)
                else:
                    print('image id: {} is not found in all image id list')
            self.image_ids = new_image_ids
        elif self.max_num_samples:
            self.image_ids = self.image_ids[:self.max_num_samples]


        self.vocab['pred_idx_to_name'] = [
            '__in_image__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx

    def filter_invalid_bbox(self, H, W, bbox, is_valid_bbox, verbose=False):

        for idx, obj_bbox in enumerate(bbox):
            if not is_valid_bbox[idx]:
                continue
            self.total_num_bbox += 1

            x, y, w, h = obj_bbox

            if (x >= W) or (y >= H):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue

            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = np.clip(x + w, 1, W)
            y1 = np.clip(y + h, 1, H)

            if (y1 - y0 < self.min_object_size) or (x1 - x0 < self.min_object_size):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue
            bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3] = x0, y0, x1, y1

        return bbox, is_valid_bbox

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def get_init_meta_data(self, image_id):
        layout_length = self.max_objects_per_image + 2
        meta_data = {
            'obj_bbox': torch.zeros([layout_length, 4]),
            'obj_class': torch.LongTensor(layout_length).fill_(self.vocab['object_name_to_idx']['__null__']),
            'is_valid_obj': torch.zeros([layout_length]),
            'filename': self.image_id_to_filename[image_id].replace('/', '_').split('.')[0]
        }

        # The first object will be the special __image__ object
        meta_data['obj_bbox'][0] = torch.FloatTensor([0, 0, 1, 1])
        meta_data['obj_class'][0] = self.vocab['object_name_to_idx']['__image__']
        meta_data['is_valid_obj'][0] = 1.0

        return meta_data

    def load_image(self, image_id):
        with open(os.path.join(self.image_dir, self.image_id_to_filename[image_id]), 'rb') as f:
            with PIL.Image.open(f) as image:
                image = image.convert('RGB')
        return image

    def load_image_cv2(self, image_id):
        image = cv2.imread(os.path.join(self.image_dir, self.image_id_to_filename[image_id]))
        return image

    #create hard-mask using method in generator
    def bbox_hard_mask_generator(self, bbox, H, W):
        num_bbox = bbox.shape[0]
        xmin, ymin, xmax, ymax = bbox[:, 0:1], bbox[:,1:2], bbox[:,2:3], bbox[:,3:4]
        ww = xmax - xmin
        hh = ymax - ymin
        x = np.repeat(np.expand_dims(np.linspace(0,1,num=W+1)[0:W],axis=0),axis=0,repeats=num_bbox) #2x128
        y = np.repeat(np.expand_dims(np.linspace(0,1,num=H+1)[0:H],axis=0),axis=0,repeats=num_bbox) #2x128
        x = (x - xmin) / ww       #([bo, W] - [bo, W])/[bo, W]
        y = (y - ymin) / hh       #([bo, H] - [bo, H])/[bo, H]
        x = np.repeat(np.expand_dims((x < 0) + (x > 1), axis=1),axis=1, repeats=H)
        y = np.repeat(np.expand_dims((y < 0) + (y > 1), axis=2),axis=2, repeats=W)
        
        bbox_hard_mask = (x+y).astype(np.float32)

        return 1 - np.all(bbox_hard_mask, axis=0, keepdims=True)



    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.

        """
        image_id = self.image_ids[index]
        image = self.load_image(image_id)
        image = np.array(image, dtype=np.float32) / 255.0

        non_fire_image_file = self.non_fire_image_list[index]
        with PIL.Image.open(non_fire_image_file) as non_fire_image:
                non_fire_image = non_fire_image.convert('RGB')
        non_fire_image = np.array(non_fire_image, dtype=np.float32) / 255.0


        H, W, _ = image.shape
        num_obj = len(self.image_id_to_objects[image_id])
        obj_bbox = np.array([obj['bbox'] for obj in self.image_id_to_objects[image_id]])            #xm,ym,w,h
        obj_class = np.array([obj['category_id'] for obj in self.image_id_to_objects[image_id]])
        is_valid_obj = [True for _ in range(num_obj)]


        # hard_mask_img = np.repeat(np.expand_dims(hard_mask.squeeze(),axis=2), axis=2, repeats=3).astype(np.uint8)*255
        # cv2.imshow('test', hard_mask_img)
        # cv2.waitKey()
        # obj_bbox[:,0::2] /= W
        # obj_bbox[:,1::2] /= H
        # image_cv2 = self.load_image_cv2(image_id=image_id)
        # image_cv2 = draw_layout(label=obj_class, bbox=obj_bbox, size=(256,256), input_img=image_cv2, object_idx_to_name=self.object_idx_to_name)
        # cv2.imshow("test2", image_cv2.astype(np.uint8))
        # cv2.waitKey()


        # get meta data
        meta_data = self.get_init_meta_data(image_id=image_id)
        meta_data['width'], meta_data['height'] = W, H
        meta_data['num_obj'] = num_obj

        # filter invalid bbox
        obj_bbox, is_valid_obj = self.filter_invalid_bbox(H=H, W=W, bbox=obj_bbox, is_valid_bbox=is_valid_obj)  #return bbox [xmin, ymin, xmax, ymax]

        # flip
        if self.left_right_flip:
            image, obj_bbox, obj_class = self.random_flip(image, obj_bbox, obj_class)

        # # random crop image and its bbox
        # if self.use_MinIoURandomCrop:
        #     image, updated_obj_bbox, updated_obj_class, tmp_is_valid_obj = self.MinIoURandomCrop(image, obj_bbox[is_valid_obj], obj_class[is_valid_obj])

        #     tmp_idx = 0
        #     tmp_tmp_idx = 0
        #     for idx, is_valid in enumerate(is_valid_obj):
        #         if is_valid:
        #             if tmp_is_valid_obj[tmp_idx]:
        #                 obj_bbox[idx] = updated_obj_bbox[tmp_tmp_idx]
        #                 tmp_tmp_idx += 1
        #             else:
        #                 is_valid_obj[idx] = False
        #             tmp_idx += 1

        #     meta_data['new_height'] = image.shape[0]
        #     meta_data['new_width'] = image.shape[1]
        #     H, W, _ = image.shape

        obj_bbox[:, 0::2] = obj_bbox[:, 0::2] / W
        obj_bbox[:, 1::2] = obj_bbox[:, 1::2] / H
        
        #create hard-mask
        bbox_hard_mask = self.bbox_hard_mask_generator(obj_bbox, H=256, W=256)      #[1,H,W]
        # bbox_hard_mask = np.repeat(np.expand_dims(bbox_hard_mask.squeeze(),axis=2), axis=2, repeats=3).astype(np.uint8)*255
        # cv2.imwrite("./outputs/images/test_mask.png", bbox_hard_mask)
        bbox_hard_mask = torch.FloatTensor(bbox_hard_mask)

        obj_bbox = torch.FloatTensor(obj_bbox[is_valid_obj])
        obj_class = torch.LongTensor(obj_class[is_valid_obj])
        

        num_selected = min(obj_bbox.shape[0], self.max_objects_per_image)
        selected_obj_idxs = random.sample(range(obj_bbox.shape[0]), num_selected)
        meta_data['obj_bbox'][1:1 + num_selected] = obj_bbox[selected_obj_idxs]
        meta_data['obj_class'][1:1 + num_selected] = obj_class[selected_obj_idxs]
        meta_data['is_valid_obj'][1:1 + num_selected] = 1.0
        meta_data['num_selected'] = num_selected
        meta_data['obj_class_name'] = [self.vocab['object_idx_to_name'][int(class_id)] for class_id in meta_data['obj_class']]
        meta_data['bbox_hard_mask'] = bbox_hard_mask
        meta_data['non_fire_image'] = self.transform(non_fire_image)

        return self.transform(image), meta_data


def fire_collate_fn_for_layout(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (N, L) giving object categories
    - masks: FloatTensor of shape (N, L, H, W)
    - is_valid_obj: FloatTensor of shape (N, L)
    """

    all_meta_data = defaultdict(list)
    all_imgs = []

    for i, (img, meta_data) in enumerate(batch):
        all_imgs.append(img[None])
        for key, value in meta_data.items():
            all_meta_data[key].append(value)

    all_imgs = torch.cat(all_imgs)
    for key, value in all_meta_data.items():
        if key in ['obj_bbox', 'obj_class', 'is_valid_obj', 'bbox_hard_mask'] or key.startswith('labels_from_layout_to_image_at_resolution'):
            all_meta_data[key] = torch.stack(value)

    return all_imgs, all_meta_data


def build_fire_dsets(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']
    params = cfg.data.parameters
    dataset = FireDataset(
        mode=mode,
        image_dir=os.path.join(params.root_dir, params[mode].image_dir),
        instances_json=os.path.join(params.root_dir, params[mode].instances_json),
        non_fire_image_dir=os.path.join(params.root_dir, params[mode].non_fire_image_dir),
        image_size=(params.image_size, params.image_size),
        mask_size=params.mask_size_for_layout_object,
        max_num_samples=params[mode].max_num_samples,
        min_object_size=params.min_object_size,
        max_object_size=params.max_object_size,
        min_objects_per_image=params.min_objects_per_image,
        max_objects_per_image=params.max_objects_per_image,
        instance_whitelist=params.instance_whitelist,
        left_right_flip=params[mode].left_right_flip,
        include_other=params.include_other,
        filter_mode=params.filter_mode,
        use_MinIoURandomCrop=params[mode].use_MinIoURandomCrop,
        specific_image_ids=params[mode].specific_image_ids
    )

    num_objs = dataset.total_objects()
    num_imgs = len(dataset)
    print('%s dataset has %d images and %d objects' % (mode, num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./configs/Fire_256x256/LayoutDiffusion_large.yaml')
    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    dataset = build_fire_dsets(cfg=cfg, mode='train')
    print(dataset[0])
    print(dataset[1])