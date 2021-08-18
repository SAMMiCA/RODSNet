import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

from utils import utils
from utils.file_io import read_img, read_disp

from pathlib import Path

class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root: Path, dataset_name='cityscapes', mode='test', save_filename=False,
                 transform=None, opts=None):
        self.root = root
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.dataset_name = dataset_name
        self.ignore_index = 255
        self.opts = opts

        cityscapes_dict = {
            'train': 'filenames/cityscapes/cityscapes_semantic_train.txt',
            'val': 'filenames/cityscapes/cityscapes_semantic_val.txt',
            'test': 'filenames/cityscapes/cityscapes_semantic_test.txt'
        }
        kitti_2015_dict = {
            'train': 'filenames/kitti_2015/KITTI_2015_train.txt',
            #'val': 'filenames/kitti_2015/KITTI_2015_val.txt',
            'val': 'filenames/kitti_2015/KITTI_2015_all_val.txt',
            'test': 'filenames/kitti_2015/KITTI_2015_test.txt'
        }

        kitti_mix_dict = {
            'train': 'filenames/kitti_mix/KITTI_MIX_train.txt',
            'val': 'filenames/kitti_mix/KITTI_MIX_all_val.txt',
            'test': 'filenames/kitti_mix/KITTI_MIX_val.txt'
        }

        scene_flow_dict = {
            'train': 'filenames/sceneflow/SceneFlow_finalpass_train.txt',
            'val': 'filenames/sceneflow/SceneFlow_finalpass_val.txt',
            'test': 'filenames/sceneflow/SceneFlow_finalpass_test.txt'
        }

        if mode not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        self.samples = []
        dataset_dict = {'cityscapes': cityscapes_dict,
                        'kitti_2015': kitti_2015_dict,
                        'sceneflow': scene_flow_dict,
                        'kitti_mix': kitti_mix_dict}
        data_filenames = dataset_dict[dataset_name][mode]
        lines = utils.read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]
            gt_label = None if len(splits) <= 3 else splits[3]
            sample = dict()

            sample['left_name'] = left_img.split('/', 1)[1]
            sample['right_name'] = right_img.split('/', 1)[1]

            sample['left'] = os.path.join(self.root, left_img)
            sample['right'] = os.path.join(self.root, right_img)
            sample['disp'] = os.path.join(self.root, gt_disp) if gt_disp is not None else None
            sample['label'] = os.path.join(self.root, gt_label) if gt_label is not None else None

            if self.opts.load_pseudo_gt and sample['disp'] is not None:
                # KITTI 2015
                if 'disp_occ_0' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ_0',
                                                                     'disp_occ_0_pseudo_gt')
                else:
                    raise NotImplementedError
            else:
                sample['pseudo_disp'] = None

            self.samples.append(sample)


    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target).astype('uint8')]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def relabel_lostandfound(self, input):
        input = Relabel(0, self.ignore_index)(input)  # ignore background 0-->255
        input = Relabel(1, 0)(input)  # road 1-->0
        input = Relabel(2, 19)(input)  # obstacle 2-->19
        return input

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        sample = {}
        sample_path = self.samples[index]

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        sample['left_name'] = sample_path['left_name']
        sample['right_name'] = sample_path['right_name']

        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]

        if sample_path['pseudo_disp'] is not None:
            sample['pseudo_disp'] = read_disp(sample_path['pseudo_disp'], subset=subset)  # [H, W]

        if sample_path['label'] is not None:
            label = self.encode_target(Image.open(sample_path['label']))
            sample['label'] = label
            sample['label'] = Image.fromarray(label.astype('uint8'))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)


class Relabel(object):
    def __init__(self, olabel, nlabel):  # change trainid label from olabel to nlabel
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        # assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor,
        #                                                            torch.ByteTensor)), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor