from __future__ import absolute_import, division, print_function

from options import Options

options = Options()
opts = options.parse()
from trainer import *


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.nn.parameter import Parameter

from dataloaders.utils import get_dataset, custom_collate
from metrics import Evaluator, TimeAverageMeter

import utils


import random
import numpy as np
import os
import network
from dataloaders.datasets import Cityscapes, CityLostFound
from dataloaders import custom_transforms as sw
import skimage.io
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from collections import namedtuple


CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])
classes = [
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([111, 74, 0])  # small obstacle
train_id_to_color.append([0, 0, 0])     # background
train_id_to_color = np.array(train_id_to_color)
id_to_train_id = np.array([c.train_id for c in classes])

num_classes = 20    # 19 cityscapes classes + small obstacle objects


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing


def file_to_tensor(filename):
    pil_img = Image.open(filename).convert('RGB')
    img = np.array(pil_img, dtype=np.float32)
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    return pil_img, torch.from_numpy(img)


def decode_target(target):
    target[target == 255] = num_classes
    # target = target.astype('uint8') + 1
    return train_id_to_color[target]


# for submit to KITTI 2015 test benchmark
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda:{}'.format(opts.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    n_gpus = len(opts.gpu_id)
    print("Number of used GPU : {}".format(n_gpus))
    print("Used GPU ID : {}".format(opts.gpu_id))

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    torch.backends.cudnn.benchmark = True

    opts.data_root = os.path.join(opts.data_root, opts.dataset)
    opts.num_classes = num_classes

    model = network.RODSNet(opts,
                             opts.max_disp,
                             num_classes=opts.num_classes,
                             device=device,
                             refinement_type=opts.refinement_type,
                             )
    model.to(device)

    evaluator = Evaluator(opts.num_classes)

    if opts.resume is not None:
        if not os.path.isfile(opts.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(opts.resume))
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.resume, map_location=device)

        loaded_pt = checkpoint['model_state']
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in loaded_pt.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict, strict=False)
    else:
        print("[!] No checkpoints found...")
        raise NotImplementedError

    num_params = utils.count_parameters(model)
    print('=> Number of trainable parameters: %d' % num_params)

    # Inference
    model.eval()
    inference_time = 0
    sample_ID = [0, 1, 2, 3, 4, 5]

    for ID in sample_ID:
        left_filename = 'samples/left/' + str(ID) + '_left.png'
        right_filename = 'samples/right/' + str(ID) + '_right.png'

        pil_left, left = file_to_tensor(left_filename)
        pil_right, right = file_to_tensor(right_filename)

        left = left.to(device, dtype=torch.float32)
        right = right.to(device, dtype=torch.float32)

        left = left.unsqueeze(0)
        right = right.unsqueeze(0)

        # Pad
        ori_height, ori_width = left.size()[2:]
        # set height and width sizes to divided by 128
        if ori_height % 128 != 0:
            val_height = ((ori_height // 128) +1)*128
        else:
            val_height = ori_height
        if ori_width % 128 != 0:
            val_width = ((ori_width // 128) + 1)*128
        else:
            val_width = ori_width

        if ori_height < val_height or ori_width < val_width:
            top_pad = opts.val_img_height - ori_height
            right_pad = opts.val_img_width - ori_width

            # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
            left = F.pad(left, (0, right_pad, top_pad, 0))
            right = F.pad(right, (0, right_pad, top_pad, 0))


        with torch.no_grad():
            pred_disp_pyramid, left_seg = model(left, right)
            pred_disp = pred_disp_pyramid[-1]

        image = left[0].detach().cpu().numpy()
        right_image = right[0].detach().cpu().numpy()
        preds = left_seg.detach().max(dim=1)[1].cpu().numpy()

        # Crop
        if ori_height < opts.val_img_height or ori_width < opts.val_img_width:
            if right_pad != 0:
                pred_disp = pred_disp[:, top_pad:, :-right_pad]
                image = image[:, top_pad:, :-right_pad]
                right_image = right_image[:, top_pad:, :-right_pad]
                preds = preds[:, top_pad:, :-right_pad]

            else:
                pred_disp = pred_disp[:, top_pad:]
                image = image[:, top_pad:]
                right_image = right_image[:, top_pad:]
                preds = preds[:, top_pad:]


        disp = pred_disp[0].detach().cpu().numpy()  # [H, W]
        save_name = 'results/' + str(ID) + '_disp.png'
        save_name_disp = os.path.join('samples', save_name)
        check_path(os.path.dirname(save_name_disp))
        plt.imsave(save_name_disp, disp, cmap='magma')

        pred = preds[0]
        pred = decode_target(pred).astype(np.uint8)
        pred = Image.fromarray(pred)

        overlay = Image.blend(pil_left, pred, alpha=0.7)
        save_name = 'results/' + str(ID) + '_overlay.png'
        save_name_sem = os.path.join('samples', save_name)
        overlay.save(save_name_sem)
