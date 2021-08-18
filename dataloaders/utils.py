import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

from dataloaders.datasets import Cityscapes, CityLostFound
from dataloaders import custom_transforms as sw

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    scale = 1
    mean = [73.15, 82.90, 72.3]
    std = [47.67, 48.49, 47.73]
    mean_rgb = tuple(np.uint8(scale * np.array(mean)))

    if opts.dataset == 'cityscapes':
        random_crop_size = (768, 768)

        target_size_crops = (random_crop_size[0], random_crop_size[1])
        target_size_crops_feats = (random_crop_size[0] // 4, random_crop_size[1] // 4)
        target_size = (opts.val_img_width, opts.val_img_height)

        train_transform = sw.Compose(
            [
             sw.RandomSquareCropAndScale(random_crop_size, ignore_id=255, mean=mean_rgb),
             sw.SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
             sw.LabelBoundaryTransform(num_classes=opts.num_classes, reduce=True),
             sw.Tensor(),
             ]
        )

        val_transform = sw.Compose(
            [sw.FixedResize(target_size),
             sw.Tensor(),
             ]
        )

        train_dst = Cityscapes(root=opts.data_root, dataset_name=opts.dataset,
                               mode='train', transform=train_transform, opts=opts)
        val_dst = Cityscapes(root=opts.data_root, dataset_name=opts.dataset,
                             mode='val', transform=val_transform, opts=opts)

    elif opts.dataset == 'city_lost':
        random_crop_size = (768, 768)

        target_size_crops = (random_crop_size[0], random_crop_size[1])
        target_size_crops_feats = (random_crop_size[0] // 4, random_crop_size[1] // 4)
        target_size = (opts.val_img_width, opts.val_img_height)

        train_transform = sw.Compose(
            [
                sw.CropBlackArea(),
                sw.RandomSquareCropAndScale(random_crop_size, ignore_id=255, mean=mean_rgb),
                sw.SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
                sw.LabelBoundaryTransform(num_classes=opts.num_classes, reduce=True),
                sw.Tensor(),
            ]
        )

        val_transform = sw.Compose(
            [sw.CropBlackArea(),
             sw.FixedResize(target_size),
             sw.Tensor(),
             ]
        )

        train_dst = CityLostFound(root=opts.data_root, dataset_name=opts.dataset,
                                  mode='train', transform=train_transform)
        val_dst = CityLostFound(root=opts.data_root, dataset_name=opts.dataset,
                                mode='val', transform=val_transform)

    elif opts.dataset == 'kitti_2015':
        # size of original input image : (1242, 375)
        random_crop_size = (896, 256)
        target_size_crops = random_crop_size
        target_size_crops_feats = (random_crop_size[0] // 4, random_crop_size[1] // 4)

        train_transform = transforms.Compose(
            [
                sw.RandomCrop_PIL(256, 896),
                sw.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                sw.SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
                sw.LabelBoundaryTransform(num_classes=opts.num_classes, reduce=True),
                sw.Tensor(),
            ]
        )
        val_transform = transforms.Compose(
            [
                sw.RandomCrop_PIL(384, 1280, validate=True),
                sw.Tensor(),
             ]
        )

        train_dst = Cityscapes(root=opts.data_root, dataset_name=opts.dataset,
                               mode='train', transform=train_transform, opts=opts)
        val_dst = Cityscapes(root=opts.data_root, dataset_name=opts.dataset,
                             mode='val', transform=val_transform, opts=opts)

    elif opts.dataset == 'kitti_mix':
        random_crop_size = (1152, 256)
        target_size_crops = random_crop_size
        target_size_crops_feats = (random_crop_size[0] // 4, random_crop_size[1] // 4)

        train_transform = transforms.Compose(
            [
                sw.RandomCrop_PIL(256, 1152),
                sw.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                sw.SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
                sw.Tensor(),
            ]
        )

        val_transform = transforms.Compose(
            [
                sw.RandomCrop_PIL(384, 1280, validate=True),
                sw.Tensor(),
             ]
        )

        train_dst = Cityscapes(root=opts.data_root, dataset_name=opts.dataset,
                               mode='train', transform=train_transform, opts=opts)
        val_dst = Cityscapes(root=opts.data_root, dataset_name=opts.dataset,
                             mode='val', transform=val_transform, opts=opts)

    elif opts.dataset == 'sceneflow':
        # origin size : 960x540
        ## random_crop : 384x384

        random_crop_size = (384, 384)
        scale = 1
        mean = [73.15, 82.90, 72.3]
        std = [47.67, 48.49, 47.73]
        mean_rgb = tuple(np.uint8(scale * np.array(mean)))

        target_size_crops = (random_crop_size[0], random_crop_size[1])
        target_size_crops_feats = (random_crop_size[0] // 4, random_crop_size[1] // 4)

        train_transform = sw.Compose(
            [
                sw.RandomSquareCropAndScale(random_crop_size, ignore_id=255, mean=mean_rgb),
                sw.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                sw.SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
                sw.Tensor(),
            ]
        )

        val_transform = sw.Compose(
            [
             sw.FixedResize((896, 512)),
             sw.Tensor(),
             ]
        )

        train_dst = Cityscapes(root=opts.data_root, dataset_name=opts.dataset,
                               mode='train', transform=train_transform, opts=opts)
        val_dst = Cityscapes(root=opts.data_root, dataset_name=opts.dataset,
                             mode='val', transform=val_transform, opts=opts)

    else:
        print("Wrong dataset was typed...")
        raise ValueError

    return train_dst, val_dst


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))  # change for val
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'citylostfound':
        n_classes = 20
        label_colours = get_citylostfound_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))  # change for val
    # rgb = torch.ByteTensor(3, label_mask.shape[0], label_mask.shape[1]).fill_(0)
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    # r = torch.from_numpy(r)
    # g = torch.from_numpy(g)
    # b = torch.from_numpy(b)

    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_citylostfound_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [111, 74,  0]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

def custom_collate(batch, del_orig_labels=False):
    keys = ['target_size', 'target_size_feats', 'alphas', 'target_level']
    values = {}
    for k in keys:
        if k in batch[0]:
            values[k] = batch[0][k]
    for b in batch:
        if del_orig_labels: del b['original_labels']
        for k in values.keys():
            del b[k]
        if 'mux_indices' in b:
            b['mux_indices'] = b['mux_indices'].view(-1)

    batch = default_collate(batch)

    # if 'image_next' in batch:
    #     batch['image'] = torch.cat([batch['image'], batch['image_next']], dim=0).contiguous()
    #     del batch['image_next']

    for k, v in values.items():
        batch[k] = v
    return batch

def colormap_bdd(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64, 128])
    cmap[1,:] = np.array([244, 35, 232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([102, 102, 156])
    cmap[4,:] = np.array([190, 153, 153])
    cmap[5,:] = np.array([153, 153, 153])

    cmap[6,:] = np.array([250, 170, 30])
    cmap[7,:] = np.array([220, 220, 0])
    cmap[8,:] = np.array([107, 142, 35])
    cmap[9,:] = np.array([152, 251, 152])
    cmap[10,:]= np.array([70, 130, 180])

    cmap[11,:]= np.array([220, 20, 60])
    cmap[12,:]= np.array([255, 0, 0])
    cmap[13,:]= np.array([0, 0, 142])
    cmap[14,:]= np.array([0, 0, 70])
    cmap[15,:]= np.array([0, 60, 100])

    cmap[16,:]= np.array([0, 80, 100])
    cmap[17,:]= np.array([0, 0, 230])
    cmap[18,:]= np.array([119, 11, 32])
    cmap[19,:]= np.array([111, 74, 0]) #多加了一类small obstacle

    return cmap

class Colorize:

    def __init__(self, n=20): # n = nClasses
        # self.cmap = colormap(256)
        self.cmap = colormap_bdd(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        # print(size)
        color_images = torch.ByteTensor(size[0], 3, size[1], size[2]).fill_(0)
        # color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        # for label in range(1, len(self.cmap)):
        for i in range(color_images.shape[0]):
            for label in range(0, len(self.cmap)):
                mask = gray_image[0] == label
                # mask = gray_image == label

                color_images[i][0][mask] = self.cmap[label][0]
                color_images[i][1][mask] = self.cmap[label][1]
                color_images[i][2][mask] = self.cmap[label][2]

        return color_images
