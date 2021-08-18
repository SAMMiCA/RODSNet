from __future__ import division
import torch
import math
import sys
import random
from PIL import Image, ImageOps
from PIL import Image as pimg
import cv2

try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

import torchvision.transforms.functional as F
import torch.nn.functional as FF

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


# __all__ = ["Compose", "RandomSquareCropAndScale", "SetTargetSize",
#            "LabelBoundaryTransform", "Tensor", "CropBlackArea", "RandomCrop_PIL",
#            "ColorJitter", "FixedResize"]


_pil_interpolation_to_str = {
    pimg.NEAREST: 'PIL.Image.NEAREST',
    pimg.BILINEAR: 'PIL.Image.BILINEAR',
    pimg.BICUBIC: 'PIL.Image.BICUBIC',
    pimg.LANCZOS: 'PIL.Image.LANCZOS',
    pimg.HAMMING: 'PIL.Image.HAMMING',
    pimg.BOX: 'PIL.Image.BOX',
}


RESAMPLE = pimg.BICUBIC
RESAMPLE_D = pimg.BILINEAR

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))



class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string





class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'




class RandomCrop_PIL(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        ori_width, ori_height = sample['left'].size
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0
            # (left,top, right, bottom)
            sample['left'] = ImageOps.expand(sample['left'], border=(0, top_pad, right_pad, 0), fill=0)
            sample['right'] = ImageOps.expand(sample['right'], border=(0, top_pad, right_pad, 0), fill=0)

            if 'disp' in sample.keys():
                sample['disp'] = ImageOps.expand(sample['disp'], border=(0, top_pad, right_pad, 0), fill=0)

            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = ImageOps.expand(sample['pseudo_disp'], border=(0, top_pad, right_pad, 0), fill=0)

            if 'label' in sample.keys():
                sample['label'] = ImageOps.expand(sample['label'], border=(0, top_pad, right_pad, 0),
                                                        fill=255)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            sample['left'] = self.crop_img(sample['left'])
            sample['right'] = self.crop_img(sample['right'])
            if 'disp' in sample.keys():
                sample['disp'] = self.crop_img(sample['disp'])
            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = self.crop_img(sample['pseudo_disp'])
            if 'label' in sample.keys():
                sample['label'] = self.crop_img(sample['label'])

        return sample

    def crop_img(self, img):
        return img.crop((self.offset_x, self.offset_y,
                         self.offset_x + self.img_width, self.offset_y + self.img_height))


class RandomCrop_PIL2(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        # print(sample['left'].shape)
        _, ori_height, ori_width = sample['left'].shape
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0
            # (left,top, right, bottom)

            sample['left'] = FF.pad(sample['left'], (0, right_pad, top_pad, 0))
            sample['right'] = FF.pad(sample['right'], (0, right_pad, top_pad, 0))

            # sample['left'] = ImageOps.expand(sample['left'], border=(0, top_pad, right_pad, 0), fill=0)
            # sample['right'] = ImageOps.expand(sample['right'], border=(0, top_pad, right_pad, 0), fill=0)

            if 'disp' in sample.keys():
                # sample['disp'] = ImageOps.expand(sample['disp'], border=(0, top_pad, right_pad, 0), fill=0)
                sample['disp'] = FF.pad(sample['disp'], (0, right_pad, top_pad, 0))

            if 'pseudo_disp' in sample.keys():
                # sample['pseudo_disp'] = ImageOps.expand(sample['pseudo_disp'], border=(0, top_pad, right_pad, 0), fill=0)
                sample['pseudo_disp'] = FF.pad(sample['pseudo_disp'], (0, right_pad, top_pad, 0))

            if 'label' in sample.keys():
                # sample['label'] = ImageOps.expand(sample['label'], border=(0, top_pad, right_pad, 0),
                #                                         fill=255)
                sample['label'] = FF.pad(sample['label'], (0, right_pad, top_pad, 0), value=255)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            sample['left'] = self.crop_img(sample['left'])
            sample['right'] = self.crop_img(sample['right'])
            if 'disp' in sample.keys():
                sample['disp'] = self.crop_img(sample['disp'])
            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = self.crop_img(sample['pseudo_disp'])
            if 'label' in sample.keys():
                sample['label'] = self.crop_img(sample['label'])

        return sample

    def crop_img(self, img):
        return img.crop((self.offset_x, self.offset_y,
                         self.offset_x + self.img_width, self.offset_y + self.img_height))


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=pimg.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomSizedCrop(RandomResizedCrop):
    """
    Note: This transform is deprecated in favor of RandomResizedCrop.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.RandomSizedCrop transform is deprecated, " +
                      "please use transforms.RandomResizedCrop instead.")
        super(RandomSizedCrop, self).__init__(*args, **kwargs)



class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        sample['left'] = transform(sample['left'])
        sample['right'] = transform(sample['right'])
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string



class RandomSquareCropAndScale:
    def __init__(self, wh, mean, ignore_id, min=.5, max=2., class_incidence=None, class_instances=None,
                 inst_classes=(3, 12, 14, 15, 16, 17, 18), scale_method=lambda scale, wh, size: int(scale * wh)):
        self.wh = wh
        self.min = min
        self.max = max
        self.mean = mean
        self.ignore_id = ignore_id
        self.random_gens = [self._rand_location]
        self.scale_method = scale_method

    def _rand_location(self, W, H, target_w, target_h, *args, **kwargs):
        try:
            w = np.random.randint(0, W - target_w + 1)
            h = np.random.randint(0, H - target_h + 1)
        except ValueError:
            print(f'Exception in RandomSquareCropAndScale: {target_w}')
            w = h = 0
        # left, upper, right, lower)
        return w, h, w + target_w, h + target_h

    def _trans(self, img: pimg, crop_box, target_size, pad_size, resample, blank_value):
        return crop_and_scale_img(img, crop_box, target_size, pad_size, resample, blank_value)

    def __call__(self, sample):
        left = sample['left']
        right = sample['right']

        scale = np.random.uniform(self.min, self.max)
        W, H = left.size
        box_size_w = self.scale_method(scale, self.wh[0], left.size)
        box_size_h = self.scale_method(scale, self.wh[1], left.size)

        pad_size = (max(box_size_w, W), max(box_size_h, H))
        target_size = (self.wh[0], self.wh[1])

        crop_fn = random.choice(self.random_gens)
        flipped = sample['flipped'] if 'flipped' in sample else False
        crop_box = crop_fn(pad_size[0], pad_size[1], box_size_w, box_size_h, sample['left_name'], flipped)

        ret_dict = {
            'left': self._trans(left, crop_box, target_size, pad_size, RESAMPLE, self.mean),
        }

        ret_dict['right'] = self._trans(right, crop_box, target_size, pad_size, RESAMPLE, self.mean)

        if 'label' in sample:
            ret_dict['label'] = self._trans(sample['label'], crop_box, target_size, pad_size, pimg.NEAREST, self.ignore_id)
        for k in ['image_prev', 'image_next']:
            if k in sample:
                ret_dict[k] = self._trans(sample[k], crop_box, target_size, pad_size, RESAMPLE,
                                          self.mean)
        if 'disp' in sample:
            ret_dict['disp'] = crop_and_scale_disp(sample['disp'], crop_box, target_size, pad_size, RESAMPLE_D, 0)
        if 'pseudo_disp' in sample:
            ret_dict['pseudo_disp'] = crop_and_scale_disp(sample['pseudo_disp'], crop_box, target_size, pad_size, RESAMPLE_D, 0)

        if 'flow' in sample:
            ret_dict['flow'] = crop_and_scale_flow(sample['flow'], crop_box, target_size, pad_size, scale)
        return {**sample, **ret_dict}


def crop_and_scale_img(img: pimg, crop_box, target_size, pad_size, resample, blank_value):
    target = pimg.new(img.mode, pad_size, color=blank_value)
    target.paste(img)
    res = target.crop(crop_box).resize(target_size, resample=resample)
    return res


def crop_and_scale_disp(img: pimg, crop_box, target_size, pad_size, resample, blank_value):
    target = pimg.new(img.mode, pad_size, color=blank_value)
    target.paste(img)
    tmp = target.crop(crop_box)
    # res = tmp.resize(target_size, resample=resample)
    res = tmp.resize(target_size, resample=resample).point(lambda i: i * (target_size[0]/tmp.size[0]))
    return res


def crop_and_scale_flow(flow, crop_box, target_size, pad_size, scale):
    def _trans(uv):
        return crop_and_scale_img(uv, crop_box, target_size, pad_size, resample=pimg.NEAREST, blank_value=0)

    u, v = [pimg.fromarray(uv.squeeze()) for uv in np.split(flow * scale, 2, axis=-1)]
    dtype = flow.dtype
    return np.stack([np.array(_trans(u), dtype=dtype), np.array(_trans(v), dtype=dtype)], axis=-1)


class FixedResize(object):
    def __init__(self, rescale_size):
        self.size = rescale_size  # size: (w, h)

    def __call__(self, sample):
        if 'disp' in sample.keys():
            assert sample['left'].size == sample['disp'].size
            sample['disp'] = sample['disp'].resize(self.size, Image.BILINEAR).point(lambda i: i * (self.size[0] / sample['disp'].width))

        if 'label' in sample.keys():
            assert sample['left'].size == sample['label'].size
            sample['label'] = sample['label'].resize(self.size, Image.NEAREST)

        assert sample['left'].size == sample['right'].size
        sample['left'] = sample['left'].resize(self.size, Image.BILINEAR)
        sample['right'] = sample['right'].resize(self.size, Image.BILINEAR)

        return sample


class SetTargetSize:
    def __init__(self, target_size, target_size_feats, stride=4):
        self.target_size = target_size
        self.target_size_feats = target_size_feats
        self.stride = stride

    def __call__(self, sample):
        if all([self.target_size, self.target_size_feats]):
            sample['target_size'] = self.target_size[::-1]
            sample['target_size_feats'] = self.target_size_feats[::-1]
        else:
            k = 'original_labels' if 'original_labels' in sample else 'left'
            sample['target_size'] = sample[k].shape[-2:]
            sample['target_size_feats'] = tuple([s // self.stride for s in sample[k].shape[-2:]])
        sample['alphas'] = [-1]
        sample['target_level'] = 0
        return sample

class CropBlackArea:
    def __call__(self, sample):

        width, height = sample['left'].size

        left = 140
        top = 30
        right = 2030
        bottom = 900
        # crop
        sample['left'] = sample['left'].crop((left, top, right, bottom))
        sample['right'] = sample['right'].crop((left, top, right, bottom))
        sample['disp'] = sample['disp'].crop((left, top, right, bottom))
        sample['label'] = sample['label'].crop((left, top, right, bottom))

        sample['left'] = sample['left'].resize((width, height), Image.BILINEAR)
        sample['right'] = sample['right'].resize((width, height), Image.BILINEAR)
        sample['disp'] = sample['disp'].resize((width,height), Image.BILINEAR).point(lambda i: i * (width/(right - left)))
        sample['label'] = sample['label'].resize((width,height), Image.NEAREST)

        return sample


class LabelBoundaryTransform:
    def __init__(self, num_classes, reduce=False,
                 ignore_id=255):
        self.num_classes = num_classes
        self.reduce = reduce
        self.ignore_id = ignore_id

    def __call__(self, example):
        labels = np.array(example['label'])
        present_classes = np.unique(labels)
        distances = np.zeros([self.num_classes] + list(labels.shape), dtype=np.float32) - 1.
        for i in range(self.num_classes):
            if i not in present_classes:
                continue
            class_mask = labels == i
            distances[i][class_mask] = cv2.distanceTransform(np.uint8(class_mask), cv2.DIST_L2, maskSize=3)[class_mask]
        if self.reduce:
            ignore_mask = labels == self.ignore_id
            distances[distances < 0] = 0
            distances = distances.sum(axis=0)

            std_d = np.std(distances)
            # var_d = np.var(distances)
            # in lost and found dataset, some image has not any GT labels --> all distances = 0
            if std_d == 0:
                # set std_d = 1, to avoid true_divide error
                # in this case, label_distances set to 0 because all pixels are ignore_mask.
                std_d = 1
            distance_factor = distances / (2 * std_d)
            label_distances = np.exp(-distance_factor)
            label_distances[ignore_mask] = 0

            example['label_distance_weight'] = label_distances
        else:
            example['label_distance_transform'] = distances
        return example


class Tensor:
    def _trans(self, img, dtype):
        img = np.array(img, dtype=dtype)
        if len(img.shape) == 3:
            img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        return torch.from_numpy(img)

    def __call__(self, example):
        ret_dict = {}
        for k in ['left', 'right',  'image_next', 'image_prev']:
            if k in example:
                ret_dict[k] = self._trans(example[k], np.float32)
        if 'disp' in example:
            ret_dict['disp'] = self._trans(example['disp'], np.float32)
        if 'pseudo_disp' in example:
            ret_dict['pseudo_disp'] = self._trans(example['pseudo_disp'], np.float32)

        if 'label' in example:
            ret_dict['label'] = self._trans(example['label'], np.int64)

        if 'original_labels' in example:
            ret_dict['original_labels'] = self._trans(example['original_labels'], np.int64)
        if 'depth_hist' in example:
            ret_dict['depth_hist'] = [self._trans(d, np.float32) for d in example['depth_hist']] if isinstance(
                example['depth_hist'], list) else self._trans(example['depth_hist'], np.float32)
        if 'pyramid' in example:
            ret_dict['pyramid'] = [self._trans(p, np.float32) for p in example['pyramid']]
        if 'pyramid_ms' in example:
            ret_dict['pyramid_ms'] = [[self._trans(p, np.float32) for p in pyramids] for pyramids in
                                      example['pyramid_ms']]
        if 'mux_indices' in example:
            ret_dict['mux_indices'] = torch.stack([torch.from_numpy(midx.flatten()) for midx in example['mux_indices']])
        if 'mux_masks' in example:
            ret_dict['mux_masks'] = [torch.from_numpy(np.uint8(mi)).unsqueeze(0) for mi in example['mux_masks']]
        if 'depth_bins' in example:
            ret_dict['depth_bins'] = torch.stack([torch.from_numpy(b) for b in example['depth_bins']])
        if 'flow' in example:
            # ret_dict['flow'] = torch.from_numpy(example['flow']).permute(2, 0, 1).contiguous()
            ret_dict['flow'] = torch.from_numpy(np.ascontiguousarray(example['flow']))
        # if 'flow_next' in example:
        #     ret_dict['flow_next'] = torch.from_numpy(example['flow_next']).permute(2, 0, 1 ).contiguous()
        if 'flow_sub' in example:
            # ret_dict['flow_sub'] = torch.from_numpy(example['flow_sub']).permute(2, 0, 1).contiguous()
            ret_dict['flow_sub'] = torch.from_numpy(np.ascontiguousarray(example['flow_sub']))
        if 'flipped' in example:
            del example['flipped']
        return {**example, **ret_dict}


class ToTensor(object):
    """Convert Image object in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # ret_dict = {}
        # for k in ['left', 'right']:
        #     if k in sample:
        #         ret_dict[k] = self._trans(sample[k], np.float32)

        sample['left'] = np.array(sample['left']).astype(np.float32).transpose((2, 0, 1))
        sample['left'] = torch.from_numpy(sample['left']).float()
        sample['right'] = np.array(sample['right']).astype(np.float32).transpose((2, 0, 1))
        sample['right'] = torch.from_numpy(sample['right']).float()

        if 'disp' in sample.keys():
            sample['disp'] = np.array(sample['disp']).astype(np.float32)
            sample['disp'] = torch.from_numpy(sample['disp']).float()
        if 'label' in sample.keys():
            sample['label'] = np.array(sample['label']).astype(np.float32)
            sample['label'] = torch.from_numpy(sample['label']).float()

        return sample