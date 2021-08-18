import torch
from torch import nn
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
from network.deform import DeformConv2d
from network.utils import _UpsampleBlend
from itertools import chain


__all__ = ['MobileNetV2', 'mobilenetv2_pyramid']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def convkxk(in_planes, out_planes, stride=1, k=3):
    """kxk convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=stride, padding=k // 2, bias=False)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        #padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, 0, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

def fixed_padding(kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end) 

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

        self.input_padding = fixed_padding( 3, dilation )

    def forward(self, x):
        x_pad = F.pad(x, self.input_padding)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)

class InvertedResidualDeform(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(InvertedResidualDeform, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),
            # pw-linear
            DeformConv2d(hidden_dim, oup, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

        self.input_padding = fixed_padding( 3, dilation )

    def forward(self, x):
        x_pad = F.pad(x, self.input_padding)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class MobileNetV2(nn.Module):
    def __init__(self, output_stride=16, width_mult=1.0, inverted_residual_setting=None, round_nearest=8,
                 ):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32

        pyramid_levels = 3
        self.pyramid_levels = 3
        self.num_features = 128
        self.pyramid_subsample = 'bicubic'
        self.align_corners = None
        mean = (73.1584, 82.9090, 72.3924)
        std = (44.9149, 46.1529, 45.3192)
        use_bn = True
        scale = 1
        k_bneck = 1
        num_features = 128

        self.register_buffer('img_mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('img_std', torch.tensor(std).view(1, -1, 1, 1))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        bn_class = nn.BatchNorm2d if use_bn else Identity
        self.register_buffer('img_mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('img_std', torch.tensor(std).view(1, -1, 1, 1))
        if scale != 1:
            self.register_buffer('img_scale', torch.tensor(scale).view(1, -1, 1, 1).float())

        self.output_stride = output_stride
        current_stride = 1
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        # input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        # features = [ConvBNReLU(3, input_channel, stride=2)]
        current_stride *= 2
        dilation=1
        previous_dilation = 1

        features = [ConvBNReLU(3, 32, stride=2)]
        # building inverted residual blocks
        for idx, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            previous_dilation = dilation
            if current_stride == output_stride:
                stride = 1
                dilation *= s
            else:
                stride = s
                current_stride *= s
            output_channel = int(c * width_mult)

            for i in range(n):
                if i==0:
                    features.append(block(input_channel, output_channel, stride, previous_dilation, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, dilation, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, 1280, kernel_size=1))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 1000),
        )

        self.bn1 = nn.ModuleList([bn_class(32) for _ in range(pyramid_levels)])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fine_tune = [self.conv1, self.maxpool, self.features, self.bn1]


        bottlenecks = []
        bottlenecks += [convkxk(16, num_features, k=k_bneck)]
        bottlenecks += [convkxk(24, num_features, k=k_bneck)]
        bottlenecks += [convkxk(32, num_features, k=k_bneck)]
        bottlenecks += [convkxk(320, num_features, k=k_bneck)]

        self.num_skip_levels = self.pyramid_levels + 3
        self.upsample_bottlenecks = nn.ModuleList(bottlenecks[::-1])

        num_pyr_modules = 2 + self.pyramid_levels
        target_sizes = [None] * num_pyr_modules
        self.target_size = None
        k_upsample = 3

        detach_upsample_skips = ()
        self.upsample_blends = nn.ModuleList(
            [_UpsampleBlend(self.num_features,
                            use_bn=True,
                            use_skip=True,
                            detach_skip=i in detach_upsample_skips,
                            fixed_size=ts,
                            k=k_upsample)
             for i, ts in enumerate(target_sizes)])

        self.upsample_blends_disp = nn.ModuleList(
            [_UpsampleBlend(self.num_features,
                            use_bn=True,
                            use_skip=True,
                            detach_skip=i in detach_upsample_skips,
                            fixed_size=ts,
                            k=k_upsample)
             for i, ts in enumerate(target_sizes)])

        self.random_init = [self.upsample_bottlenecks, self.upsample_blends, self.upsample_blends_disp]

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_down(self, image, skips, idx=-1):
        x = self.conv1(image)
        x = self.bn1[idx](x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 32, H/4xW/4]

        features = []
        x = self.features[1](x) # [B, 16, H/4xW/4]
        features += [x]
        x = self.features[2:4](x)  # [B, 24, H/8xW/8]
        features += [x]
        x = self.features[4:7](x)  # [B, 32, H/16xW/16]
        features += [x]
        x = self.features[7:-1](x)  # [B, 320, H/32xW/32]
        features += [x]

        skip_feats = [b(f) for b, f in zip(self.upsample_bottlenecks, reversed(features))]

        for i, s in enumerate(reversed(skip_feats)):
            skips[idx + i] += [s]

        return skips


    def forward(self, image):
        if isinstance(self.bn1[0], nn.BatchNorm2d):
            if hasattr(self, 'img_scale'):
                image /= self.img_scale

            image -= self.img_mean
            image /= self.img_std
        pyramid = [image]
        for l in range(1, self.pyramid_levels):
            if self.target_size is not None:
                ts = list([si // 2 ** l for si in self.target_size])
                pyramid += [
                    F.interpolate(image, size=ts, mode=self.pyramid_subsample, align_corners=self.align_corners)]
            else:
                pyramid += [F.interpolate(image, scale_factor=1 / 2 ** l, mode=self.pyramid_subsample,
                                          align_corners=self.align_corners)]
        skips = [[] for _ in range(self.num_skip_levels)]
        additional = {'pyramid': pyramid}
        for idx, p in enumerate(pyramid):
            skips = self.forward_down(p, skips, idx=idx)
        skips = skips[::-1]
        x = skips[0][0]

        for i, (sk, blend) in enumerate(zip(skips[1:], self.upsample_blends)):
            x = blend(x, sum(sk))

        ## for disparity features
        x_disp = []
        x_d = skips[0][0]
        for i, (sk, blend) in enumerate(zip(skips[1:], self.upsample_blends_disp)):
            x_d = blend(x_d, sum(sk))
            x_disp += [x_d]
        x_disp = x_disp[::-1]

        return x, x_disp, additional



def mobilenetv2_pyramid(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        # model.load_state_dict(state_dict)

        loaded_pt = state_dict
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in loaded_pt.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict, strict=False)


    return model
