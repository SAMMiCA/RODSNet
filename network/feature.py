import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

from network.deform import DeformConv2d

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


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
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1):
        #padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class ConvBNReLUDeconv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1):
        #padding = (kernel_size - 1) // 2
        super(ConvBNReLUDeconv, self).__init__(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding=padding, dilation=dilation, groups=groups, bias=False),
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


class InvertedResidualDeconv(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(InvertedResidualDeconv, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(oup * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        else:
            layers.append(ConvBNReLU(inp, oup, kernel_size=1))

        if stride > 1:
            layers.extend([
                # dw
                ConvBNReLUDeconv(hidden_dim, hidden_dim, kernel_size=4, stride=stride,
                                 dilation=dilation, padding=1, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
        else:
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation,groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            self.input_padding = fixed_padding(3, dilation)
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.stride > 1:
            x_pad = x
        else:
            x_pad = F.pad(x, self.input_padding)

        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)


class InvertedResidualDeform(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, deform=False):
        super(InvertedResidualDeform, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        if deform:
            Conv_layer = DeformConv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        else:
            Conv_layer = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),
            # pw-linear
            Conv_layer,
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


class FeatureExtractor(nn.Module):
    """Height and width need to be divided by 48, downsampled by 1/3"""

    def __init__(self, pretrained=False, progress=True, output_stride=16, **kwargs):
        super(FeatureExtractor, self).__init__()

        """ 
        MobileNet V2 main class + Deformable Convolution

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """

        # self.backbone = MobileNetV2(output_stride=output_stride, **kwargs)
        self.backbone = MobileNetV2_New(output_stride=output_stride, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                  progress=progress)
            self.backbone.load_state_dict(state_dict)

    def forward(self, x):
        # feature extraction
        x1 = self.backbone.features[0:2](x)      # ch, size : (3, HxW) => (16, H/2xW/2)
        x2 = self.backbone.features[2:4](x1)     # ch, size : (16, H/2xW/2) => (24, H/4xW/4)     ,low_level features
        x3 = self.backbone.features[4:7](x2)     # ch, size : (24, H/4xW/4) => (32, H/8xW/8)
        x4 = self.backbone.features[7:11](x3)    # ch, size : (32, H/8xW/8) => (64, H/16xW/16)
        x5 = self.backbone.features[11:18](x4)   # ch, size : (64, H/16xW/16) => (320, H/16xW/16) ,high_level_features

        return [x, x1, x2, x3, x4, x5]


class MobileNetV2(nn.Module):
    def __init__(self, start_channel=3, num_classes=1000, output_stride=8, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
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
        last_channel = 1280
        self.output_stride = output_stride
        current_stride = 1
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],      # features[1:2]     # ch, size : (32, H/2xW/2)   => (16, H/2xW/2)
                [6, 24, 2, 2],      # features[2:4]     # ch, size : (16, H/2xW/2)   => (24, H/4xW/4)
                [6, 32, 3, 2],      # features[4:7]     # ch, size : (24, H/4xW/4)   => (32, H/8xW/8)
                [6, 64, 4, 2],      # features[7:11]    # ch, size : (32, H/8xW/8)   => (64, H/16xW/16)
                [6, 96, 3, 1],      # features[11:14]   # ch, size : (64, H/16xW/16) => (96, H/16xW/16)
                [6, 160, 3, 2],     # features[14:17]   # ch, size : (96, H/16xW/16) => (160, H/16xW/16)
                [6, 320, 1, 1],     # features[17]      # ch, size : (160,H/16xW/16) => (320, H/16xW/16)
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # features[0]   # ch, size : (3, HxW) => (32, H/2xW/2)
        if start_channel == 3:
            features = [ConvBNReLU(start_channel, input_channel, stride=2, padding=1)]
        else:
            features = []

        current_stride *= 2
        dilation=1
        previous_dilation = 1

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
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
        # building last several layers
        # features[18]   # ch, size : (320, H/16xW/16) => (1280, H/16xW/16)
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


class MobileNetV2_New(nn.Module):
    def __init__(self, num_classes=1000, output_stride=8, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2_New, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.output_stride = output_stride
        current_stride = 1
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],      # features[1:2]     # ch, size : (32, H/2xW/2)   => (16, H/2xW/2)
                [6, 24, 2, 2],      # features[2:4]     # ch, size : (16, H/2xW/2)   => (24, H/4xW/4)
                [6, 32, 3, 2],      # features[4:7]     # ch, size : (24, H/4xW/4)   => (32, H/8xW/8)
                [6, 64, 4, 2],      # features[7:11]    # ch, size : (32, H/8xW/8)   => (64, H/16xW/16)
                [6, 96, 3, 1],      # features[11:14]   # ch, size : (64, H/16xW/16) => (96, H/16xW/16)
                [6, 160, 3, 2],     # features[14:17]   # ch, size : (96, H/16xW/16) => (160, H/16xW/16)
                [6, 320, 1, 1],     # features[17]      # ch, size : (160,H/16xW/16) => (320, H/16xW/16)
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # features[0]   # ch, size : (3, HxW) => (32, H/2xW/2)
        features = [ConvBNReLU(3, input_channel, stride=2, padding=1)]
        current_stride *= 2
        dilation = 1
        previous_dilation = 1

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
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
        # building last several layers
        # features[18]   # ch, size : (320, H/16xW/16) => (1280, H/16xW/16)
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


class MobileHourglass(nn.Module):
    def __init__(self, num_classes=1000, output_stride=8, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileHourglass, self).__init__()
        block = InvertedResidualDeconv
        input_channel = 320
        last_channel = 1280
        self.output_stride = output_stride
        current_stride = 1
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, d     # expand_ratio, output_channel, num_iter, stride, dilation
                [6, 160, 1, 1, 2],      # features[0]        # ch, size : (160, H/16xW/16) <= (320, H/16xW/16)
                [6, 96,  3, 1, 2],      # features[1:4]      # ch, size : (96,H/16xW/16)   <= (160, H/16xW/16)
                [6, 64,  3, 1, 1],       # features[4:7]     # ch, size : (64, H/16xW/16)  <= (96, H/16xW/16)
                [6, 32,  4, 2, 1],       # features[7:11]    # ch, size : (32, H/8xW/8)    <= (64, H/16xW/16)
                [6, 24,  3, 2, 1],       # features[11:14]   # ch, size : (24, H/4xW/4)    <= (32, H/8xW/8)
                [6, 16,  2, 2, 1],       # features[14:16]   # ch, size : (16, H/2xW/2)    <= (24, H/4xW/4)
                [1, 32,  1, 1, 1],       # features[16]      # ch, size : (32, H/2xW/2)    <= (16, H/2xW/2)
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 5-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # features[0]   # ch, size : (3, HxW) => (32, H/2xW/2)
        features = []

        # building inverted residual blocks
        for idx, (t, c, n, s, d) in enumerate(inverted_residual_setting):
            for i in range(n):
                if i == n-1:
                    output_channel = int(c * width_mult)
                    if n == 1:
                        features.append(block(input_channel, output_channel, s, d, expand_ratio=t))
                    else:
                        features.append(block(input_channel, output_channel, s, 1, expand_ratio=t))
                else:
                    output_channel = input_channel
                    features.append(block(input_channel, output_channel, 1, d, expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        # features[18]   # ch, size : (320, H/16xW/16) => (1280, H/16xW/16)
        # features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, rem):
        x = self.features(x, rem)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
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
        model.load_state_dict(state_dict)
    return model



class FeatureExtractor2(nn.Module):
    """Height and width need to be divided by 48, downsampled by 1/3"""

    def __init__(self, pretrained=False, progress=True, output_stride=16, **kwargs):
        super(FeatureExtractor2, self).__init__()

        """ 
        MobileNet V2 main class + Deformable Convolution

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """

        # self.backbone = MobileNetV2(output_stride=output_stride, **kwargs)
        self.backbone = MobileNetV2(output_stride=output_stride, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                  progress=progress)
            self.backbone.load_state_dict(state_dict)

        self.backbone_decoder = MobileHourglass(output_stride=output_stride, **kwargs)

        self.deconv1 = BasicConv(32 * 2, 32, False, is_3d=False, bn=True, relu=True, kernel_size=3, stride=1, padding=1)
        self.deconv2 = BasicConv(24 * 2, 24, False, is_3d=False, bn=True, relu=True, kernel_size=3, stride=1, padding=1)
        self.deconv3 = BasicConv(16 * 2, 16, False, is_3d=False, bn=True, relu=True, kernel_size=3, stride=1, padding=1)


        # self.backbone2 = MobileNetV2(start_channel=32, output_stride=output_stride, **kwargs)
        # # don't use pretrained results on this backbone (for disparity feature)
        # self.backbone_decoder2 = MobileHourglass(output_stride=output_stride, **kwargs)
        #
        # self.deconv1_2 = BasicConv(32 * 2, 32, False, is_3d=False, bn=True, relu=True, kernel_size=3, stride=1, padding=1)
        # self.deconv2_2 = BasicConv(24 * 2, 24, False, is_3d=False, bn=True, relu=True, kernel_size=3, stride=1, padding=1)
        # self.deconv3_2 = BasicConv(16 * 2, 16, False, is_3d=False, bn=True, relu=True, kernel_size=3, stride=1, padding=1)

        self.disp_conv_start = nn.Sequential(
                BasicConv(32, 32, kernel_size=3, padding=1),
                BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
                BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv1b = Conv2x(16, 24)
        self.conv2b = Conv2x(24, 32)
        self.conv3b = Conv2x(32, 64)
        # self.conv4b = Conv2x(64, 128)

        # self.deconv4b = Conv2x(128, 64, deconv=True)
        self.deconv3b = Conv2x(64, 32, deconv=True)
        self.deconv2b = Conv2x(32, 24, deconv=True)
        self.deconv1b = Conv2x(24, 16, deconv=True)

        # self.conv1a = BasicConv(16, 24, kernel_size=3, stride=2, padding=1)
        # self.conv2a = BasicConv(24, 32, kernel_size=3, stride=2, padding=1)
        # self.conv3a = BasicConv(32, 64, kernel_size=3, stride=2, padding=1)
        # self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)
        #
        # self.deconv4a = Conv2x(128, 96, deconv=True)
        # self.deconv3a = Conv2x(96, 64, deconv=True)
        # self.deconv2a = Conv2x(64, 48, deconv=True)
        # self.deconv1a = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        # feature extraction
        input = x

        x1 = self.backbone.features[0:2](x)      # ch, size : (3, HxW)        => (16, H/2xW/2)
        x2 = self.backbone.features[2:4](x1)     # ch, size : (16, H/2xW/2)   => (24, H/4xW/4)     ,low_level features
        x3 = self.backbone.features[4:7](x2)     # ch, size : (24, H/4xW/4)   => (32, H/8xW/8)
        x4 = self.backbone.features[7:11](x3)    # ch, size : (32, H/8xW/8)   => (64, H/16xW/16)
        x5 = self.backbone.features[11:18](x4)   # ch, size : (64, H/16xW/16) => (320, H/16xW/16) ,high_level_features

        x5_d = self.backbone_decoder.features[0:7](x5)  # (320, H/16xW/16) --> (64, H/16xW/16)
        x4_d = self.backbone_decoder.features[7:11](x5_d)  # (64, H/16xW/16) --> (32, H/8xW/8)

        assert (x4_d.size() == x3.size())
        x4_d = torch.cat((x4_d, x3), 1)
        x4_d = self.deconv1(x4_d)

        x3_d = self.backbone_decoder.features[11:14](x4_d)  # (32, H/8xW/8) --> (24, H/4xW/4)
        assert (x3_d.size() == x2.size())
        x3_d = torch.cat((x3_d, x2), 1)
        x3_d = self.deconv2(x3_d)

        x2_d = self.backbone_decoder.features[14:16](x3_d)  # (24, H/4xW/4) --> (16, H/2xW/2)
        assert (x2_d.size() == x1.size())
        x2_d = torch.cat((x2_d, x1), 1)
        x2_d = self.deconv3(x2_d)

        # x1_d = self.backbone_decoder.features[16](x2_d)     # (16, H/2xW/2)  --> (32, H/2xW/2)
        #
        # # (32, H/2xW/2)  --> (32, HxW)
        # x0_dsh = F.interpolate(x1_d, size=input_shape,
        #                      mode='bilinear', align_corners=False) * (x.size(-1) / x1_d.size(-1))
        #
        # x = self.disp_conv_start(x0_dsh)

        x1_b = self.conv1b(x2_d, x3_d)     # (16, H/2xW/2) --> (24, H/4xW/4)
        x2_b = self.conv2b(x1_b, x4_d)        # (24, H/4xW/4) --> (32, H/8xW/8)
        x3_b = self.conv3b(x2_b, x5_d)        # (32, H/8xW/8) --> (64, H/16xW/16)


        x3_bd = self.deconv3b(x3_b, x2_b)      # (64, H/16xW/16) --> (32, H/8xW/8)
        x2_bd = self.deconv2b(x3_bd, x1_b)      # (32, H/8xW/8) --> (24, H/4xW/4)
        x1_bd = self.deconv1b(x2_bd, x2_d)      # (24, H/4xW/4) --> (16, H/2xW/2)

        x1_d = self.backbone_decoder.features[16](x1_bd)     # (16, H/2xW/2)  --> (32, H/2xW/2)

        # (32, H/2xW/2)  --> (32, HxW)
        x0_dsh = F.interpolate(x1_d, size=input.shape[-2:],
                             mode='bilinear', align_corners=False) * (input.size(-1) / x1_d.size(-1))

        x_out = self.disp_conv_start(x0_dsh)    # (32, HxW) --> (32, H/3xW/3)

        # rem0 = x
        # x = self.conv1a(x)
        # rem1 = x
        # x = self.conv2a(x)
        # rem2 = x
        # x = self.conv3a(x)
        # rem3 = x
        # x = self.conv4a(x)
        # rem4 = x
        #
        # x = self.deconv4a(x, rem3)
        # rem3 = x
        # x = self.deconv3a(x, rem2)
        # rem2 = x
        # x = self.deconv2a(x, rem1)
        # rem1 = x
        # x = self.deconv1a(x, rem0)  # [B, 32, H/3, W/3]
        # rem0 = x

        # x = self.conv1b(x, rem1)
        # rem1 = x
        # x = self.conv2b(x, rem2)
        # rem2 = x
        # x = self.conv3b(x, rem3)
        # rem3 = x
        # x = self.conv4b(x, rem4)
        #
        # x = self.deconv4b(x, rem3)
        # x = self.deconv3b(x, rem2)
        # x = self.deconv2b(x, rem1)
        # x = self.deconv1b(x, rem0)  # [B, 32, H/3, W/3]

        # x5_ = self.backbone_decoder.features[0](x5)         # (320, H/16xW/16) --> (160, H/16xW/16)
        # x5_2 = self.backbone_decoder.features[1:4](x5_)     # (160, H/16xW/16) --> (96, H/16xW/16)
        # x5_d = self.backbone_decoder.features[4:7](x5_2)    # (96, H/16xW/16)  --> (64, H/16xW/16)
        # x4_d = self.backbone_decoder.features[7:11](x5_d)   # (64, H/16xW/16)  --> (32, H/8xW/8)
        # x3_d = self.backbone_decoder.features[11:14](x4_d)  # (32, H/8xW/8)  --> (24, H/4xW/4)
        # x2_d = self.backbone_decoder.features[14:16](x3_d)  # (24, H/4xW/4)  --> (16, H/2xW/2)
        # x1_d = self.backbone_decoder.features[16](x2_d)     # (16, H/2xW/2)  --> (32, H/2xW/2)

        return [input, x1, x2, x3, x4, x5, x_out]



def conv1x1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


# Used for StereoNet feature extractor
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, with_bn_relu=False, leaky_relu=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    if with_bn_relu:
        relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        conv = nn.Sequential(conv,
                             nn.BatchNorm2d(out_planes),
                             relu)
    return conv


def conv5x5(in_channels, out_channels, stride=2,
            dilation=1, use_bn=True):
    bias = False if use_bn else True
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride,
                     padding=2, dilation=dilation, bias=bias)
    relu = nn.ReLU(inplace=True)
    if use_bn:
        out = nn.Sequential(conv,
                            nn.BatchNorm2d(out_channels),
                            relu)
    else:
        out = nn.Sequential(conv, relu)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, leaky_relu=True):
        """StereoNet uses leaky relu (alpha = 0.2)"""
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class StereoNetFeature(nn.Module):
    def __init__(self, num_downsample=3):
        """Feature extractor of StereoNet
        Args:
            num_downsample: 2, 3 or 4
        """
        super(StereoNetFeature, self).__init__()

        self.num_downsample = num_downsample

        downsample = nn.ModuleList()

        in_channels = 3
        out_channels = 32
        for _ in range(num_downsample):
            downsample.append(conv5x5(in_channels, out_channels))
            in_channels = 32

        self.downsample = nn.Sequential(*downsample)

        residual_blocks = nn.ModuleList()

        for _ in range(6):
            residual_blocks.append(BasicBlock(out_channels, out_channels))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # StereoNet has no bn and relu for last conv layer,
        self.final_conv = conv3x3(out_channels, out_channels)

    def forward(self, img):
        out = self.downsample(img)  # [B, 32, H/8, W/8]
        out = self.residual_blocks(out)  # [B, 32, H/8, W/8]
        out = self.final_conv(out)  # [B, 32, H/8, W/8]

        return out


# Used for PSMNet feature extractor
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class PSMNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(PSMNetBasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class FeaturePyrmaid(nn.Module):
    def __init__(self, in_channel=32):
        super(FeaturePyrmaid, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(in_channel, in_channel, kernel_size=3, padding=1),
            BasicConv(in_channel, in_channel, kernel_size=5, stride=3, padding=2),
            DeformConv2d(in_channel, in_channel))

        self.out1 = nn.Sequential(nn.Conv2d(in_channel, in_channel * 2, kernel_size=3,
                                            stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel * 2),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=1,
                                            stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channel * 2),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  )

        self.out2 = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel * 4, kernel_size=3,
                                            stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel * 4),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(in_channel * 4, in_channel * 4, kernel_size=1,
                                            stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channel * 4),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  )

    def forward(self, x):
        # x = x[-1]
        # x: [B, 32, H, W] --> [B, 32, H/3, W/3]    if use self.conv_start
        # x = self.conv_start(x)
        out1 = self.out1(x)  # [B, 64, H/2, W/2] --> [B, 64, H/6, W/6]
        out2 = self.out2(out1)  # [B, 128, H/4, W/4] --> [B, 128, H/12, W/12]

        return [x, out1, out2]


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=128,
                 num_levels=3):
        # FPN paper uses 256 out channels by default
        super(FeaturePyramidNetwork, self).__init__()

        assert isinstance(in_channels, list)

        self.in_channels = in_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(num_levels):
            lateral_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # Inputs: resolution high -> low
        inputs = inputs[-1]
        assert len(self.in_channels) == len(inputs)

        # Build laterals
        laterals = [lateral_conv(inputs[i])
                    for i, lateral_conv in enumerate(self.lateral_convs)]

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # Build outputs
        out = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return out


class PSMNetFeature(nn.Module):
    def __init__(self):
        super(PSMNetFeature, self).__init__()
        self.inplanes = 32

        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))  # H/2

        self.layer1 = self._make_layer(PSMNetBasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(PSMNetBasicBlock, 64, 16, 2, 1, 1)  # H/4
        self.layer3 = self._make_layer(PSMNetBasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(PSMNetBasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)  # [32, H/4, W/4]

        return output_feature


# GANet feature
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn

        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, concat=True, bn=True, relu=True,
                 mdconv=False):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, bn, relu, kernel_size=3,
                                       stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if (x.size() != rem.size()):
            print(x.size(), rem.size())

        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class GANetFeature(nn.Module):
    """Height and width need to be divided by 48, downsampled by 1/3"""

    def __init__(self, feature_mdconv=False):
        super(GANetFeature, self).__init__()

        if feature_mdconv:
            self.conv_start = nn.Sequential(
                BasicConv(3, 32, kernel_size=3, padding=1),
                BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
                DeformConv2d(32, 32))
        else:
            self.conv_start = nn.Sequential(
                BasicConv(3, 32, kernel_size=3, padding=1),
                BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
                BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)

        if feature_mdconv:
            self.conv3a = DeformConv2d(64, 96, kernel_size=3, stride=2)
            self.conv4a = DeformConv2d(96, 128, kernel_size=3, stride=2)
        else:
            self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
            self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)

        if feature_mdconv:
            self.conv3b = Conv2x(64, 96, mdconv=True)
            self.conv4b = Conv2x(96, 128, mdconv=True)
        else:
            self.conv3b = Conv2x(64, 96)
            self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0_a = x
        x = self.conv1a(x)
        rem1_a = x
        x = self.conv2a(x)
        rem2_a = x
        x = self.conv3a(x)
        rem3_a = x
        x = self.conv4a(x)
        rem4_a = x

        x = self.deconv4a(x, rem3_a)
        rem3d_a = x
        x = self.deconv3a(x, rem2_a)
        rem2d_a = x
        x = self.deconv2a(x, rem1_a)
        rem1d_a = x
        x = self.deconv1a(x, rem0_a)
        rem0d_a = x

        x = self.conv1b(x, rem1d_a)
        rem1_b = x
        x = self.conv2b(x, rem2d_a)
        rem2_b = x
        x = self.conv3b(x, rem3d_a)
        rem3_b = x
        x = self.conv4b(x, rem4_a)
        rem4_b = x

        x = self.deconv4b(x, rem3_b)
        rem3d_b = x
        x = self.deconv3b(x, rem2_b)
        rem2d_b = x
        x = self.deconv2b(x, rem1_b)
        rem1d_b = x
        out = self.deconv1b(x, rem0d_a)  # [B, 32, H/3, W/3]

        return [rem2_a, rem4_a, rem0d_a, rem2_b, rem4_b, out]


class GCNetFeature(nn.Module):
    def __init__(self):
        super(GCNetFeature, self).__init__()

        self.inplanes = 32
        self.conv1 = conv5x5(3, 32)
        self.conv2 = self._make_layer(PSMNetBasicBlock, 32, 8, 1, 1, 1)
        self.conv3 = conv3x3(32, 32)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # [32, H/2, W/2]

        return x

