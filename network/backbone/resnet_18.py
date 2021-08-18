import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.utils.checkpoint as cp
import cv2
import numpy as np
from math import log2

from network.utils import _Upsample, SpatialPyramidPooling, SeparableConv2d

__all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1, separable=False):
    """3x3 convolution with padding"""
    conv_class = SeparableConv2d if separable else nn.Conv2d
    return conv_class(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def _bn_function_factory(conv, norm, relu=None):
    """return a conv-bn-relu function"""
    def bn_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x

    return bn_function


def do_efficient_fwd(block, x, efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, deleting=False,
                 separable=False):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, separable=separable)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, separable=separable)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient
        self.deleting = deleting

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.deleting is False:
            bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
            bn_2 = _bn_function_factory(self.conv2, self.bn2)

            out = do_efficient_fwd(bn_1, x, self.efficient)
            out = do_efficient_fwd(bn_2, out, self.efficient)
        else:
            out = torch.zeros_like(residual)

        out = out + residual
        relu = self.relu(out)
        # print(f'Basic Block memory: {torch.cuda.memory_allocated() // 2**20}')

        return relu, out


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
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
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True,
                 mdconv=False):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3,
                                       stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2, self.relu)
        bn_3 = _bn_function_factory(self.conv3, self.bn3, self.relu)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)
        out = do_efficient_fwd(bn_3, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)

        return relu, out


class ResNet(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=True, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn

        # rgb branch
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # depth branch
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1_d = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        upsamples = []
        # _make_layer_rgb  _make_layer
        self.layer1 = self._make_layer_rgb(block, 64, 64, layers[0])
        self.layer1_d = self._make_layer_d(block, 64, 64, layers[0])
        self.attention_1 = self.attention(64)
        self.attention_1_d = self.attention(64)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)] #  num_maps_in, skip_maps_in, num_maps_out, k: kernel size of blend conv

        self.layer2 = self._make_layer_rgb(block, 64, 128, layers[1], stride=2)
        self.layer2_d = self._make_layer_d(block, 64, 128, layers[1], stride=2)
        self.attention_2 = self.attention(128)
        self.attention_2_d = self.attention(128)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]

        self.layer3 = self._make_layer_rgb(block, 128, 256, layers[2], stride=2)
        self.layer3_d = self._make_layer_d(block, 128, 256, layers[2], stride=2)
        self.attention_3 = self.attention(256)
        self.attention_3_d = self.attention(256)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]

        self.layer4 = self._make_layer_rgb(block, 256, 512, layers[3], stride=2)
        self.layer4_d = self._make_layer_d(block, 256, 512, layers[3], stride=2)
        self.attention_4 = self.attention(512)
        self.attention_4_d = self.attention(512)

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4,
                          self.conv1_d, self.maxpool_d, self.layer1_d, self.layer2_d, self.layer3_d, self.layer4_d]
        if self.use_bn:
            self.fine_tune += [self.bn1, self.bn1_d, self.attention_1, self.attention_1_d, self.attention_2, self.attention_2_d,
                               self.attention_3, self.attention_3_d, self.attention_4, self.attention_4_d]

        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn)
        self.upsample = nn.ModuleList(list(reversed(upsamples)))

        self.random_init = [ self.spp, self.upsample]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_rgb(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            layers = [nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def _make_layer_d(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            layers = [nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        inplanes = planes * block.expansion
        self.inplanes = inplanes
        for i in range(1, blocks):
            layers += [block(inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def channel_attention(self, rgb_skip, depth_skip, attention):
        assert rgb_skip.shape == depth_skip.shape, 'rgb skip shape:{} != depth skip shape:{}'.format(rgb_skip.shape, depth_skip.shape)
        # single_attenton
        rgb_attention = attention(rgb_skip)
        depth_attention = attention(depth_skip)
        rgb_after_attention = torch.mul(rgb_skip, rgb_attention)
        depth_after_attention = torch.mul(depth_skip, depth_attention)
        skip_after_attention = rgb_after_attention + depth_after_attention
        return skip_after_attention

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)


    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, rgb):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [self.spp.forward(skip)]
        return features

    def forward_down_fusion(self, rgb, depth):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        depth = depth.unsqueeze(1)
        y = self.conv1_d(depth)
        y = self.bn1_d(y)
        y = self.relu_d(y)
        y = self.maxpool_d(y)

        features = []
        # block 1
        x, skip_rgb = self.forward_resblock(x, self.layer1)
        y, skip_depth = self.forward_resblock(y, self.layer1_d)
        x_attention = self.attention_1(x)
        y_attention = self.attention_1_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        x = x + y
        features += [skip_rgb]
        # block 2
        x, skip_rgb = self.forward_resblock(x, self.layer2)
        y, skip_depth = self.forward_resblock(y, self.layer2_d)
        x_attention = self.attention_2(x)
        y_attention = self.attention_2_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        x = x + y
        features += [skip_rgb]
        # block 3
        x, skip_rgb = self.forward_resblock(x, self.layer3)
        y, skip_depth = self.forward_resblock(y, self.layer3_d)
        x_attention = self.attention_3(x)
        y_attention = self.attention_3_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        x = x + y
        features += [skip_rgb]
        # block 4
        x, skip_rgb = self.forward_resblock(x, self.layer4)
        y, skip_depth = self.forward_resblock(y, self.layer4_d)
        x_attention = self.attention_4(x)
        y_attention = self.attention_4_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        x = x + y
        features += [self.spp.forward(x)]
        return features


    def forward_up(self, features):
        features = features[::-1]

        x = features[0]

        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}

    def forward(self, rgb, depth = None):
        if depth is None:
            return self.forward_up(self.forward_down(rgb))
        else:
            return self.forward_up(self.forward_down_fusion(rgb, depth))

    def _load_resnet_pretrained(self, url):
        pretrain_dict = model_zoo.load_url(model_urls[url])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            # print('%%%%% ', k)
            if k in state_dict:
                if k.startswith('conv1'):
                    model_dict[k] = v
                    # print('##### ', k)
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6]+'_d'+k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ResNet_hourglass(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=True, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, **kwargs):
        super(ResNet_hourglass, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn

        # rgb branch
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        upsamples = []
        upsamples_disp = []
        # _make_layer_rgb  _make_layer
        self.layer1 = self._make_layer_rgb(block, 64, 64, layers[0])
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)] #  num_maps_in, skip_maps_in, num_maps_out, k: kernel size of blend conv
        # upsamples_disp += [_Upsample(128, self.inplanes, 64, use_bn=self.use_bn, k=k_up)] #  num_maps_in, skip_maps_in, num_maps_out, k: kernel size of blend conv

        self.layer2 = self._make_layer_rgb(block, 64, 128, layers[1], stride=2)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        # upsamples_disp += [_Upsample(256, self.inplanes, 128, use_bn=self.use_bn,
        #                              k=k_up)]

        self.layer3 = self._make_layer_rgb(block, 128, 256, layers[2], stride=2)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        # upsamples_disp += [_Upsample(num_features, self.inplanes, 256, use_bn=self.use_bn,
        #                              k=k_up)]

        self.layer4 = self._make_layer_rgb(block, 256, 512, layers[3], stride=2)

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]

        if self.use_bn:
            self.fine_tune += [self.bn1]

        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn)
        self.upsample = nn.ModuleList(list(reversed(upsamples)))

        # disparity feature extractor
        self.conv4a = BasicConv(512, 1024, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(1024, 512, deconv=True)
        self.deconv3a = Conv2x(512, 256, deconv=True)
        self.deconv2a = Conv2x(256, 128, deconv=True)
        self.deconv1a = Conv2x(128, 64, deconv=True)

        self.conv1b = Conv2x(64, 128)
        self.conv2b = Conv2x(128, 256)
        self.conv3b = Conv2x(256, 512)
        self.conv4b = Conv2x(512, 1024)

        self.deconv4b = Conv2x(1024, 512, deconv=True)
        self.deconv3b = Conv2x(512, 256, deconv=True)
        self.deconv2b = Conv2x(256, 128, deconv=True)
        self.deconv1b = Conv2x(128, 64, deconv=True)

        self.conv_final = BasicConv(64, 32, kernel_size=3, stride=1, padding=1)

        self.random_init = [self.spp, self.upsample,
                            self.conv4a, self.conv_final,
                            self.deconv4a, self.deconv3a, self.deconv2a, self.deconv1a,
                            self.conv1b, self.conv2b, self.conv3b, self.conv4b,
                            self.deconv4b, self.deconv3b, self.deconv2b, self.deconv1b]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_rgb(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            layers = [nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        inplanes = planes * block.expansion
        self.inplanes = inplanes
        for i in range(1, blocks):
            layers += [block(inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, rgb):
        x = self.conv1(rgb)     # rgb : Bx3xHxW --> x : Bx64xH/2xW/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # Bx64xH/2xW/2 --> Bx64xH/4xW/4

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]      # Bx64xH/4xW/4 --> features[0]: Bx64xH/4xW/4
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]      # Bx64xH/4xW/4 --> features[1]: Bx128xH/8xW/8
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]      # Bx128xH/8xW/8 --> features[2]: Bx256xH/16xW/16
        x, skip = self.forward_resblock(x, self.layer4)     # Bx256xH/16xW/16 --> Bx512xH/32xW/32
        features += [skip]      # features[3]: Bx512xH/32xW/32
        features += [self.spp.forward(skip)]                # features[4]: Bx512xH/32xW/32 --> Bx128xH/32xW/32
        return features

    def forward_up(self, features):
        features = features[::-1]   # reverse order
        # features[0]:Bx128xH/32xW/32, features[1]:Bx512xH/32xW/32
        # features[2]:Bx256xH/16xW/16, features[3]:Bx128xH/8xW/8, features[4]:Bx64xH/4xW/4

        x = features[0]     #Bx128xH/32xW/32

        upsamples = []  # upsample[0]:Bx128xH/16xW/16, upsample[1]:Bx128xH/8xW/8. upsample[2]:Bx128xH/4xW/4
        for skip, up in zip(features[2:5], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}

    def forward_up_for_disp(self, features):
        features = features[::-1]   # reverse order
        # features[0]:Bx128xH/32xW/32, features[1]:Bx512xH/32xW/32
        # features[2]:Bx256xH/16xW/16, features[3]:Bx128xH/8xW/8, features[4]: Bx64xH/4xW/4

        # for semantic seg feature extractor
        x_sem = features[0]  # Bx128xH/32xW/32
        upsamples = []  # upsample[0]:Bx128xH/16xW/16, upsample[1]:Bx128xH/8xW/8. upsample[2]:Bx128xH/4xW/4
        for skip, up in zip(features[2:5], self.upsample):
            x_sem = up(x_sem, skip)
            upsamples += [x_sem]


        # for disparity feature extractor
        x = self.conv4a(features[1])    # Bx1024xH/64xW/64
        rem4 = x

        rem_deconva = []
        x = self.deconv4a(x, features[1])               # Bx512xH/32xW/32
        rem_deconva += [x]
        x = self.deconv3a(x, features[2])               # Bx256xH/16xW/16
        rem_deconva += [x]
        x = self.deconv2a(x, features[3])               # Bx128xH/8xW/8
        rem_deconva += [x]
        x = self.deconv1a(x, features[4])               # Bx64xH/4xW/4
        rem_deconva += [x]

        rem_convb = []
        x = self.conv1b(x, rem_deconva[2])              # Bx128xH/8xW/8
        rem_convb += [x]
        x = self.conv2b(x, rem_deconva[1])              # Bx256xH/16xW/16
        rem_convb += [x]
        x = self.conv3b(x, rem_deconva[0])              # Bx512xH/32xW/32
        rem_convb += [x]
        x = self.conv4b(x, rem4)                        # Bx1024xH/64xW/64

        x = self.deconv4b(x, rem_convb[2])              # Bx512xH/32xW/32
        x = self.deconv3b(x, rem_convb[1])              # Bx256xH/16xW/16
        x = self.deconv2b(x, rem_convb[0])              # Bx128xH/8xW/8
        x = self.deconv1b(x, rem_deconva[3])            # Bx64xH/4xW/4

        x_disp = x
        # x_disp = self.conv_final(x)                     # Bx32xH/4xW/4


        return x_sem, x_disp, {'features': features, 'upsamples': upsamples}


    def forward(self, rgb):
        return self.forward_up_for_disp(self.forward_down(rgb))
        # return self.forward_up(self.forward_down(rgb))


class ResNet_swift(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=False, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, spp_drop_rate=0.0,
                 upsample_skip=True, upsample_only_skip=False,
                 detach_upsample_skips=(), detach_upsample_in=False,
                 target_size=None, output_stride=4, mean=(73.1584, 82.9090, 72.3924),
                 std=(44.9149, 46.1529, 45.3192), scale=1, separable=False,
                 upsample_separable=False, **kwargs):
        super(ResNet_swift, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn
        self.separable = separable
        self.register_buffer('img_mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('img_std', torch.tensor(std).view(1, -1, 1, 1))
        if scale != 1:
            self.register_buffer('img_scale', torch.tensor(scale).view(1, -1, 1, 1).float())

        self.detach_upsample_in = detach_upsample_in
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.target_size = target_size
        if self.target_size is not None:
            h, w = target_size
            target_sizes = [(h // 2 ** i, w // 2 ** i) for i in range(2, 6)]
        else:
            target_sizes = [None] * 4
        upsamples = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        upsamples += [
            _Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, detach_skip=2 in detach_upsample_skips, fixed_size=target_sizes[0],
                      separable=upsample_separable)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        upsamples += [
            _Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, detach_skip=1 in detach_upsample_skips, fixed_size=target_sizes[1],
                      separable=upsample_separable)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        upsamples += [
            _Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, detach_skip=0 in detach_upsample_skips, fixed_size=target_sizes[2],
                      separable=upsample_separable)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        if self.use_bn:
            self.fine_tune += [self.bn1]

        num_levels = 3
        self.spp_size = kwargs.get('spp_size', num_features)
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=num_features, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn, drop_rate=spp_drop_rate,
                                         fixed_size=target_sizes[3])
        num_up_remove = max(0, int(log2(output_stride) - 2))
        self.upsample = nn.ModuleList(list(reversed(upsamples[num_up_remove:])))

        self.random_init = [self.spp, self.upsample]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn,
                        separable=self.separable)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn,
                             separable=self.separable)]

        return nn.Sequential(*layers)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image):
        if hasattr(self, 'img_scale'):
            image /= self.img_scale
        image -= self.img_mean
        image /= self.img_std

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [self.spp.forward(skip)]
        return features

    def forward_up(self, features):
        features = features[::-1]

        x = features[0]
        if self.detach_upsample_in:
            x = x.detach()

        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}

    def forward(self, image):
        return self.forward_up(self.forward_down(image))

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet_hourglass(BasicBlock, [2, 2, 2, 2], **kwargs)
    model = ResNet_swift(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        print('pretrained dict loaded sucessfully')
    return model


