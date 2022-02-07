import torch
import torch.nn as nn
import torch.nn.functional as F

from network.feature import BasicBlock, BasicConv, Conv2x
from network.deform import DeformConv2d
from network.warp import disp_warp
from network.utils import upsample, _BNReluConv


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


class StereoNetRefinement(nn.Module):
    def __init__(self):
        super(StereoNetRefinement, self).__init__()

        # Original StereoNet: left, disp
        self.conv = conv2d(4, 32)

        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img=None):
        """Upsample low resolution disparity prediction to
        corresponding resolution as image size
        Args:
            low_disp: [B, H, W]
            left_img: [B, 3, H, W]
            right_img: [B, 3, H, W]
        """
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        disp = disp * scale_factor  # scale correspondingly

        concat = torch.cat((disp, left_img), dim=1)  # [B, 4, H, W]
        out = self.conv(concat)
        out = self.dilated_blocks(out)
        residual_disp = self.final_conv(out)

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]

        return disp


class StereoDRNetRefinement(nn.Module):
    def __init__(self):
        super(StereoDRNetRefinement, self).__init__()

        # Left and warped error
        in_channels = 6

        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor

        # Warp right image to left view with current disparity
        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        error = warped_right - left_img  # [B, C, H, W]

        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]

        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(disp)  # [B, 16, H, W]
        concat2 = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        out = self.dilated_blocks(concat2)  # [B, 32, H, W]
        residual_disp = self.final_conv(out)  # [B, 1, H, W]

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]

        return disp


class HourglassRefinement(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self, device):
        super(HourglassRefinement, self).__init__()
        self.device = device

        # Left and warped error
        in_channels = 6
        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.conv_start = DeformConv2d(32, 32)

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = DeformConv2d(64, 96, kernel_size=3, stride=2)
        self.conv4a = DeformConv2d(96, 128, kernel_size=3, stride=2)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96, mdconv=True)
        self.conv4b = Conv2x(96, 128, mdconv=True)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):

        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor

        # Warp right image to left view with current disparity
        warped_right = disp_warp(right_img, disp, self.device)[0]  # [B, C, H, W]
        error = warped_right - left_img  # [B, C, H, W]
        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]
        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(disp)  # [B, 16, H, W]
        x = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        residual_disp = self.final_conv(x)  # [B, 1, H, W]

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]

        return disp


class Refine_disp_sem(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self, num_class, op='concat', low_disp_plus_1=False):
        super(Refine_disp_sem, self).__init__()

        # Left and warped error
        in_channels = 3
        self.op = op
        self.low_disp_plus_1 = low_disp_plus_1
        self.num_classes = num_class
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = conv2d(64, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity
        self.conv3 = conv2d(128, 16)

        if (self.op == 'concat'):
            self.conv_start = BasicConv(48, 32, kernel_size=3, stride=1, padding=1)
        elif (self.op == 'add'):
            self.conv_start = BasicConv(16, 32, kernel_size=3, stride=1, padding=1)
        else:
            raise NotImplementedError

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)


        self.deconv1 = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                               stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                               stride=2, padding=1)

        self.final_conv_disp = nn.Conv2d(32, 1, 3, 1, 1)
        self.final_conv_sem = nn.Conv2d(32, 128, 3, 1, 1)

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)

    def forward(self, low_disp, left_img, left_sem):

        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H/4, W/4]
        scale_factor = left_img.size(-1) / low_disp.size(-1)

        left_feat = self.conv0(left_img)
        left_feat = self.bn(left_feat)
        left_feat = self.relu(left_feat)
        left_feat = self.maxpool(left_feat)

        if self.low_disp_plus_1:
            low_disp = low_disp + 1

        conv1 = self.conv1(left_feat)  # [B, 16, H/4, W/4]
        conv2 = self.conv2(low_disp)  # [B, 16, H/4, W/4]
        conv3 = self.conv3(left_sem)    # [B, 16, H/4, W/4]

        if (self.op == 'concat'):
            x = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]
            x = torch.cat((x, conv3), dim=1)  # [B, 32+16, H, W]
        elif (self.op == 'add'):
            x = conv1 + conv2 + conv3
        else:
            raise NotImplementedError


        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        x = self.deconv4a(x, rem3)
        x = self.deconv3a(x, rem2)
        x = self.deconv2a(x, rem1)
        x = self.deconv1a(x, rem0)


        sem = self.final_conv_sem(x)  # [B, 128, H/4, W/4]

        x = self.deconv1(x)
        x = self.deconv2(x)         # [B, 32, H, W]
        disp = self.final_conv_disp(x)  # [B, 1, H, W]

        residual_disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        disp = disp + residual_disp

        disp = F.relu(disp, inplace=True)
        disp = disp.squeeze(1)  # [B, H, W]
        disp = disp * scale_factor


        return disp, sem


class Refine_New1(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self, num_class):
        super(Refine_New1, self).__init__()

        # Left and warped error
        in_channels = 3
        self.num_classes = num_class
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = conv2d(64, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity
        self.conv3 = conv2d(128, 16)

        self.conv_start = BasicConv(48, 32, kernel_size=3, stride=1, padding=1)

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)


        self.deconv1 = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                               stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                               stride=2, padding=1)

        self.deconv1_sem = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                                          stride=2, padding=1)
        self.deconv2_sem = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                                          stride=2, padding=1)

        self.final_conv_disp = nn.Conv2d(32, 1, 3, 1, 1)
        self.final_conv_sem = nn.Conv2d(32, 128, 3, 1, 1)


    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)

    def forward(self, low_disp, left_img, left_sem):

        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H/4, W/4]
        scale_factor = left_img.size(-1) / low_disp.size(-1)

        left_feat = self.conv0(left_img)
        left_feat = self.bn(left_feat)
        left_feat = self.relu(left_feat)
        left_feat = self.maxpool(left_feat)

        conv1 = self.conv1(left_feat)  # [B, 16, H/4, W/4]
        conv2 = self.conv2(low_disp)  # [B, 16, H/4, W/4]
        conv3 = self.conv3(left_sem)    # [B, 16, H/4, W/4]
        x = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]
        x = torch.cat((x, conv3), dim=1)  # [B, 32+16, H, W]

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        x = self.deconv4a(x, rem3)
        x = self.deconv3a(x, rem2)
        x = self.deconv2a(x, rem1)
        x = self.deconv1a(x, rem0)

        sem_x = self.deconv1_sem(x)
        sem_x = self.deconv2_sem(sem_x)
        sem = self.final_conv_sem(sem_x)  # [B, 128, H, W]

        x = self.deconv1(x)
        x = self.deconv2(x)         # [B, 32, H, W]
        disp = self.final_conv_disp(x)  # [B, 1, H, W]

        residual_disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        disp = disp + residual_disp

        disp = F.relu(disp, inplace=True)
        disp = disp.squeeze(1)  # [B, H, W]
        disp = disp * scale_factor


        return disp, sem



# Refine_New8 --> ResHourglass
class ResHourglass(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self, num_class, op='add', low_disp_plus_1=False):
        super(ResHourglass, self).__init__()

        self.op = op
        self.low_disp_plus_1 = low_disp_plus_1
        in_channels = 3
        self.num_classes = num_class
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = conv2d(64, 32)
        self.conv2 = conv2d(1, 32)  # on low disparity
        self.conv3 = conv2d(128, 32)

        if self.op == 'add':
            self.conv_start = BasicConv(32, 32, kernel_size=3, stride=1, padding=1)
        elif self.op == 'concat':
            self.conv_start = BasicConv(32 * 3, 32, kernel_size=3, stride=1, padding=1)
        else:
            raise NotImplementedError

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)


        self.deconv1 = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                               stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                               stride=2, padding=1)

        self.final_conv_disp = nn.Conv2d(32, 1, 3, 1, 1)
        self.final_conv_sem = nn.Conv2d(32, 128, 3, 1, 1)

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)

    def forward(self, low_disp, left_img, left_sem):

        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H/4, W/4]
        scale_factor = left_img.size(-1) / low_disp.size(-1)

        left_feat = self.conv0(left_img)
        left_feat = self.bn(left_feat)
        left_feat = self.relu(left_feat)
        left_feat = self.maxpool(left_feat)

        if self.low_disp_plus_1:
            low_disp = low_disp + 1

        conv1 = self.conv1(left_feat)  # [B, 16, H/4, W/4]
        conv2 = self.conv2(low_disp)  # [B, 16, H/4, W/4]
        conv3 = self.conv3(left_sem)    # [B, 16, H/4, W/4]
        if self.op == 'add':
            x = conv1 + conv2 + conv3
            rem_start = x
            x = self.conv_start(x)
            rem0 = x
        elif self.op == 'concat':
            x = torch.cat((conv1, conv2), dim=1)  # [B, 32*2, H, W]
            x = torch.cat((x, conv3), dim=1)  # [B, 32*3, H, W]
            x = self.conv_start(x)
            rem_start = x
            rem0 = x
        else:
            raise NotImplementedError

        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        x = self.deconv4a(x, rem3)
        x = self.deconv3a(x, rem2)
        x = self.deconv2a(x, rem1)
        x = self.deconv1a(x, rem0)

        x = x + rem_start

        sem = self.final_conv_sem(x)  # [B, 128, H/4, W/4]

        x = self.deconv1(x)
        x = self.deconv2(x)         # [B, 32, H, W]
        disp = self.final_conv_disp(x)  # [B, 1, H, W]

        residual_disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        disp = disp + residual_disp

        disp = F.relu(disp, inplace=True)
        disp = disp.squeeze(1)  # [B, H, W]
        disp = disp * scale_factor


        return disp, sem


# Refine_New10 --> ResStackedHourglass
class ResStackedHourglass(nn.Module):
    """Height and width need to be divided by 16"""
    def __init__(self, num_class, op='add', low_disp_plus_1=False):
        super(ResStackedHourglass, self).__init__()

        self.op = op
        self.low_disp_plus_1 = low_disp_plus_1
        in_channels = 3
        self.num_classes = num_class
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = conv2d(64, 32)
        self.conv2 = conv2d(1, 32)  # on low disparity
        self.conv3 = conv2d(128, 32)

        if self.op == 'add':
            self.conv_start = BasicConv(32, 32, kernel_size=3, stride=1, padding=1)
        elif self.op == 'concat':
            self.conv_start = BasicConv(32*3, 32, kernel_size=3, stride=1, padding=1)
        else:
            raise NotImplementedError

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96, mdconv=True)
        self.conv4b = Conv2x(96, 128, mdconv=True)
        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

        self.deconv1 = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                               stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                               stride=2, padding=1)

        self.final_conv_disp = nn.Conv2d(32, 1, 3, 1, 1)
        self.final_conv_sem = nn.Conv2d(32, 128, 3, 1, 1)

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)

    def forward(self, low_disp, left_img, left_sem):

        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H/4, W/4]
        scale_factor = left_img.size(-1) / low_disp.size(-1)

        left_feat = self.conv0(left_img)
        left_feat = self.bn(left_feat)
        left_feat = self.relu(left_feat)
        left_feat = self.maxpool(left_feat)

        if self.low_disp_plus_1:
            low_disp = low_disp + 1

        conv1 = self.conv1(left_feat)  # [B, 16, H/4, W/4]
        conv2 = self.conv2(low_disp)  # [B, 16, H/4, W/4]
        conv3 = self.conv3(left_sem)    # [B, 16, H/4, W/4]

        if self.op == 'add':
            x = conv1 + conv2 + conv3
            rem0_start = x
            x = self.conv_start(x)
            rem0 = x
        elif self.op == 'concat':
            x = torch.cat((conv1, conv2), dim=1)  # [B, 32*2, H, W]
            x = torch.cat((x, conv3), dim=1)  # [B, 32*3, H, W]
            rem_start = x
            x = self.conv_start(x)
            rem0_start = x
        else:
            raise NotImplementedError

        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        if self.op == 'add':
            x = self.deconv1a(x, rem0)
            rem0 = x
        elif self.op == 'concat':
            x = self.deconv1a(x, rem0_start)
            rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        x = x + rem0_start

        sem = self.final_conv_sem(x)  # [B, 128, H/4, W/4]

        x = self.deconv1(x)
        x = self.deconv2(x)         # [B, 32, H, W]
        disp = self.final_conv_disp(x)  # [B, 1, H, W]

        residual_disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        disp = disp + residual_disp

        disp = F.relu(disp, inplace=True)
        disp = disp.squeeze(1)  # [B, H, W]
        disp = disp * scale_factor


        return disp, sem

# Refine_New33 --> TripleStackedHourglass
class TripleStackedHourglass(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self, num_class, op='concat', low_disp_plus_1=False):
        super(TripleStackedHourglass, self).__init__()

        # Left and warped error
        in_channels = 3
        self.low_disp_plus_1 = low_disp_plus_1
        self.op = op

        self.num_classes = num_class
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = conv2d(64, 32)
        self.conv2 = conv2d(1, 32)  # on low disparity
        self.conv3 = conv2d(128, 32)

        self.conv_start = BasicConv(32*3, 32, kernel_size=3, stride=1, padding=1)

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96, mdconv=True)
        self.conv4b = Conv2x(96, 128, mdconv=True)
        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

        self.conv1c = Conv2x(32, 48)
        self.conv2c = Conv2x(48, 64)
        self.conv3c = Conv2x(64, 96, mdconv=True)
        self.conv4c = Conv2x(96, 128, mdconv=True)
        self.deconv4c = Conv2x(128, 96, deconv=True)
        self.deconv3c = Conv2x(96, 64, deconv=True)
        self.deconv2c = Conv2x(64, 48, deconv=True)
        self.deconv1c = Conv2x(48, 32, deconv=True)

        self.deconv1 = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                               stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, bias=False, kernel_size=4,
                               stride=2, padding=1)

        self.final_conv_disp = nn.Conv2d(32, 1, 3, 1, 1)
        self.final_conv_sem = nn.Conv2d(32, 128, 3, 1, 1)

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)

    def forward(self, low_disp, left_img, left_sem):

        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H/4, W/4]
        scale_factor = left_img.size(-1) / low_disp.size(-1)

        left_feat = self.conv0(left_img)
        left_feat = self.bn(left_feat)
        left_feat = self.relu(left_feat)
        left_feat = self.maxpool(left_feat)

        # if self.low_disp_plus_1:
        #     low_disp = low_disp + 1

        conv1 = self.conv1(left_feat)  # [B, 32, H/4, W/4]
        conv2 = self.conv2(low_disp + 1)  # [B, 32, H/4, W/4]
        conv3 = self.conv3(left_sem)    # [B, 32, H/4, W/4]

        x = torch.cat((conv1, conv2), dim=1)  # [B, 32*2, H, W]
        x = torch.cat((x, conv3), dim=1)  # [B, 32*3, H, W]

        # x = conv1 + conv2 + conv3
        rem_start = x

        x = self.conv_start(x)
        rem0_start = x      # [B, 32, H, W]

        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0_start)
        rem0 = x        # [B, 32, H, W]

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        rem4 = x
        x = self.deconv4b(x, rem3)
        rem3 = x
        x = self.deconv3b(x, rem2)
        rem2 = x
        x = self.deconv2b(x, rem1)
        rem1 = x
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]
        rem0 = x

        x = self.conv1c(x, rem1)
        rem1 = x
        x = self.conv2c(x, rem2)
        rem2 = x
        x = self.conv3c(x, rem3)
        rem3 = x
        x = self.conv4c(x, rem4)
        rem4 = x
        x = self.deconv4c(x, rem3)
        rem3 = x
        x = self.deconv3c(x, rem2)
        rem2 = x
        x = self.deconv2c(x, rem1)
        rem1 = x
        x = self.deconv1c(x, rem0)  # [B, 32, H, W]
        rem0 = x

        x = x + rem0_start

        sem = self.final_conv_sem(x)  # [B, 128, H/4, W/4]

        x = self.deconv1(x)
        x = self.deconv2(x)         # [B, 32, H, W]
        disp = self.final_conv_disp(x)  # [B, 1, H, W]

        residual_disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        disp = disp + residual_disp

        disp = F.relu(disp, inplace=True)
        disp = disp.squeeze(1)  # [B, H, W]
        disp = disp * scale_factor


        return disp, sem

