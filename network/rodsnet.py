import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from network.aggregation import AdaptiveAggregation
from network.estimation import DisparityEstimation
from network.refinement import StereoNetRefinement, StereoDRNetRefinement, \
    HourglassRefinement, Refine_disp_sem
from network.cost import CostVolumePyramid

from network.backbone.resnet_pyramid import resnet18_pyramid
from network.backbone.efficientnet_pyramid import efficientnet_pyramid
from network.backbone.mobilenetv2_pyramid import mobilenetv2_pyramid

from network.utils import _BNReluConv, upsample


class RODSNet(nn.Module):
    def __init__(self,
                 opts,
                 max_disp,
                 num_downsample=2,
                 num_classes=19,
                 device=None,
                 feature_similarity='correlation',
                 aggregation_type='adaptive',
                 num_scales=3,
                 num_fusions=6,
                 deformable_groups=2,
                 mdconv_dilation=2,
                 refinement_type='ours',
                 no_intermediate_supervision=False,
                 num_stage_blocks=1,
                 num_deform_blocks=3):
        super(RODSNet, self).__init__()

        self.opts = opts
        self.refinement_type = refinement_type
        self.num_downsample = num_downsample
        self.aggregation_type = aggregation_type
        self.num_scales = num_scales
        self.max_disp = max_disp // 4      # it depends on feature extractor's width-height scale
        self.num_classes = num_classes
        self.device = device

        scale = 1
        mean = [73.15, 82.90, 72.3]
        std = [47.67, 48.49, 47.73]

        backbone = self.opts.model
        if backbone == 'resnet18':
            self.feature_extractor = resnet18_pyramid(pretrained=True, pyramid_levels=3, k_upsample=3,
                                                    scale=scale, mean=mean, std=std, k_bneck=1, output_stride=4,
                                                    efficient=True)
        elif backbone == 'efficientnetb0':
            self.feature_extractor = efficientnet_pyramid(pretrained=True, pyramid_levels=3, k_upsample=3,
                                                          mean=mean, std=std, num_classes=self.num_classes,
                                                          )
        elif backbone == 'mobilenetv2':
            self.feature_extractor = mobilenetv2_pyramid(pretrained=True
                                                         )
        else:
            raise NotImplementedError

        if self.opts.train_semantic:
            self.segmentation = _BNReluConv(self.feature_extractor.num_features, self.num_classes, batch_norm=True,
                                            k=1, bias=True)
            self.loss_ret_additional = False
            self.img_req_grad = False
            self.upsample_logits = True
            self.multiscale_factors = (.5, .75, 1.5, 2.)

        if self.opts.train_disparity:

            # Cost volume construction
            self.cost_volume = CostVolumePyramid(self.max_disp,
                                                  feature_similarity=feature_similarity)

            # Cost aggregation
            self.aggregation = AdaptiveAggregation(max_disp=self.max_disp,
                                                   num_scales=num_scales,
                                                   num_fusions=num_fusions,
                                                   num_stage_blocks=num_stage_blocks,
                                                   num_deform_blocks=num_deform_blocks,
                                                   mdconv_dilation=mdconv_dilation,
                                                   deformable_groups=deformable_groups,
                                                   intermediate_supervision=not no_intermediate_supervision)

            match_similarity = False if feature_similarity in ['difference', 'concat'] else True

            # Disparity estimation
            self.disparity_estimation = DisparityEstimation(self.max_disp, self.device, match_similarity)

            # Refinement
            if self.opts.with_refine:
                if self.refinement_type is not None and self.refinement_type != 'None':
                    if self.refinement_type in ['stereonet', 'stereodrnet', 'hourglass']:
                        refine_module_list = nn.ModuleList()
                        for i in range(num_downsample):
                            if self.refinement_type == 'stereonet':
                                refine_module_list.append(StereoNetRefinement())
                            elif self.refinement_type == 'stereodrnet':
                                refine_module_list.append(StereoDRNetRefinement())
                            elif self.refinement_type == 'hourglass':
                                refine_module_list.append(HourglassRefinement(self.device))
                            else:
                                raise NotImplementedError
                        self.refinement = refine_module_list
                    elif self.refinement_type == 'ours':
                        self.refinement_new = Refine_disp_sem(self.num_classes)
                    else:
                        raise NotImplementedError

    def feature_extraction(self, img):
        x_sem, x_disp, info = self.feature_extractor(img)
        return x_sem, x_disp

    def predict_segmentation(self, features):
        segmentation = self.segmentation.forward(features)
        return segmentation

    def cost_volume_construction(self, left_feature, right_feature):
        cost_volume = self.cost_volume(left_feature, right_feature)

        if isinstance(cost_volume, list):
            if self.num_scales == 1:
                cost_volume = [cost_volume[0]]  # ablation purpose for 1 scale only
        elif self.aggregation_type == 'adaptive':
            cost_volume = [cost_volume]
        return cost_volume

    def disparity_computation(self, aggregation):
        if isinstance(aggregation, list):
            disparity_pyramid = []
            length = len(aggregation)  # D/3, D/6, D/12
            for i in range(length):
                disp = self.disparity_estimation(aggregation[length - 1 - i])  # reverse
                disparity_pyramid.append(disp)  # D/12, D/6, D/3
        else:
            disparity = self.disparity_estimation(aggregation)
            disparity_pyramid = [disparity]

        return disparity_pyramid

    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        if self.refinement_type is not None and self.refinement_type != 'None':
            for i in range(self.num_downsample):
                scale_factor = 1. / pow(2, self.num_downsample - i - 1)

                if scale_factor == 1.0:
                    curr_left_img = left_img
                    curr_right_img = right_img
                else:
                    curr_left_img = F.interpolate(left_img,
                                                  scale_factor=scale_factor,
                                                  mode='bilinear', align_corners=False)
                    curr_right_img = F.interpolate(right_img,
                                                   scale_factor=scale_factor,
                                                   mode='bilinear', align_corners=False)
                inputs = (disparity, curr_left_img, curr_right_img)
                disparity = self.refinement[i](*inputs)
                disparity_pyramid.append(disparity)  # [H/2, H]

        return disparity_pyramid

    def forward(self, left_img, right_img):

        left_sem_feature, left_disp_feature = self.feature_extraction(left_img)
        right_sem_feature, right_disp_feature = self.feature_extraction(right_img)

        # train semantic with disparity
        if self.opts.train_semantic and self.opts.train_disparity:
            left_disp_fpn_feature = left_disp_feature[:3]
            right_disp_fpn_feature = right_disp_feature[:3]

            cost_volume = self.cost_volume_construction(left_disp_fpn_feature, right_disp_fpn_feature)
            aggregation = self.aggregation(cost_volume)     # [Bx64xH/4xW/4, Bx128xH/8xW/8, Bx256xH/16xW/16]

            disparity_pyramid = self.disparity_computation(aggregation)

            if self.opts.with_refine:
                if self.refinement_type == 'hourglass':
                    disparity_pyramid += self.disparity_refinement(left_img, right_img,
                                                                   disparity_pyramid[-1])
                elif self.refinement_type == 'ours':
                    disp, sem = self.refinement_new(disparity_pyramid[-1], left_img, left_sem_feature)
                    disparity_pyramid += [disp]
                    left_sem_feature = left_sem_feature + sem

            left_segmentation = self.predict_segmentation(left_sem_feature)
            # right_segmentation = self.predict_segmentation(right_sem_feature)

            left_segmentation = upsample(left_segmentation, left_img.shape[2:])

            return disparity_pyramid, left_segmentation

        # train semantic only
        elif self.opts.train_semantic and not self.opts.train_disparity:
            left_segmentation = self.predict_segmentation(left_sem_feature)
            left_segmentation = upsample(left_segmentation, left_img.shape[2:])

            return left_segmentation

        # train disparity only
        elif self.opts.train_disparity and not self.opts.train_semantic:
            left_disp_fpn_feature = left_disp_feature[:3]
            right_disp_fpn_feature = right_disp_feature[:3]

            cost_volume = self.cost_volume_construction(left_disp_fpn_feature, right_disp_fpn_feature)
            aggregation = self.aggregation(cost_volume)

            disparity_pyramid = self.disparity_computation(aggregation)

            if self.opts.with_refine:
                disparity_pyramid += self.disparity_refinement(left_img, right_img,
                                                               disparity_pyramid[-1])

            return disparity_pyramid

        else:
            raise NotImplementedError

    def random_init_params(self):
        if self.opts.train_semantic:
            return chain(*([self.segmentation.parameters(), self.feature_extractor.random_init_params()]))
        elif self.opts.train_disparity:
            return chain(*([self.feature_extractor.random_init_params()]))
        else:
            raise NotImplementedError

    def fine_tune_params(self):
        return self.feature_extractor.fine_tune_params()

    def disparity_params(self):
        params = [self.cost_volume.parameters(),
                  self.aggregation.parameters(), self.disparity_estimation.parameters()]
        if self.opts.with_refine:
            if self.refinement_type == 'ours':
                params += [self.refinement_new.parameters()]
            else:
                params += [self.refinement.parameters()]
        return chain(*(params))
