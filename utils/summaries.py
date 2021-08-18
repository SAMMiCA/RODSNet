import os
import torch
from torchvision.utils import make_grid
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, opts, writer, dataset, left_image, right_image, target, seg_left, seg_right, global_step, pred_disp, gt_disp, warped_right=None):
        if opts.train_semantic and opts.train_disparity:
            pred_disp = pred_disp.unsqueeze(1)

            grid_image = make_grid(left_image[:3].clone().cpu().data, 7, normalize=True)
            writer.add_image('Left Image', grid_image, global_step)
            grid_image = make_grid(right_image[:3].clone().cpu().data, 7, normalize=True)
            writer.add_image('Right Image', grid_image, global_step)

            grid_image = make_grid(pred_disp[:3].clone().cpu().data, 7, normalize=True)  # normalize=False?
            writer.add_image('Predicted Disparity map', grid_image, global_step)

            grid_image = make_grid(gt_disp[:3].clone().cpu().data, 7, normalize=True)  # normalize=False?
            writer.add_image('GT Disparity map', grid_image, global_step)

            grid_image = make_grid(decode_seg_map_sequence(torch.max(seg_left[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 7, normalize=False, range=(0, 255))
            writer.add_image('Predicted Left label', grid_image, global_step)

            grid_image = make_grid(decode_seg_map_sequence(torch.max(seg_right[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 7, normalize=False, range=(0, 255))
            writer.add_image('Warp Right to Left label', grid_image, global_step)

            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 7, normalize=False, range=(0, 255))
            writer.add_image('GT semantic label', grid_image, global_step)

        elif opts.train_semantic and not opts.train_disparity:
            grid_image = make_grid(left_image[:3].clone().cpu().data, 6, normalize=True)
            writer.add_image('Left Image', grid_image, global_step)

            grid_image = make_grid(decode_seg_map_sequence(torch.max(seg_left[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 6, normalize=False, range=(0, 255))
            writer.add_image('Predicted Left label', grid_image, global_step)

            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 6, normalize=False, range=(0, 255))
            writer.add_image('GT semantic label', grid_image, global_step)

        elif not opts.train_semantic and opts.train_disparity:
            pred_disp = pred_disp.unsqueeze(1)

            grid_image = make_grid(left_image[:3].clone().cpu().data, 5, normalize=True)
            writer.add_image('Left Image', grid_image, global_step)

            grid_image = make_grid(right_image[:3].clone().cpu().data, 5, normalize=True)
            writer.add_image('Right Image', grid_image, global_step)

            grid_image = make_grid(pred_disp[:3].clone().cpu().data, 5, normalize=True)  # normalize=False?
            writer.add_image('Predicted Disparity map', grid_image, global_step)

            # grid_image = make_grid(warped_right[:3].clone().cpu().data, 5, normalize=True)  # normalize=False?
            # writer.add_image('Warped Right map', grid_image, global_step)

            grid_image = make_grid(gt_disp[:3].clone().cpu().data, 5, normalize=True)  # normalize=False?
            writer.add_image('GT Disparity map', grid_image, global_step)

        else:
            raise NotImplementedError


