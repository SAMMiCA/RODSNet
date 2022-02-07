import torch.nn as nn
import torch.nn.functional as F
import torch

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class SemsegCrossEntropy(nn.Module):
    def __init__(self, num_classes=19, ignore_id=255, print_each=20):
        super(SemsegCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.step_counter = 0
        self.print_each = print_each

    def loss(self, y, t):
        if y.shape[2:4] != t.shape[1:3]:
            y = upsample(y, t.shape[1:3])
        return F.cross_entropy(y, target=t, ignore_index=self.ignore_id)

    def forward(self, logits, labels, **kwargs):
        loss = self.loss(logits, labels)
        # if (self.step_counter % self.print_each) == 0:
        #     print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
        self.step_counter += 1
        return loss


class BoundaryAwareFocalLoss(nn.Module):
    def __init__(self, gamma=0, num_classes=19, ignore_id=19, print_each=20, weight=None, device=None, opts=None):
        super(BoundaryAwareFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.print_each = print_each
        self.step_counter = 0
        self.gamma = gamma
        self.weight = weight
        self.device = device
        self.opts = opts

    def forward(self, input, target, batch, **kwargs):
        if input.shape[-2:] != target.shape[-2:]:
            input = upsample(input, target.shape[-2:])
        target[target == self.ignore_id] = 0  # we can do this because alphas are zero in ignore_id places
        label_distance_weight = batch['label_distance_weight'].to(self.device)
        N = (label_distance_weight.data > 0.).sum()
        if N.le(0):
            return torch.zeros(size=(0,), device=self.device, requires_grad=True).sum()
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        alphas = label_distance_weight.view(-1)
        weight = self.weight[target].view(-1).to(self.device)

        if self.opts.with_depth_level_loss:
            disp_distance_weight = batch['disp_distance_weight'].to(self.device)
            betas = disp_distance_weight.view(-1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        if self.opts.without_balancing:
            loss = -1 * torch.exp(self.gamma * (1 - pt)) * logpt
        elif self.opts.without_class_balancing:
            loss = -1 * alphas * torch.exp(self.gamma * (1 - pt)) * logpt
        elif self.opts.without_semantic_border:
            loss = -1 * weight * torch.exp(self.gamma * (1 - pt)) * logpt
        elif self.opts.with_depth_level_loss:
            loss = -1 * weight * alphas * betas * torch.exp(self.gamma * (1 - pt)) * logpt
        else:
            loss = -1 * weight * alphas * torch.exp(self.gamma * (1 - pt)) * logpt
        loss = loss.sum() / N

        # if (self.step_counter % self.print_each) == 0:
        #     print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
        self.step_counter += 1

        return loss


class FocalLoss2(nn.Module):
    def __init__(self, gamma=0, num_classes=19, ignore_id=19, print_each=20, weight=None, device=None):
        super(FocalLoss2, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.print_each = print_each
        self.step_counter = 0
        self.gamma = gamma
        self.weight = weight
        self.device = device

    def forward(self, input, target, batch, **kwargs):
        if input.shape[-2:] != target.shape[-2:]:
            input = upsample(input, target.shape[-2:])
        target[target == self.ignore_id] = 0  # we can do this because alphas are zero in ignore_id places
        label_distance_weight = batch['label_distance_weight'].to(self.device)
        N = (label_distance_weight.data > 0.).sum()
        if N.le(0):
            return torch.zeros(size=(0,), device=self.device, requires_grad=True).sum()
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        alphas = label_distance_weight.view(-1)
        weight = self.weight[target].view(-1).to(self.device)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        loss = -1 * torch.exp(self.gamma * (1 - pt)) * logpt
        loss = loss.sum() / N

        # if (self.step_counter % self.print_each) == 0:
        #     print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
        self.step_counter += 1

        return loss


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, device=None, n_gpus=1): # ignore_index=255
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.device = device
        self.n_gpus = n_gpus

    def build_loss(self, mode='cross_entropy'):
        """Choices: ['cross_entropy' or 'focal_loss']"""
        if mode == 'cross_entropy':
            return self.CrossEntropyLoss
        elif mode == 'focal_loss':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        # if self.cuda:
        #     criterion = criterion.cuda()

        if self.n_gpus > 1:
            criterion = nn.Dataparallel(criterion)
        criterion.to(self.device)

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        # if self.cuda:
        #     criterion = criterion.cuda()

        if self.n_gpus > 1:
            criterion = nn.Dataparallel(criterion)
        criterion.to(self.device)

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


def DisparityLosses(pyramid_weight, pred_disp_pyramid, gt_disp, mask):
    disp_loss = 0

    for k in range(len(pred_disp_pyramid)):
        pred_disp = pred_disp_pyramid[k]
        weight = pyramid_weight[k]

        if pred_disp.size(-1) != gt_disp.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                      mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
                                     reduction='mean')
        disp_loss += weight * curr_loss

    return disp_loss


class DispLosses(nn.Module):
    def __init__(self,weight=None, device=None):
        super(DispLosses, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, pred_disp, gt_disp, mask, **kwargs):
        loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
                                     reduction='mean')

        return loss


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()
