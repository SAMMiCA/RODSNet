import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import datetime
import utils
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt

from network.warp import disp_warp
from metrics.disparity_metric import d1_metric, thres_metric
from utils.init_trainer import InitOpts
from utils.loss import SegmentationLosses, DisparityLosses, get_smooth_loss


class Trainer(InitOpts):
    def __init__(self, options):
        super().__init__(options)

    def train(self):
        interval_loss = 0.0
        train_epoch_loss = 0.0
        print_cycle = 0.0
        data_cycle = 0.0

        # empty the cache
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

        # switch to train mode
        self.model.train()
        num_img_tr = len(self.train_loader)

        if self.opts.train_semantic:
            self.criterion.step_counter = 0

        # Learning rate summary
        base_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('base_lr', base_lr, self.cur_epochs)

        self.evaluator.reset()

        last_data_time = time.time()
        for i, sample in enumerate(self.train_loader):
            data_loader_time = time.time() - last_data_time
            data_cycle += data_loader_time
            self.num_iter += 1
            model_start_time = time.time()

            left = sample['left'].to(self.device, dtype=torch.float32)
            right = sample['right'].to(self.device, dtype=torch.float32)

            if 'label' in sample.keys():
                labels = sample['label'].to(self.device, dtype=torch.long)
            if 'disp' in sample.keys():
                gt_disp = sample['disp'].to(self.device)
                mask = (gt_disp > 0) & (gt_disp < self.opts.max_disp)
                if not mask.any():
                    continue
            if 'pseudo_disp' in sample.keys():
                pseudo_gt_disp = sample['pseudo_disp'].to(self.device)
                pseudo_mask = (pseudo_gt_disp > 0) & (pseudo_gt_disp < self.opts.max_disp) & (~mask)

            self.optimizer.zero_grad()

            if self.opts.train_semantic and self.opts.train_disparity:
                pred_disp_pyramid, left_seg = self.model(left, right)
                pred_disp = pred_disp_pyramid[-1]
                pyramid_weight = self.pyramid_weight[len(pred_disp_pyramid)]
                assert len(pyramid_weight) == len(pred_disp_pyramid)

                disp_loss = DisparityLosses(pyramid_weight, pred_disp_pyramid, gt_disp, mask)
                if self.opts.dataset == 'kitti_mix':
                    loss2 = 0
                else:
                    loss2 = self.criterion(left_seg, labels, sample)

                pred_disp = self.resize_pred_disp(pred_disp, gt_disp)
                if 'pseudo_disp' in sample.keys() and self.opts.load_pseudo_gt:
                    pseudo_curr_loss = DisparityLosses(pyramid_weight, pred_disp_pyramid, pseudo_gt_disp, pseudo_mask)
                    total_loss = (disp_loss * self.opts.disp_weight) + \
                                 (loss2 * self.opts.sem_weight) + \
                                 pseudo_curr_loss * self.opts.pseudo_disp_weight
                else:
                    total_loss = (disp_loss * self.opts.disp_weight) + (loss2 * self.opts.sem_weight)

            elif self.opts.train_semantic and not self.opts.train_disparity:
                left_seg = self.model(left, right)
                loss2 = self.criterion(left_seg, labels, sample)
                total_loss = loss2
                disp_loss, pyramid_loss = 0, 0
                pred_disp = None

            elif not self.opts.train_semantic and self.opts.train_disparity:
                pred_disp_pyramid = self.model(left, right)
                pred_disp = pred_disp_pyramid[-1]
                pyramid_weight = self.pyramid_weight[len(pred_disp_pyramid)]
                assert len(pyramid_weight) == len(pred_disp_pyramid)
                disp_loss = DisparityLosses(pyramid_weight, pred_disp_pyramid, gt_disp, mask)
                pred_disp = self.resize_pred_disp(pred_disp, gt_disp)
                total_loss = disp_loss
            else:
                raise NotImplementedError

            interval_loss += total_loss
            train_epoch_loss += total_loss

            total_loss.backward()
            self.optimizer.step()
            one_cycle_time = time.time() - model_start_time
            print_cycle += one_cycle_time

            if self.num_iter % self.opts.print_freq == 0:
                interval_loss = interval_loss / self.opts.print_freq
                print("Epoch: [%3d/%3d] Itrs: [%5d/%5d] dataloader time : %4.2fs training time: %4.2fs time_per_img: %4.2fs Loss=%f" %
                      (self.cur_epochs, self.opts.epochs, i, num_img_tr, data_cycle, print_cycle,
                       print_cycle/self.opts.print_freq/self.opts.batch_size, interval_loss))
                self.writer.add_scalar('train/total_loss_iter', interval_loss, self.num_iter)
                interval_loss = 0.0
                print_cycle = 0.0
                data_cycle = 0.0

            if self.num_iter % self.opts.summary_freq == 0:
                summary_time_start = time.time()
                # calculate segmentation scores
                if self.opts.dataset != 'kitti_mix':
                    if self.opts.train_semantic and self.opts.train_disparity:
                        preds = left_seg.detach().max(dim=1)[1].cpu().numpy()
                        targets = labels.cpu().numpy()
                        self.evaluator.add_batch(targets, preds)
                        score = self.evaluator.get_results()
                    elif self.opts.train_semantic and not self.opts.train_disparity:
                        preds = left_seg.detach().max(dim=1)[1].cpu().numpy()
                        targets = labels.cpu().numpy()
                        self.evaluator.add_batch(targets, preds)
                        score = self.evaluator.get_results()
                    elif not self.opts.train_semantic and self.opts.train_disparity:
                        score = None
                    else:
                        raise NotImplementedError
                else:
                    score = None

                # calculate disparity accuracy
                self.performance_check_train(disp_loss, total_loss, pred_disp, gt_disp, mask, score)
                summary_time = time.time() - summary_time_start
                print("summary_time : {}".format(summary_time))
            last_data_time = time.time()
            del total_loss, sample

        train_epoch_loss = train_epoch_loss / num_img_tr
        self.writer.add_scalar('train/total_loss_epoch', train_epoch_loss, self.cur_epochs)

    def validate(self):
        """Do validation and return specified samples"""
        print("validation...")
        self.evaluator.reset()
        img_id = 0
        self.time_val = []

        # empty the cache to infer in high res
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        # switch to evaluate mode
        self.model.eval()

        val_epe, val_d1, val_thres1, val_thres2, val_thres3 = 0, 0, 0, 0, 0
        valid_samples = 0
        score = 0
        mean_epe, mean_d1, mean_thres1, mean_thres2, mean_thres3 = 0, 0, 0, 0, 0
        num_val = len(self.val_loader)

        with torch.no_grad():
            start = time.time()
            for i, sample in enumerate(self.val_loader):
                data_time = time.time() - start
                self.time_val_dataloader.append(data_time)

                left = sample['left'].to(self.device, dtype=torch.float32)
                right = sample['right'].to(self.device, dtype=torch.float32)

                # # Warm-up
                # if i == 0:
                #     with torch.no_grad():
                #             self.model(left, right)

                if 'label' in sample.keys():
                    labels = sample['label'].to(self.device, dtype=torch.long)
                if 'disp' in sample.keys():
                    gt_disp = sample['disp'].to(self.device)
                    mask = (gt_disp > 0) & (gt_disp < self.opts.max_disp)
                    if not mask.any():
                        continue
                if 'pseudo_disp' in sample.keys():
                    pseudo_gt_disp = sample['pseudo_disp'].to(self.device)
                    pseudo_mask = (pseudo_gt_disp > 0) & (pseudo_gt_disp < self.opts.max_disp) & (~mask)  # inverse mask

                valid_samples += 1
                start_time = time.time()
                if self.opts.train_semantic and self.opts.train_disparity:
                    pred_disp_pyramid, left_seg = self.model(left, right)
                    model_time = time.time()
                    fwt = model_time - start_time
                    pred_disp = pred_disp_pyramid[-1]
                    pred_disp = self.resize_pred_disp(pred_disp, gt_disp)

                    val_epe, val_d1, val_thres1, val_thres2, val_thres3 = \
                        self.calculate_disparity_error(pred_disp, gt_disp, mask,
                                                       val_epe, val_d1, val_thres1, val_thres2, val_thres3)

                    if self.opts.dataset != 'kitti_mix':
                        preds = left_seg.detach().max(dim=1)[1].cpu().numpy()
                        targets = labels.cpu().numpy()
                        self.evaluator.add_batch(targets, preds)
                    evaluate_time = time.time() - model_time
                    # print('calculate mIoU accuracy time1 : %.3f, time2 : %.3f' % (time1 - time0, time.time()-time1))
                elif self.opts.train_semantic and not self.opts.train_disparity:
                    left_seg = self.model(left, right)
                    fwt = time.time() - start_time
                    preds = left_seg.detach().max(dim=1)[1].cpu().numpy()
                    targets = labels.cpu().numpy()
                    self.evaluator.add_batch(targets, preds)
                    # print('calculate mIoU accuracy time1 : %.3f, time2 : %.3f' % (time1 - time0, time.time()-time1))
                    pred_disp, gt_disp = None, None
                elif not self.opts.train_semantic and self.opts.train_disparity:
                    pred_disp_pyramid = self.model(left, right)
                    fwt = time.time() - start_time
                    pred_disp = pred_disp_pyramid[-1]
                    targets, preds = None, None     # semantic label & results
                    pred_disp = self.resize_pred_disp(pred_disp, gt_disp)

                    val_epe, val_d1, val_thres1, val_thres2, val_thres3 = \
                        self.calculate_disparity_error(pred_disp, gt_disp, mask,
                                                       val_epe, val_d1, val_thres1, val_thres2, val_thres3)
                else:
                    raise NotImplementedError

                # first batch stucked on some process.. --> time cost is wierd on i==0
                if i != 0:
                    self.time_val.append(fwt)
                    if i % self.opts.val_print_freq == 0:
                        # check validation fps
                        print("[%d/%d] Model passed time (bath size=%d): %.3f (Mean time per img: %.3f), Dataloader time : %.3f" % (
                            i, num_val,
                            self.opts.val_batch_size, fwt,
                            sum(self.time_val) / len(self.time_val) / self.opts.val_batch_size, data_time))

                if self.opts.save_val_results:
                    # save all validation results images
                    if 'pseudo_disp' in sample.keys():
                        if self.opts.dataset == 'kitti_2015':
                            self.save_valid_img_in_results_kitti(left, right, targets, preds, pred_disp, gt_disp, i,
                                                           pseudo_gt_disp)
                        else:
                            self.save_valid_img_in_results(left, right, targets, preds, pred_disp, gt_disp, i,
                                                           pseudo_gt_disp)
                    else:
                        if self.opts.dataset == 'kitti_2015':
                            self.save_valid_img_in_results_kitti(left, right, targets, preds, pred_disp, gt_disp, i)
                        else:
                            self.save_valid_img_in_results(left, right, targets, preds, pred_disp, gt_disp, i)
                    img_id += 1

                start = time.time()
            del sample

        # test validation performance
        if self.opts.train_semantic and self.opts.dataset != 'kitti_mix':
            score = self.evaluator.get_results()

        if self.opts.train_disparity:
            mean_epe = val_epe / valid_samples
            mean_d1 = val_d1 / valid_samples
            mean_thres1 = val_thres1 / valid_samples
            mean_thres2 = val_thres2 / valid_samples
            mean_thres3 = val_thres3 / valid_samples

        self.performance_test(score, mean_epe, mean_d1, mean_thres1, mean_thres2, mean_thres3)

        if not self.opts.test_only:
            self.save_checkpoints(mean_epe, score)

            if self.opts.train_semantic and self.opts.dataset != 'kitti_mix':
                if score['Mean IoU'] > self.best_score:  # save best model
                    self.best_score = score['Mean IoU']
                    self.best_score_epoch = self.cur_epochs
                    self.save_checkpoints(mean_epe, score, is_best=True, best_type='score')
                print('\nbest score epoch: {}, best score: {}'.format(self.best_score_epoch, self.best_score))

            if self.opts.train_disparity:
                if self.opts.dataset == 'kitti_2015' or self.opts.dataset == 'kitti_mix':
                    # compare d1 score (save variable name as epe for convenient)
                    if mean_d1 < self.best_epe:
                        self.best_epe = mean_d1
                        self.best_epe_epoch = self.cur_epochs
                        self.save_checkpoints(mean_d1, score, is_best=True, best_type='epe')
                    print('best d1 epoch: {}, best d1: {}'.format(self.best_epe_epoch, self.best_epe))
                else:
                    if mean_epe < self.best_epe:
                        self.best_epe = mean_epe
                        self.best_epe_epoch = self.cur_epochs
                        self.save_checkpoints(mean_epe, score, is_best=True, best_type='epe')
                    print('best epe epoch: {}, best epe: {}'.format(self.best_epe_epoch, self.best_epe))

    def test(self):
        self.validate()

    def save_checkpoints(self, epe, score, is_best=False, best_type=None):
        if self.n_gpus > 1:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        self.saver.save_checkpoint({
            'epoch': self.cur_epochs,
            "num_iter": self.num_iter,
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'epe': epe,
            'score': score,
            'best_score': self.best_score,
            'best_epe': self.best_epe,
            'best_score_epoch': self.best_score_epoch,
            'best_epe_epoch': self.best_epe_epoch
        }, is_best, best_type)

    def performance_check_train(self, disp_loss, total_loss, pred_disp, gt_disp, mask, score):
        if self.opts.train_semantic and self.opts.dataset != 'kitti_mix':
            self.writer.add_scalar('train/mIoU', score["Mean IoU"], self.num_iter)
            self.writer.add_scalar('train/OverallAcc', score["Overall Acc"], self.num_iter)
            self.writer.add_scalar('train/MeanAcc', score["Mean Acc"], self.num_iter)
            self.writer.add_scalar('train/fwIoU', score["FreqW Acc"], self.num_iter)

        if self.opts.train_disparity:
            # pred_disp = pred_disp.squeeze(1)

            epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
            self.writer.add_scalar('train/epe', epe.item(), self.num_iter)
            self.writer.add_scalar('train/disp_loss', disp_loss.item(), self.num_iter)
            self.writer.add_scalar('train/total_loss', total_loss.item(), self.num_iter)

            d1 = d1_metric(pred_disp, gt_disp, mask)
            self.writer.add_scalar('train/d1', d1.item(), self.num_iter)
            thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
            thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
            thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
            self.writer.add_scalar('train/thres1', thres1.item(), self.num_iter)
            self.writer.add_scalar('train/thres2', thres2.item(), self.num_iter)
            self.writer.add_scalar('train/thres3', thres3.item(), self.num_iter)


    def performance_test(self, val_score, mean_epe, mean_d1, mean_thres1, mean_thres2, mean_thres3):
        print('Validation:')
        print('[Epoch: %d]' % (self.cur_epochs))

        if self.opts.train_semantic and self.opts.dataset != 'kitti_mix':
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

            if not self.opts.test_only:
                self.writer.add_scalar('val/mIoU', mIoU, self.cur_epochs)
                self.writer.add_scalar('val/Acc', Acc, self.cur_epochs)
                self.writer.add_scalar('val/Acc_class', Acc_class, self.cur_epochs)
                self.writer.add_scalar('val/fwIoU', FWIoU, self.cur_epochs)

            print(self.evaluator.to_str(val_score))
        else:
            mIoU, Acc, Acc_class, FWIoU = 0, 0, 0, 0

        if self.opts.train_disparity:
            if not self.opts.test_only:
                self.writer.add_scalar('val/epe', mean_epe, self.cur_epochs)
                self.writer.add_scalar('val/d1', mean_d1, self.cur_epochs)
                self.writer.add_scalar('val/thres1', mean_thres1, self.cur_epochs)
                self.writer.add_scalar('val/thres2', mean_thres2, self.cur_epochs)
                self.writer.add_scalar('val/thres3', mean_thres3, self.cur_epochs)

        self.saver.save_val_results(self.cur_epochs, mean_epe, mean_d1,
                                    mean_thres1, mean_thres2, mean_thres3,
                                    mIoU, Acc)

        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {},"
              " epe:{}, d1:{}, thres1:{}, thres2:{} thres3:{}".format(Acc, Acc_class, mIoU, FWIoU,
                                                                      mean_epe, mean_d1,
                                                                      mean_thres1, mean_thres2, mean_thres3))

    def calculate_disparity_error(self, pred_disp, gt_disp, mask,
                                  val_epe, val_d1, val_thres1, val_thres2, val_thres3):
        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
        thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres1 += thres1.item()
        val_thres2 += thres2.item()
        val_thres3 += thres3.item()

        return val_epe, val_d1, val_thres1, val_thres2, val_thres3

    def make_directory(self, root, folders):
        if not os.path.exists(os.path.join(root, folders)):
            os.mkdir(os.path.join(root, folders))

    def save_valid_img_in_results(self, left, right, targets, preds, pred_disp, gt_disp, img_id, pseudo_disp=None):
        if not os.path.exists(os.path.join(self.saver.experiment_dir, 'results')):
            os.mkdir(os.path.join(self.saver.experiment_dir, 'results'))

        root_dir = os.path.join(self.saver.experiment_dir, 'results')
        self.make_directory(root_dir, 'left_image')
        self.make_directory(root_dir, 'right_image')
        self.make_directory(root_dir, 'gt_sem')
        self.make_directory(root_dir, 'pred_sem')
        self.make_directory(root_dir, 'overlay')

        # for i in range(len(left)):
        i = 0
        image = left[i].detach().cpu().numpy()
        right_image = right[i].detach().cpu().numpy()
        # image = (self.denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
        image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).transpose(1, 2, 0).astype(np.uint8)
        image = Image.fromarray(image)
        if self.opts.dataset == 'kitti_2015':
            image = image.crop((0, 8, 1242, 8 + 375))
        image.save(os.path.join(self.saver.experiment_dir, 'results', 'left_image', '%d_left_image.png' % img_id))

        right_image = (
                    (right_image - np.min(right_image)) / (np.max(right_image) - np.min(right_image)) * 255).transpose(
            1, 2, 0).astype(
            np.uint8)
        right_image = Image.fromarray(right_image)
        if self.opts.dataset == 'kitti_2015':
            right_image = right_image.crop((0, 8, 1242, 8 + 375))
        right_image.save(
            os.path.join(self.saver.experiment_dir, 'results', 'right_image', '%d_right_image.png' % img_id))

        if self.opts.train_semantic:
            target = targets[i]
            target = self.val_loader.dataset.decode_target(target).astype(np.uint8)
            target = Image.fromarray(target)
            if self.opts.dataset == 'kitti_2015':
                target = target.crop((0, 8, 1242, 8 + 375))
            target.save(
                os.path.join(self.saver.experiment_dir, 'results', 'gt_sem', '%d_gt_sem.png' % img_id))

            pred = preds[i]
            pred = self.val_loader.dataset.decode_target(pred).astype(np.uint8)
            pred = Image.fromarray(pred)
            if self.opts.dataset == 'kitti_2015':
                pred = pred.crop((0, 8, 1242, 8 + 375))
            pred.save(os.path.join(self.saver.experiment_dir, 'results', 'pred_sem', '%d_pred_sem.png' % img_id))

            overlay = Image.blend(image, pred, alpha=0.7)
            overlay.save(os.path.join(self.saver.experiment_dir, 'results', 'overlay', '%d_overlay.png' % img_id))

        if self.opts.train_disparity:
            self.make_directory(root_dir, 'pred_disp')
            self.make_directory(root_dir, 'gt_disp')
            self.make_directory(root_dir, 'pred_disp_magma')
            self.make_directory(root_dir, 'gt_disp_magma')

            pred_disp_ = pred_disp[i]
            pred_disp_ = (pred_disp_.detach().cpu().numpy()).astype(np.uint8)
            gt_disp_ = (gt_disp[i].detach().cpu().numpy()).astype(np.uint8)

            pred_disp_I = Image.fromarray(pred_disp_)
            gt_disp_I = Image.fromarray(gt_disp_)
            if self.opts.dataset == 'kitti_2015':
                pred_disp_I = pred_disp_I.crop((0, 8, 1242, 8 + 375))
                gt_disp_I = gt_disp_I.crop((0, 8, 1242, 8 + 375))
            pred_disp_I.save(
                os.path.join(self.saver.experiment_dir, 'results', 'pred_disp', '%d_pred_disp.png' % img_id))
            gt_disp_I.save(os.path.join(self.saver.experiment_dir, 'results', 'gt_disp', '%d_gt_disp.png' % img_id))

            vmax = np.percentile(pred_disp_, 95)
            normalizer = mpl.colors.Normalize(vmin=pred_disp_.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            out_img = (mapper.to_rgba(pred_disp_)[:, :, :3] * 255).astype(np.uint8)
            out_img = Image.fromarray(out_img)
            if self.opts.dataset == 'kitti_2015':
                out_img = out_img.crop((0, 8, 1242, 8 + 375))

            out_img.save(
                os.path.join(self.saver.experiment_dir, 'results', 'pred_disp_magma',
                             '%d_pred_disp_magma.png' % img_id))

            vmax = np.percentile(gt_disp_, 95)
            normalizer = mpl.colors.Normalize(vmin=gt_disp_.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            out_img = (mapper.to_rgba(gt_disp_)[:, :, :3] * 255).astype(np.uint8)
            out_img = Image.fromarray(out_img)
            if self.opts.dataset == 'kitti_2015':
                out_img = out_img.crop((0, 8, 1242, 8 + 375))
            out_img.save(
                os.path.join(self.saver.experiment_dir, 'results', 'gt_disp_magma', '%d_gt_disp_magma.png' % img_id))

            if pseudo_disp is not None:
                self.make_directory(root_dir, 'gt_pseudo_disp')
                self.make_directory(root_dir, 'gt_pseudo_disp_magma')

                pseudo_disp_ = (pseudo_disp[0].detach().cpu().numpy()).astype(np.uint8)
                pseudo_disp_I = Image.fromarray(pseudo_disp_)
                if self.opts.dataset == 'kitti_2015':
                    pseudo_disp_I = pseudo_disp_I.crop((0, 8, 1242, 8 + 375))
                pseudo_disp_I.save(
                    os.path.join(self.saver.experiment_dir, 'results', 'gt_pseudo_disp',
                                 '%d_gt_pseudo_disp.png' % img_id))

                vmax = np.percentile(pseudo_disp_, 95)
                normalizer = mpl.colors.Normalize(vmin=pseudo_disp_.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                out_img = (mapper.to_rgba(pseudo_disp_)[:, :, :3] * 255).astype(np.uint8)
                out_img = Image.fromarray(out_img)
                if self.opts.dataset == 'kitti_2015':
                    out_img = out_img.crop((0, 8, 1242, 8 + 375))
                out_img.save(
                    os.path.join(self.saver.experiment_dir, 'results', 'gt_pseudo_disp_magma',
                                 '%d_gt_pseudo_disp_magma.png' % img_id))


    def save_valid_img_in_results_kitti(self, left, right, targets, preds, pred_disp, gt_disp, img_id, pseudo_disp=None):
        if not os.path.exists(os.path.join(self.saver.experiment_dir, 'results')):
            os.mkdir(os.path.join(self.saver.experiment_dir, 'results'))

        root_dir = os.path.join(self.saver.experiment_dir, 'results')
        self.make_directory(root_dir, 'left_image')
        self.make_directory(root_dir, 'right_image')
        self.make_directory(root_dir, 'gt_sem')
        self.make_directory(root_dir, 'pred_sem')
        self.make_directory(root_dir, 'overlay')

        # draw first images in each batch to save memory
        i = 0
        image = left[i].detach().cpu().numpy()
        right_image = right[i].detach().cpu().numpy()
        # image = (self.denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
        image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).transpose(1, 2, 0).astype(np.uint8)
        image = Image.fromarray(image)
        if self.opts.dataset == 'kitti_2015':
            image = image.crop((0, 8, 1242, 8 + 375))
        image.save(os.path.join(self.saver.experiment_dir, 'results', 'left_image', '%d_left_image.png' % img_id))

        right_image = (
                    (right_image - np.min(right_image)) / (np.max(right_image) - np.min(right_image)) * 255).transpose(
            1, 2, 0).astype(
            np.uint8)
        right_image = Image.fromarray(right_image)
        if self.opts.dataset == 'kitti_2015':
            right_image = right_image.crop((0, 8, 1242, 8 + 375))
        right_image.save(
            os.path.join(self.saver.experiment_dir, 'results', 'right_image', '%d_right_image.png' % img_id))

        if self.opts.train_semantic:
            target = targets[i]
            target = self.val_loader.dataset.decode_target(target).astype(np.uint8)
            target = Image.fromarray(target)
            if self.opts.dataset == 'kitti_2015':
                target = target.crop((0, 8, 1242, 8 + 375))
            target.save(
                os.path.join(self.saver.experiment_dir, 'results', 'gt_sem', '%d_gt_sem.png' % img_id))

            pred = preds[i]
            pred = self.val_loader.dataset.decode_target(pred).astype(np.uint8)
            pred = Image.fromarray(pred)
            if self.opts.dataset == 'kitti_2015':
                pred = pred.crop((0, 8, 1242, 8 + 375))
            pred.save(os.path.join(self.saver.experiment_dir, 'results', 'pred_sem', '%d_pred_sem.png' % img_id))

            overlay = Image.blend(image, pred, alpha=0.7)
            overlay.save(os.path.join(self.saver.experiment_dir, 'results', 'overlay', '%d_overlay.png' % img_id))

        if self.opts.train_disparity:
            self.make_directory(root_dir, 'pred_disp')
            self.make_directory(root_dir, 'gt_disp')
            self.make_directory(root_dir, 'pred_disp_magma')
            self.make_directory(root_dir, 'gt_disp_magma')

            pred_disp_ = pred_disp[i]
            pred_disp_ = (pred_disp_.detach().cpu().numpy()).astype(np.uint8)
            gt_disp_ = (gt_disp[i].detach().cpu().numpy()).astype(np.uint8)

            pred_disp_I = Image.fromarray(pred_disp_)
            gt_disp_I = Image.fromarray(gt_disp_)
            if self.opts.dataset == 'kitti_2015':
                pred_disp_I = pred_disp_I.crop((0, 8, 1242, 8 + 375))
                gt_disp_I = gt_disp_I.crop((0, 8, 1242, 8 + 375))
            pred_disp_I.save(
                os.path.join(self.saver.experiment_dir, 'results', 'pred_disp', '%d_pred_disp.png' % img_id))
            gt_disp_I.save(os.path.join(self.saver.experiment_dir, 'results', 'gt_disp', '%d_gt_disp.png' % img_id))

            pred_disp_color = pred_disp_[8:-1, :1242]
            plt.imshow(pred_disp_color, cmap="magma")
            plt.imsave(os.path.join(self.saver.experiment_dir, 'results', 'pred_disp_magma',
                             '%d_pred_disp_magma.png' % img_id), pred_disp_color)

            gt_disp_color = gt_disp_[8:-1, :1242]
            plt.imshow(gt_disp_color, cmap="magma")
            plt.imsave(os.path.join(self.saver.experiment_dir, 'results', 'gt_disp_magma',
                             '%d_gt_disp_magma.png' % img_id), gt_disp_color)

            if pseudo_disp is not None:
                self.make_directory(root_dir, 'gt_pseudo_disp')
                self.make_directory(root_dir, 'gt_pseudo_disp_magma')

                pseudo_disp_ = (pseudo_disp[0].detach().cpu().numpy()).astype(np.uint8)
                pseudo_disp_I = Image.fromarray(pseudo_disp_)
                if self.opts.dataset == 'kitti_2015':
                    pseudo_disp_I = pseudo_disp_I.crop((0, 8, 1242, 8 + 375))
                pseudo_disp_I.save(
                    os.path.join(self.saver.experiment_dir, 'results', 'gt_pseudo_disp',
                                 '%d_gt_pseudo_disp.png' % img_id))

                pseudo_disp_color = pseudo_disp_[8:-1, :1242]
                plt.imshow(pseudo_disp_color, cmap="magma")
                plt.imsave(os.path.join(self.saver.experiment_dir, 'results', 'gt_pseudo_disp_magma',
                                        '%d_gt_pseudo_disp_magma.png' % img_id), pseudo_disp_color)


    def calculate_estimate(self, epoch, iter):
        num_img_tr = len(self.train_loader)
        num_img_val = len(self.val_loader)
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
                       (num_img_tr * self.opts.epochs - (
                               iter + 1 + epoch * num_img_tr))) + \
                   int(self.batch_time_e.avg * num_img_val * (
                           self.opts.epochs - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    def resize_pred_disp(self, pred_disp, gt_disp):
        if pred_disp.size(-1) < gt_disp.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, (gt_disp.size(-2), gt_disp.size(-1)),
                                      mode='bilinear', align_corners=False) * (
                                gt_disp.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]
        return pred_disp