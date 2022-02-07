import os
import shutil
import torch
from collections import OrderedDict
import glob
import json
import sys

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        self.experiment_dir = os.path.join(self.args.save_dir, self.experiment_dir)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.save_args(args)
        filename = 'command.txt'
        self.save_command(self.experiment_dir, filename)

    def save_args(self, args, filename='args.json'):
        args_dict = vars(args)
        save_path = os.path.join(self.experiment_dir, filename)


        with open(save_path, 'w') as f:
            json.dump(args_dict, f, indent=4, sort_keys=False)

    def save_command(self, save_path, filename='command_train.txt'):
        command = sys.argv
        save_file = os.path.join(save_path, filename)
        with open(save_file, 'w') as f:
            f.write(' '.join(command))

    def save_checkpoint(self, state, is_best, best_type, filename='latest_checkpoint.pth'):  # filename from .pth.tar change to .pth?
        """Saves checkpoint to disk"""

        save_to_dataset = False

        if save_to_dataset:
            self.experiment_dir = os.path.join('/root/dataset/', self.experiment_dir)
            if not os.path.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)

        if is_best:
            if best_type == 'epe':
                best_pred = state['best_epe']
                with open(os.path.join(self.experiment_dir, 'best_epe_pred.txt'), 'w') as f:
                    typo = str(best_pred) + "in epoch :{}".format(state['best_epe_epoch'])
                    f.write(typo)
                filename = os.path.join(self.experiment_dir, best_type + '_best_checkpoint.pth')
                torch.save(state, filename)

            elif best_type == 'score':
                best_pred = state['best_score']
                with open(os.path.join(self.experiment_dir, 'best_score_pred.txt'), 'w') as f:
                    typo = str(best_pred) + "in epoch :{}".format(state['best_score_epoch'])
                    f.write(typo)
                filename = os.path.join(self.experiment_dir, best_type+'_best_checkpoint.pth')
                torch.save(state, filename)

            elif best_type == 'obs_score':
                best_pred = state['best_obs_score']
                with open(os.path.join(self.experiment_dir, 'best_obs_score_pred.txt'), 'w') as f:
                    typo = str(best_pred) + "in epoch :{}".format(state['best_obs_score_epoch'])
                    f.write(typo)
                filename = os.path.join(self.experiment_dir, best_type+'_best_checkpoint.pth')
                torch.save(state, filename)

            elif best_type == 'epoch':
                epoch_num = state['epoch']
                filename = os.path.join(self.experiment_dir, best_type+ str(epoch_num) + '_checkpoint.pth')
                torch.save(state, filename)

            else:
                raise NotImplementedError

        else:
            filename = os.path.join(self.experiment_dir, filename)
            torch.save(state, filename)


    def save_val_results(self, epoch, epe, d1, thres1, thres2, thres3, mIoU, Acc,
                         filename='val_results.txt'):
        val_file = os.path.join(self.experiment_dir, filename)

        # Save validation results
        with open(val_file, 'a') as f:
            f.write('epoch: %03d\t' % epoch)
            f.write('epe: %.3f\t' % epe)
            f.write('d1: %.4f\t' % d1)
            f.write('thres1: %.4f\t' % thres1)
            f.write('thres2: %.4f\t' % thres2)
            f.write('thres3: %.4f\t' % thres3)
            f.write('mIoU: %.4f\t' % mIoU)
            f.write('Acc: %.4f\n' % Acc)

    def save_file_return(self, filename='val_results.txt'):
        val_file = os.path.join(self.experiment_dir, filename)

        return val_file

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['dataset'] = self.args.dataset
        # p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
