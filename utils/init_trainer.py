import torch
import torch.nn as nn
from torch.utils import data
from torch.nn.parameter import Parameter

from dataloaders.utils import get_dataset, custom_collate
from metrics import Evaluator, TimeAverageMeter

import utils
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.calculate_weights import calculate_weigths_labels_new
from utils.loss import BoundaryAwareFocalLoss, FocalLoss2

import random
import numpy as np
import os
import re


import network

class InitOpts():
    def __init__(self, options):
        self._init_options(options)
        self._init_gpu_settings()
        self._init_dataloader()
        self._init_model_map()
        self._init_optimizer()
        self._init_criterion()
        self._init_checkpoints()
        self._init_scheduler()
        self._init_saver()
        if not self.opts.test_only:
            self._init_tensorboard_writer()

    def _init_options(self, options):
        self.opts = options

        if self.opts.dataset == 'cityscapes' or self.opts.dataset == 'kitti_2015' or self.opts.dataset == 'kitti_mix':
            self.opts.num_classes = 19
        elif self.opts.dataset == 'city_lost':
            self.opts.num_classes = 20      # 19 cityscapes classes + small obstacle objects
        elif self.opts.dataset == 'sceneflow':
            self.opts.num_classes = 0
        else:
            raise NotImplementedError

        # set data path using opts.dataset name
        self.opts.data_root = os.path.join(self.opts.data_root, self.opts.dataset)

        self.batch_time_t = TimeAverageMeter()
        self.data_time_t = TimeAverageMeter()
        self.batch_time_e = TimeAverageMeter()
        self.time_val = []
        self.time_val_dataloader = []

    def _init_gpu_settings(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opts.gpu_id
        self.device = torch.device('cuda:{}'.format(self.opts.gpu_id) if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % self.device)

        self.n_gpus = len(self.opts.gpu_id)
        print("Number of used GPU : {}".format(self.n_gpus))
        print("Used GPU ID : {}".format(self.opts.gpu_id))

        # Setup random seed
        torch.manual_seed(self.opts.random_seed)
        np.random.seed(self.opts.random_seed)
        random.seed(self.opts.random_seed)
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed(self.opts.random_seed)
        torch.cuda.manual_seed_all(self.opts.random_seed)
        torch.backends.cudnn.determinisitc = True



    def _init_dataloader(self):
        # Setup dataloader
        self.denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if self.opts.test_only:
            self.opts.val_batch_size = 1

        self.train_dst, self.val_dst = get_dataset(self.opts)
        self.train_loader = data.DataLoader(
            self.train_dst, batch_size=self.opts.batch_size, shuffle=True, num_workers=8,
            pin_memory=False,
            drop_last=True, collate_fn=custom_collate)
        self.val_loader = data.DataLoader(
            self.val_dst, batch_size=self.opts.val_batch_size, shuffle=False, num_workers=8,
            pin_memory=False,
            collate_fn=custom_collate)
        print("Dataset: %s, Train set: %d, Val set: %d" %
              (self.opts.dataset, len(self.train_dst), len(self.val_dst)))

    def _init_model_map(self):
        # Set up model
        self.model = network.RODSNet(self.opts,
                                   self.opts.max_disp,
                                   num_classes=self.opts.num_classes,
                                   device=self.device,
                                   refinement_type=self.opts.refinement_type,
                                   )

        num_params = utils.count_parameters(self.model)
        print('=> Number of trainable parameters: %d' % num_params)

    def _init_optimizer(self):
        # Set up metrics
        self.evaluator = Evaluator(self.opts.num_classes, self.opts)

        # Set up optimizer
        if self.opts.optimizer_policy == 'SGD':
            specific_params = list(filter(utils.filter_specific_params,
                                          self.model.named_parameters()))
            base_params = list(filter(utils.filter_base_params,
                                      self.model.named_parameters()))

            feature_extractor_params = list(filter(utils.filter_feature_extractor_params,
                                                   self.model.named_parameters()))

            specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
            base_params = [kv[1] for kv in base_params]
            feature_extractor_params = [kv[1] for kv in feature_extractor_params]
            specific_lr = self.opts.lr * 0.1
            feature_extractor_lr = self.opts.lr

            params_group = [
                {'params': base_params, 'lr': self.opts.lr},
                {'params': specific_params, 'lr': specific_lr},
                {'params': feature_extractor_params, 'lr': feature_extractor_lr},
            ]

            if self.opts.train_semantic:
                semantic_params = list(filter(utils.filter_semantic_params,
                                              self.model.named_parameters()))
                semantic_params = [kv[1] for kv in semantic_params]
                semantic_lr = self.opts.lr * 10

                params_group += [
                    {'params': semantic_params, 'lr': semantic_lr},
                ]

            self.optimizer = torch.optim.SGD(params_group,
                                             lr=self.opts.lr, momentum=0.9, weight_decay=self.opts.weight_decay)
        elif self.opts.optimizer_policy == 'ADAM':
            # Optimizer

            fine_tune_factor = 4
            train_params = [
                {'params': self.model.random_init_params(), 'lr': self.opts.lr,
                 'weight_decay': self.opts.weight_decay},
                {'params': self.model.fine_tune_params(), 'lr': self.opts.lr / fine_tune_factor,
                 'weight_decay': self.opts.weight_decay / fine_tune_factor},

            ]

            if self.opts.train_disparity:
                specific_params = list(filter(utils.filter_specific_params,
                                              self.model.named_parameters()))
                base_params = list(filter(utils.filter_base_params,
                                          self.model.named_parameters()))
                specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
                base_params = [kv[1] for kv in base_params]
                specific_lr = self.opts.lr
                disp_lr = self.opts.lr * 10 / fine_tune_factor
                weight_decay = self.opts.weight_decay / fine_tune_factor

                # if self.opts.transfer_disparity:
                #     specific_lr /= 10
                #     disp_lr /= 10
                #     weight_decay /= 10

                train_params += [
                    {'params': base_params, 'lr': disp_lr,
                     'weight_decay': weight_decay},
                    {'params': specific_params, 'lr': specific_lr / fine_tune_factor,
                     'weight_decay': weight_decay},
                    # {'params': self.model.disparity_params(), 'lr': (self.opts.lr*10) / (fine_tune_factor),
                    #  'weight_decay': self.opts.weight_decay / fine_tune_factor},
                ]

            self.optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.99))

        else:
            raise NotImplementedError

    def _init_criterion(self):
        ''' Set up criterion '''
        # whether to use class balanced weights
        if self.opts.use_balanced_weights and self.opts.train_semantic and self.opts.dataset != 'kitti_mix':
            print('Use balanced weights for unbalanced semantic classes...')
            # classes_weights_path = os.path.join(self.opts.data_root,
            #                                     self.opts.dataset + '_classes_weights_19.npy')
            classes_weights_path = os.path.join(self.opts.data_root,
                                                self.opts.dataset + '_classes_weights_' +
                                                str(self.opts.num_classes) + '_new_raw.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels_new(classes_weights_path, self.opts.dataset,
                                                  self.train_loader, self.opts.num_classes)

            print('Whole_datasets pixel ratio:{}'.format(weight))
            epsilon = self.opts.epsilon  # experimental setup
            weight = 1 / (np.log(1 + epsilon + weight))
            print('refined pixel ratio for class imbalance:{}'.format(weight))
            print('max/min ratio:{} \n'.format(np.max(weight)/np.min(weight)))
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        if self.opts.train_semantic:
            if self.opts.without_balancing:
                self.criterion = FocalLoss2(gamma=.5, num_classes=self.opts.num_classes,
                                            ignore_id=255, weight=weight, device=self.device)
            else:
                self.criterion = BoundaryAwareFocalLoss(gamma=.5, num_classes=self.opts.num_classes,
                                                        ignore_id=255, weight=weight, device=self.device,
                                                        opts=self.opts)

        # disparity Loss weights
        self.pyramid_weight = {
            5: [1 / 3, 2 / 3, 1.0, 1.0, 1.0],
            4: [1 / 3, 2 / 3, 1.0, 1.0],
            3: [1.0, 1.0, 1.0],     # 1 scale only
            2: [1.0, 1.0],
            1: [1.0]                # highest loss only
        }

    def _init_checkpoints(self):
        # Restore
        self.cur_epochs = 0
        self.num_iter = 0

        self.best_epe = 999.0
        self.best_score = 0.0
        self.best_obs_score = 0.0
        self.best_epe_epoch = -1
        self.best_score_epoch = -1
        self.best_obs_score_epoch = -1

        if self.opts.resume is not None:
            self.opts.resume = os.path.join(self.opts.resume_dir, self.opts.resume)
            if not os.path.isfile(self.opts.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.opts.resume))
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(self.opts.resume, map_location=self.device)
            self.model.to(self.device)
            loaded_pt = checkpoint['model_state']
            model_dict = self.model.state_dict()

            if self.opts.continue_training:
                self.opts.start_epoch = checkpoint['epoch'] + 1
                self.cur_epochs = checkpoint['epoch'] + 1
                self.num_iter = checkpoint['num_iter'] + 1
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.best_score = checkpoint['best_score']
                self.best_epe = checkpoint['best_epe']
                self.best_score_epoch = checkpoint['best_score_epoch']
                self.best_epe_epoch = checkpoint['best_epe_epoch']

                if 'best_obs_score' in checkpoint.keys():
                    self.best_obs_score = checkpoint['best_obs_score']
                if 'best_obs_score_epoch' in checkpoint.keys():
                    self.best_obs_score_epoch = checkpoint['best_obs_score_epoch']

                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in loaded_pt.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                self.model.load_state_dict(model_dict, strict=False)
                print("Training state restored from %s" % self.opts.resume)
                print("=> loaded checkpoint '{}' (epoch {})".format(self.opts.resume, checkpoint['epoch']))
                print("Resume Training from epochs {}".format(self.cur_epochs))

            elif self.opts.transfer_disparity:
                print("Training state restored from %s for transfer learning" % self.opts.resume)
                print("Training from epochs {}".format(self.cur_epochs))
                pretrained_dict = {k: v for k, v in loaded_pt.items() if k in model_dict if
                                       k.split('.')[0] not in ['feature_extractor', 'refinement']}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                self.model.load_state_dict(model_dict, strict=False)

            else:
                print("Just Testing results of pretrained network model...")
                print("If you want to continue training with checkpoints, add --continue_training options!")
                # 1. filter out unnecessary keys
                pretrained_dict = {}
                for k, v in loaded_pt.items():
                    split_key = k.split('.')
                    main_key = split_key[0]
                    remain_key = split_key[1:]
                    if re.sub(r'[0-9]+', '', main_key) == 'refinement_new':
                        # print('find ' + k)
                        ch_k = 'refinement_new.' + ".".join(remain_key)
                        # print('change it to ' + ch_k)
                        pretrained_dict[ch_k] = v
                    elif (k in model_dict):
                        # print("matched " + k)
                        pretrained_dict[k] = v
                    else:
                        pass
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                self.model.load_state_dict(model_dict, strict=False)

            del checkpoint, pretrained_dict  # free memory
        else:
            print("[!] No checkpoints found, Retrain...")
            self.model.to(self.device)

        if self.opts.dataset == 'kitti_mix':
            # freeze semantic modules
            print("[Freeze semantic segmentation modules in kitti_mix...]")
            for name, p in self.model.named_parameters():
                # "feature_extractor.upsample_blends."
                if "feature_extractor.upsample_blends." in name:
                    print(name)
                    p.requires_grad = False
                if "segmentation." in name:
                    print(name)
                    p.requires_grad = False

    def _init_scheduler(self):
        # lr_min = 1e-6
        lr_min = self.opts.last_lr
        if self.opts.continue_training:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.opts.epochs, lr_min, last_epoch=self.cur_epochs)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.opts.epochs, lr_min)

    def _init_saver(self):
        # Define Saver
        self.saver = Saver(self.opts)
        self.saver.save_experiment_config()

    def _init_tensorboard_writer(self):
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
