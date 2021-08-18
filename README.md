# RODSNet: End-to-end Real-time Obstacle Detection Networkfor Safe Self-driving via Multi-task Learning



## Installation

Our code is based on PyTorch 1.2.0, CUDA 10.0 and python 3.7. 

We recommend using [conda](https://www.anaconda.com/distribution/) for installation: 

```shell
conda env create -f environment.yaml
```

After installing dependencies, build deformable convolution:

```shell
cd network/deform_conv && bash build.sh
```


## Datasets
Download [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), 
,[Cityscapes](https://www.cityscapes-dataset.com/), and [Lost and Found](http://www.6d-vision.com/lostandfounddataset) datasets. 


To simultaneously detect class-agnostic obstacles (from Lost and Found) and 19 annotated labels (from Cityscapes), we created a `city_lost` directory by mixing  cityscapes and Lost and found datasets. 

Our folder structure is as follows:
```
datasets
└── sceneflow
│   ├── Driving
│   │   ├── disparity
│   │   └── frames_finalpass
│   ├── FlyingThings3D
│   │   ├── disparity
│   │   └── frames_finalpass
│   └── Monkaa
│       ├── disparity
│       └── frames_finalpass
├── kitti_2012
│   ├── training
│   │   ├── colored_0
│   │   ├── colored_1
│   │   └── disp_occ
│   └── testing
│       ├── colored_0
│       └── colored_1
├── kitti_2015
│   ├── training
│   │   ├── image_2
│   │   ├── image_3
│   │   ├── disp_occ_0
│   │   └── semantic
│   └── testing
│       ├── image_2
│       └── image_3
└── cityscapes
|   ├── leftImg8bit
|   ├── rightImg8bit
|   ├── disparity
|   └── gtFine
|       ├── train
|       ├── val
|       └── test
└── city_lost
    ├── leftImg8bit
    |   ├── train
    |   |   ├── 01_Hanns_Klemm_Str_45
    |   |   ├── 03_Hanns_Klemm_Str_19
    |   |   ├── ...
    |   |   └── zurich
    |   └── val
    |       ├── 02_Hanns_Klemm_Str_44
    |       ├── 04_Maurener_Weg_8
    |       ├── ...
    |       └── munster
    ├── rightImg8bit
    |   ├── same structure with leftImg8bit
    ├── disparity
    |   ├── same structure with leftImg8bit
    └── gtFine
    |   ├── train
    |   |   ├── aachen
    |   |   ├── bochum
    |   |   ├── ...
    |   |   └── zurich
    |   └── val
    |       ├── frankfurt
    |       ├── lindau
    |       └── munster
    └── gtCoarse
        ├── train
        |   ├── 01_Hanns_Klemm_Str_45
        |   ├── 03_Hanns_Klemm_Str_19
        |   ├── ...
        |   └── 14_Otto_Lilienthal_Str_24
        └── val
            ├── 02_Hanns_Klemm_Str_44
            ├── 04_Maurener_Weg_8
            ├── ...
            └── 15_Rechbergstr_Deckenpfronn
```

## Pretrained weights
All pretrained models are available in [here](link).

We assume the downloaded weights are located under the `$RODSNet/ckpt` directory.



## Training and Evaluation
Detailed commands for training and evaluation are described in `script/train_test_guide.txt`. 

For training our RODSNet on `city_lost` datasets, type below command:
```shell
python main.py --gpu_id 0 --dataset city_lost --checkname resnet18_train_citylost_eps_1e-1_without_transfer \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--train_semantic --train_disparity --with_refine --refinement_type ours --batch_size 4 --val_batch_size 4 \
--epsilon 1e-1
```
Trained results are saved in `$RODSNet/run/[dataset]/[checkname]/experiment_0/` directory.


To enable fast experimenting, evaluation runs on-the-fly without saving the intermediate results. 

If you want to save any results, add `--save_val_results` option.
Then, output results will be saved in `$RODSNet/run/[dataset]/[checkname]/experiment_0/results` folder.


To evaluate our performance on `city_lost` dataset with pretrained results, type below command:
```shell
python main.py --gpu_id 0 --dataset city_lost --checkname city_lost_test \
--with_refine  --refinement_type ours --val_batch_size 1 --train_semantic --train_disparity --epsilon 1e-1 \
--resume ckpt/city_lost/best_model_city_lost/score_best_checkpoint.pth --test_only
```


## Sample Test for fast inference.
```shell
python sample_test.py --gpu_id 0 \
--with_refine \
--refinement_type ours \
--train_disparity --train_semantic \
--resume ckpt/city_lost/best_model_city_lost/score_best_checkpoint.pth
```


## Acknowledgements
Part of the code is adopted from previous works: [AANet](https://github.com/haofeixu/aanet), and [RFNet](https://github.com/AHupuJR/RFNet). The deformable convolution op is taken from [mmdetection](https://github.com/open-mmlab/mmdetection). We thank the original authors for their awesome repos.  

This  work  was  supported  by  theInstitute  for  Information  &  Communications  Technology  Promotion  (IITP)grant funded by the Korea government (MSIT) (No.2020-0-00440, Develop-ment of Artificial Intelligence Technology that Continuously Improves Itselfas the Situation Changes in the Real World).

