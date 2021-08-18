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


To simultaneously detect class-agnostic obstacles (from Lost and Found) and 19 annotated labels (from Cityscapes), we created a city_lost directory by mixing  cityscapes and Lost and found datasets. 

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





