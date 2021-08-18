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
│   │   └── colored_0
│   │   └── colored_1
│   │   └── disp_occ
│   ├── testing
│   │   └── colored_0
│   │   └── colored_1
├── kitti_2015
│   ├── training
│   │   └── image_2
│   │   └── image_3
│   │   └── disp_occ_0
│   │   └── semantic
│   ├── testing
│   │   └── image_2
│   │   └── image_3
└── cityscapes
    ├── leftImg8bit
    │   ├── train
    │   └── val
    │   └── test
    ├── rightImg8bit
    │   ├── train
    │   └── val
    │   └── test
    └── disparity
    │   ├── train
    │   └── val
    │   └── test
    ├── gtFine
        ├── train
        └── val
        └── test          
```


If you want to use multi-dataset training, mix Cityscapes and Lost and Found datasets and the directory structure should be like this:
    
    ├─disparity
    │  ├─test
    │  │  ├─berlin
    │  │  ├─bielefeld
    │  │  ├─bonn
    │  │  ├─...
    │  │  └─munich
    │  ├─train
    │  │  ├─01_Hanns_Klemm_Str_45
    │  │  ├─03_Hanns_Klemm_Str_19
    │  │  ├─...
    │  │  └─zurich
    │  └─val
    │      ├─02_Hanns_Klemm_Str_44
    │      ├─04_Maurener_Weg_8
    │      ├─05_Schafgasse_1
    │      ├─...
    │      └─munster
    ├─gtFine
    │  ├─train
    │  │  ├─01_Hanns_Klemm_Str_45
    │  │  ├─03_Hanns_Klemm_Str_19
    │  │  ├─...
    │  │  └─zurich
    │  └─val
    │      ├─02_Hanns_Klemm_Str_44
    │      ├─04_Maurener_Weg_8
    │      ├─05_Schafgasse_1
    │      ├─...
    │      └─munster
    └─leftImg8bit
        ├─test
        │  ├─berlin
        │  ├─bielefeld
        │  ├─bonn
        │  ├─...
        │  └─munich
        ├─train
        │  ├─01_Hanns_Klemm_Str_45
        │  ├─03_Hanns_Klemm_Str_19
        │  ├─...
        │  └─zurich
        └─val
            ├─02_Hanns_Klemm_Str_44
            ├─04_Maurener_Weg_8
            ├─05_Schafgasse_1
            ├─...
            └─munster



