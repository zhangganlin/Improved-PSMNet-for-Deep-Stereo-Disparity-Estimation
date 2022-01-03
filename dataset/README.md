# Dataset
* KITTI stereo 2015 
  
  ([http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php))
* Scene Flow (driving part) 
  
  ([https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html))

Only left image, right image and groundtruth disparity are needed. 

Download these two datasets and extract them into ```dataset``` folder. The folder structure should be as follow:
```
dataset
├── data_scene_flow_2015
│   ├── testing
│   │   ├── image_2
│   │   └── image_3
│   └── training
│       ├── disp_occ_0
│       ├── image_2
│       └── image_3
├── driving_disparity
│   └── 35mm_focallength
│       └── scene_forwards
│           ├── fast
│           │   ├── left
│           │   └── right
│           └── slow
│               ├── left
│               └── right
└── driving_frame_cleanpass
    └── 35mm_focallength
        └── scene_forwards
            ├── fast
            │   ├── left
            │   └── right
            └── slow
                ├── left
                └── right
```
