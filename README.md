# Improved PSMNet for Deep Stereo Disparity Estimation

This repository contains the code of Deep Learning course project.

## Conda Virtual Environment
```
conda env create -f deep-learning-env.yaml
```
Activate the conda environment before run the code.
```
conda activate deep-learning
```

## Dataset
* KITTI stereo 2015
* KITTI stereo 2012

```bash
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip
```
Download these two datasets and extract them into ```dataset``` folder. The folder structure should be as follow:
```
Improved-PSMNet-for-Deep-Stereo-Disparity-Estimation
│   ...
│      
└───dataset
    └───data_scene_flow_2015
    │   └───testing   
    │   └───training
    └───data_stereo_flow_2012
        └───testing   
        └───training
```

## Test
Make sure that the GPU is available. When test the network, run
```bash
bash run_test.sh
```

## Euler Cluster
Some notes about how to train the network on ETHZ's Euler Cluster are listed in [how-to-hand-in-job.md](how-to-hand-in-job.md)