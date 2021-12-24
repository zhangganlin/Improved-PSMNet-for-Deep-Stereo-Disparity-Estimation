#!/bin/bash

python Test_img.py --KITTI 2015 \
               --datapath ./dataset/data_scene_flow_2015/testing/ \
               --loadmodel ./trained/checkpoint_49.tar \
               --model stackhourglass \
               --maxdisp 192 \
               --leftimg dataset/data_scene_flow_2015/training/image_2/000006_10.png \
               --rightimg dataset/data_scene_flow_2015/training/image_3/000006_10.png \
               
               #--loadmodel ./trained/checkpoint_10.tar \
