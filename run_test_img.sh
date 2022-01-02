#!/bin/bash

python Test_img.py --loadmodel ./trained/gwc_seg/checkpoint_79.tar \
                   --model stackhourglass \
                   --maxdisp 192 \
                   --leftimg dataset/data_scene_flow_2015/training/image_2/000006_10.png \
                   --rightimg dataset/data_scene_flow_2015/training/image_3/000006_10.png \
                   --segimg dataset/data_scene_flow_2015/training/seg/000006_10.png \
                   --no-cuda \
                   --gwc \
                   --seg
               
