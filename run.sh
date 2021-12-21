#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath dataset/data_scene_flow_2015/training/ \
               --epochs 0 \
               --savemodel ./trained/ \
               --loadmodel ./trained/checkpoint_10.tar \
               



python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath dataset/data_scene_flow_2015/training/ \
                   --epochs 300 \
                   --loadmodel ./trained/checkpoint_10.tar \
                   --savemodel ./trained/

