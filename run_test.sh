#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath dataset/data_scene_flow_2015/training/ \
               --epochs 100 \
               --savemodel ./trained/ \
               --batchsize 8 \
               --numworker 4 \
               
               #--loadmodel ./trained/checkpoint_10.tar \

