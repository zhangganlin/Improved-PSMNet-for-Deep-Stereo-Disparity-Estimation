#!/bin/bash

python finetune.py --maxdisp 192 \
               --model dilated \
               --gwc \
               --seg \
               --kittidatapath dataset/data_scene_flow_2015/training/ \
               --epochs 300 \
               --savemodel ./trained \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 0 \
               --loadmodel ./trained/dilated_gwc_seg/checkpoint_79.tar \