#!/bin/bash

python finetune.py --maxdisp 192 \
               --model dilated \
               --datapath /cluster/scratch/xishen/dataset/ \
               --epochs 200 \
               --savemodel /cluster/scratch/xishen/finetune \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 100 \
               --gwc \
               --loadmodel /cluster/scratch/xishen/trained/gwc_dilated/checkpoint_79.tar \
                #    --loadmodel /cluster/scratch/xishen/finetune/kitticheckpoint_99dilated_gwc.tar \