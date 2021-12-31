#!/bin/bash

python finetune.py --maxdisp 192 \
               --model dilated \
               --datapath /cluster/scratch/xishen/dataset/ \
               --epochs 50 \
               --savemodel /cluster/scratch/xishen/finetune \
               --loadmodel /cluster/scratch/xishen/trained/dilated_gwc_seg/checkpoint_49.tar \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 0 \
               --seg \
               --gwc \