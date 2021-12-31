#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /cluster/scratch/xishen/dataset/ \
               --epochs 50 \
               --savemodel /cluster/scratch/xishen/finetune/ \
               --loadmodel /cluster/scratch/xishen/trained/gwc_seg/checkpoint_79.tar \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 0 \
               --seg \
               --gwc \