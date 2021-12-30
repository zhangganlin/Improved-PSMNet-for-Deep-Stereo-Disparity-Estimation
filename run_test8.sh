#!/bin/bash

python main.py --maxdisp 192 \
               --model dilated \
               --datapath /cluster/scratch/xishen/dataset/ \
               --epochs 30 \
               --savemodel /cluster/scratch/xishen/trained/dilated_gwc_seg/ \
               --loadmodel /cluster/scratch/xishen/trained/dilated_gwc_seg/checkpoint_49.tar \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 50 \
               --seg \
               --gwc \