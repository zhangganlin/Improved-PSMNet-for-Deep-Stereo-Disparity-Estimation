#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /cluster/scratch/xishen/dataset/ \
               --epochs 30 \
               --savemodel /cluster/scratch/xishen/trained/gwc_seg/ \
               --loadmodel /cluster/scratch/xishen/trained/gwc_seg/checkpoint_49.tar \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 50 \
               --seg \
               --gwc \