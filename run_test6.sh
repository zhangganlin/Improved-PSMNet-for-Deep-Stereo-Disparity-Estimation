#!/bin/bash

python finetune.py --maxdisp 192 \
               --model dilated \
               --datapath /cluster/scratch/xishen/dataset/ \
               --epochs 50 \
               --savemodel /cluster/scratch/xishen/finetune \
               --loadmodel /cluster/scratch/xishen/trained/gwc_dilated/checkpoint_49.tar \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 0 \
               --gwc \