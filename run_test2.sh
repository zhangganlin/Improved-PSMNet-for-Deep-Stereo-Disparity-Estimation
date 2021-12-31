#!/bin/bash

python finetune.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /cluster/scratch/zhangga/dataset/ \
               --epochs 50 \
               --savemodel /cluster/scratch/zhangga/finetune \
               --loadmodel /cluster/scratch/zhangga/trained/seg/checkpoint_79.tar \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 0 \
               --seg \
               
	       #--loadmodel ./trained/gwc_dilated_seg/checkpoint_30.tar \

