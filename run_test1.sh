#!/bin/bash

python finetune.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /cluster/scratch/zhangga/dataset/ \
               --epochs 100 \
               --savemodel /cluster/scratch/zhangga/finetune \
               --loadmodel /cluster/scratch/zhangga/trained/new_psm/checkpoint_79.tar \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 0 \
             
               
	       #--loadmodel ./trained/gwc_dilated_seg/checkpoint_30.tar \

