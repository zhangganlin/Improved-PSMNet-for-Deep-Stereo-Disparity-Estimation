#!/bin/bash

python finetune.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /cluster/scratch/zhangga/dataset/ \
               --epochs 200 \
               --savemodel /cluster/scratch/zhangga/finetune \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 100 \
               --seg \
               --loadmodel /cluster/scratch/zhangga/trained/seg/checkpoint_79.tar \
            #    --loadmodel /cluster/scratch/zhangga/finetune/kitticheckpoint_99psm_seg.tar \
               
	       #--loadmodel ./trained/gwc_dilated_seg/checkpoint_30.tar \

