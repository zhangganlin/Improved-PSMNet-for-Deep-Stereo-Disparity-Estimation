#!/bin/bash

python finetune.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /cluster/scratch/zhangga/dataset/ \
               --epochs 200 \
               --savemodel /cluster/scratch/zhangga/finetune \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 100 \
               --loadmodel /cluster/scratch/zhangga/trained/new_psm/checkpoint_79.tar \
             
            #    --loadmodel /cluster/scratch/zhangga/finetune/kitticheckpoint_99psm.tar  \
               
	       #--loadmodel ./trained/gwc_dilated_seg/checkpoint_30.tar \

