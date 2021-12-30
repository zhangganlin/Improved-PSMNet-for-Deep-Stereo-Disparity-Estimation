#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /cluster/scratch/zhangga/dataset/ \
               --epochs 30 \
               --savemodel /cluster/scratch/zhangga/trained/new_psm/ \
               --loadmodel /cluster/scratch/zhangga/trained/new_psm/checkpoint_49.tar \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 50 \
             
               
	       #--loadmodel ./trained/gwc_dilated_seg/checkpoint_30.tar \

