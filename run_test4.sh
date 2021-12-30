#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /cluster/scratch/zhangga/dataset/ \
               --epochs 50 \
               --savemodel /cluster/scratch/zhangga/trained/new_psm/ \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 0 \
             
               
	       #--loadmodel ./trained/gwc_dilated_seg/checkpoint_30.tar \

