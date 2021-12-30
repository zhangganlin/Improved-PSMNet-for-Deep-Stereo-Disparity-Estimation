#!/bin/bash

python main.py --maxdisp 192 \
               --model dilated \
               --datapath /cluster/scratch/zhangga/dataset/ \
               --epochs 50 \
               --savemodel /cluster/scratch/zhangga/trained/dilated_seg/ \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 0 \
               --seg
               
	       #--loadmodel ./trained/gwc_dilated_seg/checkpoint_30.tar \

