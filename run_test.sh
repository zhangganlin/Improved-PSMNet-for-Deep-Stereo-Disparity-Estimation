#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /cluster/scratch/zhangga/dataset/ \
               --epochs 50 \
               --savemodel ./trained/ \
               --batchsize 8 \
               --numworker 4 \
               
               #--loadmodel ./trained/checkpoint_10.tar \

