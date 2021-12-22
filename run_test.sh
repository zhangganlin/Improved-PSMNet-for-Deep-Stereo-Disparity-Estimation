#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /cluster/scratch/zhangga/dataset/ \
               --epochs 1 \
               --savemodel ./trained/ \
               
               #--loadmodel ./trained/checkpoint_10.tar \

