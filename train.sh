#!/bin/bash
python generate_seg.py --kitti15 ./dataset/data_scene_flow_2015/training \
                       --driving ./dataset \

MODEL_PATH=trained/dilated_gwc_seg
if [ ! -e $MODEL_PATH ]; then
  mkdir -p $MODEL_PATH
fi

python main.py --maxdisp 192 \
               --model dilated \
               --seg \
               --gwc \
               --datapath ./dataset/ \
               --kittidatapath ./dataset/data_scene_flow_2015/training/ \
               --epochs 80 \
               --savemodel ./trained/dilated_gwc_seg \
               --batchsize 8 \
               --numworker 4 \
               --startepoch 0 \


