#!/usr/bin/env bash

CONFIG=/home/ubuntu/visdrone/mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_visdrone.py
GPUS=1
PORT=${PORT:-29504}
GPU_IDS=0
CHECKPOINT=/home/ubuntu/nhi_workspace/mmdetection/work_dirs/cascade_rcnn_hrnetv2_w40_20e_visdrone.py/latest.pth

PYTHON=${PYTHON:-"python"}
# $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     tools/train.py $CONFIG --launcher pytorch ${@:3} --resume-from $CHECKPOINT
$PYTHON tools/train.py $CONFIG --gpu-ids $GPU_IDS #--resume-from $CHECKPOINT