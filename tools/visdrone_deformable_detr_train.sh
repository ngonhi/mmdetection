#!/usr/bin/env bash

CONFIG=/home/ubuntu/visdrone/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_visdrone.py
GPUS=1
PORT=${PORT:-29501}
GPU_IDS=0
CHECKPOINT=/home/ubuntu/nhi_workspace/mmdetection/work_dirs/detectors_cascade_rcnn_r50_1x_visdrone.py/latest.pth
PYTHON=${PYTHON:-"python"}

# $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     tools/train.py $CONFIG --launcher pytorch ${@:3} --resume-from $CHECKPOINT
$PYTHON tools/train.py $CONFIG --gpu-ids $GPU_IDS #--resume-from $CHECKPOINT