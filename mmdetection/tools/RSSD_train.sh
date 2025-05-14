#!/bin/bash

# export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=$PET_NPROC_PER_NODE
export NNODES=$PET_NNODES
export MASTER_PORT=$MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CONFIG=configs/mm_grounding_dino/RSSD/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD.py

torchrun --nproc_per_node=$GPUS_PER_NODE --nnode=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    tools/train.py \
    ${CONFIG} \
    --launcher pytorch \
    --amp