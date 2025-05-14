#!/bin/bash

# export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=$PET_NPROC_PER_NODE
export NNODES=$PET_NNODES
export MASTER_PORT=$MASTER_PORT

export CONFIG=projects/MI-OAD/Pretrain_4x8x4_1x_MI-OAD.py

torchrun --nproc_per_node=$GPUS_PER_NODE --nnode=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    tools/train.py \
    ${CONFIG} \
    --launcher pytorch \
    --amp 

