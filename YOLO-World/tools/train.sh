#!/bin/bash
#export NCCL_SOCKET_IFNAME=eth0
#export NCCL_IB_DISABLE=0
#export NCCL_IB_CUDA_SUPPORT=1
#export NCCL_IB_GID_INDEX=0
#export NCCL_DEBUG=INFO
#export NCCL_IB_TIMEOUT=23
#export NCCL_IB_RETRY_CNT=7
#export NCCL_IB_HCA=mlx5_2,mlx5_5


#wandb login
#apt-get install -y libibverbs1

export GPUS_PER_NODE=$PET_NPROC_PER_NODE
export NNODES=$PET_NNODES
export MASTER_PORT=$MASTER_PORT


export CONFIG=configs/MI_AOD/yolo_world_l_MIAOD_4x8x4_Pretrain.py


# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nproc_per_node=$GPUS_PER_NODE \
#     --nnode=$NNODES \
#     --node_rank=$RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     $(dirname "$0")/train.py \
#     ${CONFIG} \
#     --launcher pytorch ${@:3}

torchrun --nproc_per_node=$GPUS_PER_NODE --nnode=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    tools/train.py \
    ${CONFIG} \
    --launcher pytorch \
    --amp \
    --resume /data/FM/weiguoting/ICCV/code/YOLO-World_RSSD/work_dirs/yolo_world_l_RSSD_4x8x2_Pretrain_have_negtive_have_load_from_40_epochs/epoch_20.pth
