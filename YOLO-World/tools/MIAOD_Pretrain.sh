export GPUS_PER_NODE=$PET_NPROC_PER_NODE
export NNODES=$PET_NNODES
export MASTER_PORT=$MASTER_PORT


export CONFIG=configs/MI_AOD/yolo_world_l_MIAOD_4x8x4_Pretrain.py

torchrun --nproc_per_node=$GPUS_PER_NODE --nnode=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    tools/train.py \
    ${CONFIG} \
    --launcher pytorch \
    --amp