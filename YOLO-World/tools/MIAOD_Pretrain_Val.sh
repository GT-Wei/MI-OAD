CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
    configs/MI_AOD/yolo_world_l_MIAOD_4x8x4_Pretrain.py \
    work_dirs/MI-AOD/Pretrain/epoch_12.pth \
    8 \
    --work-dir work_dirs/MI-AOD/Pretrain_Val/12pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
    configs/MI_AOD/yolo_world_l_MIAOD_4x8x4_Pretrain.py \
    work_dirs/MI-AOD/Pretrain/epoch_16.pth \
    8 \
    --work-dir work_dirs/MI-AOD/Pretrain_Val/16pth

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
#     configs/MI_AOD/yolo_world_l_MIAOD_4x8x4_Pretrain.py \
#     work_dirs/MI-AOD/Pretrain/epoch_10.pth \
#     8 \
#     --work-dir work_dirs/MI-AOD/Pretrain_Val/10pth