CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
    configs/MI_AOD/yolo_world_l_MIAOD_4x8x4_Pretrain.py \
    work_dirs/MI-AOD/Pretrain/epoch_20.pth \
    8 \
    --work-dir work_dirs/test_debug

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  ./tools/dist_test.sh \
    configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_detection.py \
    work_dirs/yolo_world_l_RSSD_4x8x2_SFT_have_negtive_have_load_from_20_epochs/epoch_20.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/SFT_Val/Detection_Types

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  ./tools/dist_test.sh \
#     configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_detection_ZSD.py \
#     work_dirs/yolo_world_l_RSSD_4x8x2_Pretrain_have_negtive_have_load_from_40_epochs/epoch_20.pth \
#     6 \
#     --work-dir work_dirs/RSSD_Final/Pretrain_Val/Detection_Types_ZSD

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
    configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_phrase.py \
    work_dirs/yolo_world_l_RSSD_4x8x2_SFT_have_negtive_have_load_from_20_epochs/epoch_20.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/SFT_Val/Grounding_Phrase_Types 

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
#     configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_phrase_ZSD.py \
#     work_dirs/yolo_world_l_RSSD_4x8x2_Pretrain_have_negtive_have_load_from_40_epochs/epoch_20.pth \
#     6 \
#     --work-dir work_dirs/RSSD_Final/Pretrain_Val/Grounding_Phrase_Types_ZSD 

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
    configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_sentence.py \
    work_dirs/yolo_world_l_RSSD_4x8x2_SFT_have_negtive_have_load_from_20_epochs/epoch_20.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/SFT_Val/Grounding_Sentence_Types

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
#     configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_sentence_ZSD.py \
#     work_dirs/yolo_world_l_RSSD_4x8x2_Pretrain_have_negtive_have_load_from_40_epochs/epoch_20.pth \
#     6 \
#     --work-dir work_dirs/RSSD_Final/Pretrain_Val/Grounding_Sentence_Types_ZSD
