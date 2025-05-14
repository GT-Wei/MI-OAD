
# CUDA_VISIBLE_DEVICES=1,2 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types.py \
#     work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD/epoch_5.pth \
#     2 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_Phrase_Types \
#     --show-dir work_dirs/RSSD/Pretrain_Val/Grounding_Phrase_Types


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_ALL_Types.py \
#     work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD/epoch_5.pth \
#     8 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_ALL_Types


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_ALL_Types_ZSD.py \
#     work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD/epoch_5.pth \
#     8 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_ALL_Types_ZSD


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types.py \
#     work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD/epoch_5.pth \
#     8 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Detection_Types

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types_ZSD.py \
    work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD_sample_negtive/epoch_5.pth \
    8 \
    --work-dir work_dirs/RSSD/Pretrain_Val/Detection_Types_ZSD

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types.py \
    work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD/epoch_5.pth \
    8 \
    --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_Phrase_Types 

CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types.py \
    work_dirs/RSSD_Final/Swin-T_Pretrain_1x8x4_1x_RSSD_A100/epoch_1.pth \
    1 \
    --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_Phrase_Types 
    --show-dir work_dirs/RSSD/Pretrain_Val/Grounding_Phrase_Types
    
CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types.py \
    work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD/epoch_1.pth \
    1 \
    --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_Phrase_Types 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types_ZSD.py \
#     work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD/epoch_5.pth \
#     8 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_Phrase_Types_ZSD 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Sentence_Types.py \
#     work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD/epoch_5.pth \
#     8 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_Sentence_Types

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Sentence_Types_ZSD.py \
#     work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD/epoch_5.pth \
#     8 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_Sentence_Types_ZSD
