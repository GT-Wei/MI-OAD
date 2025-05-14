

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_ALL_Types.py \
#     work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD_sample_negtive/epoch_12.pth \
#     8 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_ALL_Types


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_ALL_Types_ZSD.py \
#     work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD_sample_negtive/epoch_12.pth \
#     8 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_ALL_Types_ZSD


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
    projects/Attributes-OAD-GroundingDINO/Pretrain_1x8x4_1x_MI-AOD.py \
    work_dirs/MI-AODv1.1/Pretrain/epoch_5.pth \
    8 \
    --work-dir work_dirs/Attributes-OAD-GroundingDINO/Pretrain_Val/Detection_Types

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types.py \
    work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD_sample_negtive/epoch_12.pth \
    8 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val/Detection_Types

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types_ZSD.py \
    work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD_sample_negtive/epoch_12.pth \
    8 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val/Detection_Types_ZSD

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types.py \
    work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD_sample_negtive/epoch_12.pth \
    8 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val/Grounding_Phrase_Types 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types_ZSD.py \
    work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD_sample_negtive/epoch_12.pth \
    8 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val/Grounding_Phrase_Types_ZSD 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Sentence_Types.py \
    work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD_sample_negtive/epoch_12.pth \
    8 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val/Grounding_Sentence_Types

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Sentence_Types_ZSD.py \
    work_dirs/RSSD_Final/Swin-T_Pretrain_8x8x1_1x_RSSD_sample_negtive/epoch_12.pth \
    8 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val/Grounding_Sentence_Types_ZSD
