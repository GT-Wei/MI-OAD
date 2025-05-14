

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_ALL_Types.py \
#     work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive/epoch_6.pth \
#     8 \
#     --work-dir work_dirs/RSSD/Fine_tune_Val_6pth/Grounding_ALL_Types


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_ALL_Types_ZSD.py \
#     work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive/epoch_6.pth \
#     8 \
#     --work-dir work_dirs/RSSD/Fine_tune_Val_6pth/Grounding_ALL_Types_ZSD


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types.py \
    work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive/epoch_6.pth \
    8 \
    --work-dir work_dirs/RSSD_Final/Fine_tune_Val_6pth/Detection_Types

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types_ZSD.py \
#     work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive/epoch_6.pth \
#     8 \
#     --work-dir work_dirs/RSSD_Final/Fine_tune_Val_6pth/Detection_Types_ZSD

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types.py \
    work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive/epoch_6.pth \
    8 \
    --work-dir work_dirs/RSSD_Final/Fine_tune_Val_6pth/Grounding_Phrase_Types 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types_ZSD.py \
#     work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive/epoch_6.pth \
#     8 \
#     --work-dir work_dirs/RSSD_Final/Fine_tune_Val_6pth/Grounding_Phrase_Types_ZSD 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Sentence_Types.py \
    work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive/epoch_6.pth \
    8 \
    --work-dir work_dirs/RSSD_Final/Fine_tune_Val_6pth/Grounding_Sentence_Types

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Sentence_Types_ZSD.py \
#     work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive/epoch_6.pth \
#     8 \
#     --work-dir work_dirs/RSSD_Final/Fine_tune_Val_6pth/Grounding_Sentence_Types_ZSD
