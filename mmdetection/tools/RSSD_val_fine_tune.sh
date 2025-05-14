

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_ALL_Types.py \
#     work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive_Real_GZSD/epoch_12.pth \
#     6 \
#     --work-dir work_dirs/RSSD/Fine_tune_Val_12pth/Grounding_ALL_Types


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_ALL_Types_ZSD.py \
#     work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive_Real_GZSD/epoch_12.pth \
#     6 \
#     --work-dir work_dirs/RSSD/Fine_tune_Val_12pth/Grounding_ALL_Types_ZSD


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types.py \
    work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive_Real_GZSD/epoch_12.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Visual/Detection_Types \
    --show-dir work_dirs/RSSD_Final/Visual/Detection_Types
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types_ZSD.py \
#     work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive_Real_GZSD/epoch_12.pth \
#     6 \
#     --work-dir work_dirs/RSSD_Final/Fine_tune_Val_Real_12pth/Detection_Types_ZSD

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types.py \
    work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive_Real_GZSD/epoch_12.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Visual/Grounding_Phrase_Types \
    --show-dir work_dirs/RSSD_Final/Visual/Grounding_Phrase_Types

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types_ZSD.py \
#     work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive_Real_GZSD/epoch_12.pth \
#     6 \
#     --work-dir work_dirs/RSSD_Final/Fine_tune_Val_Real_12pth/Grounding_Phrase_Types_ZSD 

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Sentence_Types.py \
    work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive_Real_GZSD/epoch_12.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Visual/Grounding_Sentence_Types \
    --show-dir work_dirs/RSSD_Final/Visual/Grounding_Sentence_Types
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Sentence_Types_ZSD.py \
#     work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive_Real_GZSD/epoch_12.pth \
#     6 \
#     --work-dir work_dirs/RSSD_Final/Fine_tune_Val_Real_12pth/Grounding_Sentence_Types_ZSD
