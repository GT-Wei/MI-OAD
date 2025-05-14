

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_ALL_Types.py \
#     pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
#     6 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_ALL_Types


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
#     configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_ALL_Types_ZSD.py \
#     pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
#     6 \
#     --work-dir work_dirs/RSSD/Pretrain_Val/Grounding_ALL_Types_ZSD


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types.py \
    pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val_GroundingDINO/Detection_Types

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types_ZSD.py \
    pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val_GroundingDINO/Detection_Types_ZSD

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types.py \
    pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val_GroundingDINO/Grounding_Phrase_Types 

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Phrase_Types_ZSD.py \
    pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val_GroundingDINO/Grounding_Phrase_Types_ZSD 

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Sentence_Types.py \
    pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val_GroundingDINO/Grounding_Sentence_Types

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
    configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Grounding_Sentence_Types_ZSD.py \
    pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val_GroundingDINO/Grounding_Sentence_Types_ZSD
