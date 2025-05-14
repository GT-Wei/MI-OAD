# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  ./tools/dist_test.sh \
#     configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_detection.py \
#     work_dirs/yolo_world_l_RSSD_4x8x2_Pretrain_have_negtive_have_load_from_40_epochs/epoch_40.pth \
#     6 \
#     --work-dir work_dirs/RSSD_Final/Pretrain_Val_yolo_world/Detection_Types

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  ./tools/dist_test.sh \
    configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_detection_ZSD.py \
    pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-7a5eea3b.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val_yolo_world/Detection_Types_ZSD

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
#     configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_phrase.py \
#     pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-7a5eea3b.pth \
#     6 \
#     --work-dir work_dirs/RSSD_Final/Pretrain_Val_yolo_world/Grounding_Phrase_Types 

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
    configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_phrase_ZSD.py \
    pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-7a5eea3b.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val_yolo_world/Grounding_Phrase_Types_ZSD 

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
#     configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_sentence.py \
#     pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-7a5eea3b.pth \
#     6 \
#     --work-dir work_dirs/RSSD_Final/Pretrain_Val_yolo_world/Grounding_Sentence_Types

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ./tools/dist_test.sh \
    configs/RSSD/val/yolo_world_l_RSSD_4x8x2_val_sentence_ZSD.py \
    pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-7a5eea3b.pth \
    6 \
    --work-dir work_dirs/RSSD_Final/Pretrain_Val_yolo_world/Grounding_Sentence_Types_ZSD
