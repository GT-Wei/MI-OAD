python demo/image_demo.py /data/FM/weiguoting/ICML/datasets/RSSDD_label/Visdrone/val/images/Visdrone_0000001_02999_d_0000005__640__0.jpg \
        configs/mm_grounding_dino/RSSD/val/grounding_dino_swin-t_Detection_Types.py \
        --weights work_dirs/RSSD_Final/grounding_dino_swin-t_finetune_8x8x1_1x_RSSD_Train_SFT_sample_negtive_Real_GZSD/epoch_12.pth \
        --texts 'car. green car' -c