_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

load_from='/data/FM/code/YOLO-World_RSSD/pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-7a5eea3b.pth'
work_dir = 'work_dirs/MI-OAD/Pretrain'

# hyper-parameters
num_classes = 75
num_training_classes = 75
max_epochs = 40
close_mosaic_epochs = 10
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 4

persistent_workers = False


albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldDualPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              text_enhancder=dict(type='ImagePoolingAttentionModule',
                                  embed_channels=256,
                                  num_heads=8)),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(1, 40),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value='#'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts', 'MIAOD_caption'))
]
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='MultiModalMosaic',
    #      img_scale=_base_.img_scale,
    #      pad_val=114.0,
    #      pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.0,
        scaling_ratio_range=(1.0, 1.0),
        max_shear_degree=0.0,
        use_mask_refine=False,
    ),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    *text_transform,
]
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]

###########  Pretrain
MIAOD_Pretrain_Detection_datasets = dict(
    # _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5MIAOD_datasets',
        data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
        ann_file='Detection/Pretrain_All.json',
        data_prefix=dict(img='images/'),
        classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_Classes.json',
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_Classes.json',
    pipeline=train_pipeline)

MIAOD_Pretrain_Caption_datasets = dict(
    # _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5MIAOD_datasets',
        data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
        ann_file='Caption/Pretrain_All.json',
        data_prefix=dict(img='images/'),
        classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_Classes.json',
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_Classes.json',
    pipeline=train_pipeline)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        num_workers=8,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(_delete_=True,
                                     type='ConcatDataset',
                                     datasets=[
                                         MIAOD_Pretrain_Detection_datasets,
                                         MIAOD_Pretrain_Caption_datasets
                                     ]))

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts','MIAOD_caption'))
]

### ZSD
MIAOD_Detection_Val_ZSD_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5MIAOD_datasets',
        data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
        ann_file='Detection/Val_ZSD_All.json',
        data_prefix=dict(img='images/'),
        classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Novel_Classes.json',
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Novel_Classes.json',
    pipeline=test_pipeline)
MIAOD_Caption_Val_ZSD_Phrase_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5MIAOD_datasets',
        data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
        ann_file='Caption/Val_ZSD_Phrase_ALL.json',
        data_prefix=dict(img='images/'),
        classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Novel_Classes.json',
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Novel_Classes.json',
    pipeline=test_pipeline)
MIAOD_Caption_Val_ZSD_Sentence_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5MIAOD_datasets',
        data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
        ann_file='Caption/Val_ZSD_Sentence_ALL.json',
        data_prefix=dict(img='images/'),
        classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Novel_Classes.json',
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Novel_Classes.json',
    pipeline=test_pipeline)


MIAOD_Detection_Val_ZSD_evaluator = dict(
    type='mmdet.CocoMetric',
    iou_thrs = [0.5],
    proposal_nums=(
        1,
        10,
        100
    ),
    ann_file='/data/FM/datasets/MI-OAD-v1.1-2M/Detection/Val_ZSD_All.json',
    metric='bbox',
    classwise=True)
MIAOD_Caption_Val_ZSD_Phrase_evaluator = dict(
    type='mmdet.CocoMetric',
    iou_thrs = [0.5],
    proposal_nums=(
        1,
        10,
        100
    ),
    ann_file='/data/FM/datasets/MI-OAD-v1.1-2M/Caption/Val_ZSD_Phrase_ALL.json',
    metric='bbox',
    classwise=True)
MIAOD_Caption_Val_ZSD_Sentence_evaluator = dict(
    type='mmdet.CocoMetric',
    iou_thrs = [0.5],
    proposal_nums=(
        1,
        10,
        100
    ),
    ann_file='/data/FM/datasets/MI-OAD-v1.1-2M/Caption/Val_ZSD_Sentence_ALL.json',
    metric='bbox',
    classwise=True)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
                _delete_=True,
                type='mmdet.ConcatDataset', 
                datasets=[
                        MIAOD_Detection_Val_ZSD_dataset,
                        MIAOD_Caption_Val_ZSD_Phrase_dataset,
                        MIAOD_Caption_Val_ZSD_Sentence_dataset
                    ],
                ignore_keys=['classes', 'palette']
            ))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='mmdet.MultiDatasetsEvaluator',
    metrics=[
                MIAOD_Detection_Val_ZSD_evaluator,
                MIAOD_Caption_Val_ZSD_Phrase_evaluator,
                MIAOD_Caption_Val_ZSD_Sentence_evaluator
            ],
    dataset_prefixes=[
                        'MIAOD_Detection_Val_ZSD_evaluator', 
                        'MIAOD_Caption_Val_ZSD_Phrase_evaluator',
                        'MIAOD_Caption_Val_ZSD_Sentence_evaluator'
                    ])
test_evaluator=val_evaluator

# training settings
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=-1,
        save_best=None,
        interval=save_epoch_intervals))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=20,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                     'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')
find_unused_parameters=True