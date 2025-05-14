backend_args = None

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

###########
DIOR_train_dataset = dict(
    type='OVADET_Dataset',
    data_root='/home',
    ann_file='disk1/code/mmdetection/data/COCO_JSON/DIOR_Train.json',
    data_prefix=dict(img='disk/ICML/datasets/ovadetr_datasets/DIOR_filter/train/images/'),
    classes_json='data/RS_texts/DIOR_Train.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)

DOTA_train_dataset = dict(
    type='OVADET_Dataset',
    data_root='/home',
    ann_file='disk1/code/mmdetection/data/COCO_JSON/DOTA_Train_v1.5.json',
    data_prefix=dict(img='disk/ICML/datasets/ovadetr_datasets/DOTA_Split_filter/train/images/'),
    classes_json='data/RS_texts/DOTA_Train_v1.5.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)

xView_train_dataset = dict(
    type='OVADET_Dataset',
    data_root='/home',
    ann_file='disk1/code/mmdetection/data/COCO_JSON/xView_Train.json',
    data_prefix=dict(img='disk/ICML/datasets/ovadetr_datasets/xView_Split_filter/train/images/'),
    classes_json='data/RS_texts/xView_Train.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)


DIOR_GZSD_val_dataset = dict(
    type='OVADET_Dataset',
    data_root='/home/disk/ICML',
    ann_file='ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/DIOR_GZSD.json',
    data_prefix=dict(img='datasets/ovadetr_datasets/DIOR_filter/val/images/'),
    classes_json='data/RS_texts/DIOR_GZSD.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)

DIOR_ZSD_val_dataset = dict(
    type='OVADET_Dataset',
    data_root='/home/disk/ICML',
    ann_file='ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/DIOR_ZSD.json',
    data_prefix=dict(img='datasets/ovadetr_datasets/DIOR_filter/val/images/'),
    classes_json='data/RS_texts/DIOR_ZSD.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)

DOTA_GZSD_val_dataset = dict(
    type='OVADET_Dataset',
    data_root='/home/disk/ICML',
    ann_file='ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/DOTA_GZSD_v1.5.json',
    data_prefix=dict(img='datasets/ovadetr_datasets/DOTA_Split_filter/val/images/'),
    classes_json='data/RS_texts/DOTA_GZSD_v1.5.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)

DOTA_ZSD_val_dataset = dict(
    type='OVADET_Dataset',
    data_root='/home/disk/ICML',
    ann_file='ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/DOTA_ZSD_v1.5.json',
    data_prefix=dict(img='datasets/ovadetr_datasets/DOTA_Split_filter/val/images/'),
    classes_json='data/RS_texts/DOTA_ZSD_v1.5.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)

xView_GZSD_val_dataset = dict(
    type='OVADET_Dataset',
    data_root='/home/disk/ICML',
    ann_file='ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/xView_GZSD.json',
    data_prefix=dict(img='datasets/ovadetr_datasets/xView_Split_filter/val/images/'),
    classes_json='data/RS_texts/xView_GZSD.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)

xView_ZSD_val_dataset = dict(
    type='OVADET_Dataset',
    data_root='/home/disk/ICML',
    ann_file='ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/xView_ZSD.json',
    data_prefix=dict(img='datasets/ovadetr_datasets/xView_Split_filter/val/images/'),
    classes_json='data/RS_texts/xView_ZSD.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)

###########
DIOR_GZSD_evaluator = dict(
    type='CocoMetric',
    proposal_nums=(
        100,
        300,
        1000,
    ),
    ann_file='/home/disk/ICML/ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/DIOR_GZSD.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)
DIOR_ZSD_evaluator = dict(
    type='CocoMetric',
    proposal_nums=(
        100,
        300,
        1000,
    ),
    ann_file='/home/disk/ICML/ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/DIOR_ZSD.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)

DOTA_GZSD_evaluator = dict(
    type='CocoMetric',
    proposal_nums=(
        100,
        300,
        1000,
    ),
    ann_file='/home/disk/ICML/ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/DOTA_GZSD_v1.5.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)
DOTA_ZSD_evaluator = dict(
    type='CocoMetric',
    proposal_nums=(
        100,
        300,
        1000,
    ),
    ann_file='/home/disk/ICML/ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/DOTA_ZSD_v1.5.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)

xView_GZSD_evaluator = dict(
    type='CocoMetric',
    proposal_nums=(
        100,
        300,
        1000,
    ),
    ann_file='/home/disk/ICML/ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/xView_GZSD.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)
xView_ZSD_evaluator = dict(
    type='CocoMetric',
    proposal_nums=(
        100,
        300,
        1000,
    ),
    ann_file='/home/disk/ICML/ICCV/Final/OVA-DETR-pytorch-base/class_text_dict/ovadetr_dota_dior_xview_detection_v1_category_each_v1.5/xView_ZSD.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)
###########


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        # _delete_=True,
        type='ConcatDataset',
        datasets=[DIOR_train_dataset,
                DOTA_train_dataset,
                xView_train_dataset
                ],
        ignore_keys=['classes', 'palette'])
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
            # _delete_=True,
            type='ConcatDataset', 
            datasets=[
                    DIOR_GZSD_val_dataset,
                    DIOR_ZSD_val_dataset,
                    # DOTA_GZSD_val_dataset,
                    # DOTA_ZSD_val_dataset,
                    # xView_GZSD_val_dataset,
                    # xView_ZSD_val_dataset,
                ],
            ignore_keys=['classes', 'palette']
        ))
test_dataloader = val_dataloader

val_evaluator = dict(
    # _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=[
                DIOR_GZSD_evaluator,
                DIOR_ZSD_evaluator,
                # DOTA_GZSD_evaluator,
                # DOTA_ZSD_evaluator, 
                # xView_GZSD_evaluator,
                # xView_ZSD_evaluator
            ],
    dataset_prefixes=[
                        'DIOR_GZSD', 
                        'DIOR_ZSD',
                        # 'DOTA_GZSD',
                        # 'DOTA_ZSD',
                        # 'xView_GZSD',
                        # 'xView_ZSD'
                    ])
test_evaluator=val_evaluator
