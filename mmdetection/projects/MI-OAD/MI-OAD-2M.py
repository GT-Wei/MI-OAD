backend_args = None
num_training_classes = 75
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
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomLoadText',
         num_neg_samples=(1, 40),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value='#'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'MIAOD_caption', 'original_idx'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities', 'MIAOD_caption'))
]

###########  Pretrain
MIAOD_Pretrain_Detection_datasets = dict(
    type='MIAOD_datasets',
    data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
    ann_file='Detection/Pretrain_All.json',
    data_prefix=dict(img='images/'),
    classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_Classes.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
MIAOD_Pretrain_Caption_datasets = dict(
    type='MIAOD_datasets',
    data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
    ann_file='Caption/Pretrain_All.json',
    data_prefix=dict(img='images/'),
    classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_Classes.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)


####  Fine-Tune
MIAOD_FT_Detection_datasets = dict(
    type='MIAOD_datasets',
    data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
    ann_file='Detection/Fine-Tune_All.json',
    data_prefix=dict(img='images/'),
    classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_And_Novel_Classes.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
MIAOD_FT_Caption_datasets = dict(
    type='MIAOD_datasets',
    data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
    ann_file='Caption/Fine-Tune_All.json',
    data_prefix=dict(img='images/'),
    classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_And_Novel_Classes.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)


### ZSD
MIAOD_Detection_Val_ZSD_dataset = dict(
    type='MIAOD_datasets',
    data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
    ann_file='Detection/Val_ZSD_All.json',
    data_prefix=dict(img='images/'),
    classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Novel_Classes.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)
MIAOD_Caption_Val_ZSD_Phrase_dataset = dict(
    type='MIAOD_datasets',
    data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
    ann_file='Caption/Val_ZSD_Phrase_ALL.json',
    data_prefix=dict(img='images/'),
    classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Novel_Classes.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)
MIAOD_Caption_Val_ZSD_Sentence_dataset = dict(
    type='MIAOD_datasets',
    data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
    ann_file='Caption/Val_ZSD_Sentence_ALL.json',
    data_prefix=dict(img='images/'),
    classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Novel_Classes.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)

MIAOD_Detection_Val_ZSD_evaluator = dict(
    type='CocoMetric',
    iou_thrs = [0.5],
    proposal_nums=(
        1,
        10,
        100
    ),
    ann_file='/data/FM/datasets/MI-OAD-v1.1-2M/Detection/Val_ZSD_All.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)
MIAOD_Caption_Val_ZSD_Phrase_evaluator = dict(
    type='CocoMetric',
    iou_thrs = [0.5],
    proposal_nums=(
        1,
        10,
        100
    ),
    ann_file='/data/FM/datasets/MI-OAD-v1.1-2M/Caption/Val_ZSD_Phrase_ALL.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)
MIAOD_Caption_Val_ZSD_Sentence_evaluator = dict(
    type='CocoMetric',
    iou_thrs = [0.5],
    proposal_nums=(
        1,
        10,
        100
    ),
    ann_file='/data/FM/datasets/MI-OAD-v1.1-2M/Caption/Val_ZSD_Sentence_ALL.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)


### Val
MIAOD_Detection_Val_dataset = dict(
    type='MIAOD_datasets',
    data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
    ann_file='Detection/Val_All.json',
    data_prefix=dict(img='images/'),
    classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_And_Novel_Classes.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)
MIAOD_Caption_Val_Phrase_dataset = dict(
    type='MIAOD_datasets',
    data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
    ann_file='Caption/Val_Phrase_ALL.json',
    data_prefix=dict(img='images/'),
    classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_And_Novel_Classes.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)
MIAOD_Caption_Val_Sentence_dataset = dict(
    type='MIAOD_datasets',
    data_root='/data/FM/datasets/MI-OAD-v1.1-2M',
    ann_file='Caption/Val_Sentence_ALL.json',
    data_prefix=dict(img='images/'),
    classes_json='/data/FM/datasets/MI-OAD-v1.1-2M/datasets_categories_list/ALL_Base_And_Novel_Classes.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline,
    backend_args=backend_args)


MIAOD_Detection_Val_evaluator = dict(
    type='CocoMetric',
    iou_thrs = [0.5],
    proposal_nums=(
        1,
        10,
        100
    ),
    ann_file='/data/FM/datasets/MI-OAD-v1.1-2M/Detection/Val_All.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)
MIAOD_Caption_Val_Phrase_evaluator = dict(
    type='CocoMetric',
    iou_thrs = [0.5],
    proposal_nums=(
        1,
        10,
        100
    ),
    ann_file='/data/FM/datasets/MI-OAD-v1.1-2M/Caption/Val_Phrase_ALL.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)
MIAOD_Caption_Val_Sentence_evaluator = dict(
    type='CocoMetric',
    iou_thrs = [0.5],
    proposal_nums=(
        1,
        10,
        100
    ),
    ann_file='/data/FM/datasets/MI-OAD-v1.1-2M/Caption/Val_Sentence_ALL.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            MIAOD_Pretrain_Detection_datasets,
            MIAOD_Pretrain_Caption_datasets
        ])
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
                    MIAOD_Detection_Val_ZSD_dataset,
                    MIAOD_Caption_Val_ZSD_Phrase_dataset,
                    MIAOD_Caption_Val_ZSD_Sentence_dataset
                ],
            ignore_keys=['classes', 'palette']
        ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
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
