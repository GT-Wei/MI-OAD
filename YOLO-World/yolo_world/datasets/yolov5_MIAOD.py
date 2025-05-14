# Copyright (c) Tencent Inc. All rights reserved.
from .MIAOD_datasets import MIAOD_datasets

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmyolo.registry import DATASETS


@DATASETS.register_module()
class YOLOv5MIAOD_datasets(BatchShapePolicyDataset, MIAOD_datasets):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with Objects365V1Dataset.
    See `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
