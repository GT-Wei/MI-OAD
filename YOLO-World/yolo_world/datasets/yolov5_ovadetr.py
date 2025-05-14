# Copyright (c) Tencent Inc. All rights reserved.
from .ovadet_datasets import OVADET_Dataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmyolo.registry import DATASETS


@DATASETS.register_module()
class YOLOv5OVADETRDataset(BatchShapePolicyDataset, OVADET_Dataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with Objects365V1Dataset.
    See `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
