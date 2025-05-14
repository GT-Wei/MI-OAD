# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
# import spacy
from sentence_transformers import SentenceTransformer

@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes

        super().__init__(*args, **kwargs)
        
        # self.sentence_bert = SentenceTransformer("/data/FM/weiguoting/ICCV/code/YOLO-World_RSSD/pretrained_models/sentence-transformers/stsb-roberta-large")
        self.sentect_text_linear_merger = nn.Linear(1024, 512)
        
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            # texts = ['']
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        
        # if self.training:
        #     rssd_captions = batch_data_samples['RSSD_caption']  # 长度与 batch_size 一致
            
        #     for batch_id, rssd_caption in enumerate(rssd_captions):
        #         if rssd_caption:
        #             bboxes_info_all = batch_data_samples['bboxes_labels']  # [num_boxes, 6]
        #             mask = (bboxes_info_all[:, 0] == batch_id)
        #             bboxes_info = bboxes_info_all[mask]  # 取到只属于这个 batch 的 box 信息
                    
        #             if bboxes_info.shape[0] == 0:
        #                 # 说明该 batch_id 没有标注框，可能是空的图像或其他原因
        #                 # 如果这种情况合法，可以继续；如果不合法，可以 raise Exception
        #                 continue

        #             gt_labels = bboxes_info[:, 1]
        #             unique_labels = torch.unique(gt_labels)
        #             if len(unique_labels) != 1:
        #                 raise ValueError(
        #                     f"Batch {batch_id} 的 gt_labels 不一致: {unique_labels.tolist()}"
        #                 )
                    
        #             gt_label_index = int(unique_labels.item())
        #             texts[batch_id][gt_label_index] = rssd_caption
        #             # del rssd_caption, bboxes_info_all, mask, bboxes_info, gt_labels, unique_labels, gt_label_index
        # else:
        #     for batch_id, data_samples in enumerate(batch_data_samples):
        #         gt_label_tmp = data_samples.gt_instances.labels 
        #         tmp_text_prompts = list(texts[batch_id])
                
        #         gt_unique_labels = torch.unique(gt_label_tmp)
        #         gt_unique_labels = [label.item() for label in gt_unique_labels]  # 将张量转换为Python整数列表
                
        #         mask = [False] * len(tmp_text_prompts)

        #         for label in gt_unique_labels:
        #             if label < len(mask):
        #                 mask[label] = True
        #             else:
        #                 raise ValueError("Label index out")
                
        #         filtered_prompts = []
        #         for i in range(len(mask)):
        #             if mask[i]:
        #                 filtered_prompts.append(tmp_text_prompts[i])
        #             else:
        #                 filtered_prompts.append('#')
                
        #         rssd_caption = data_samples.get('RSSD_caption', None)
        #         if rssd_caption:
        #             if len(gt_unique_labels) == 0:
        #                 continue # 无对应label
        #             if len(gt_unique_labels) != 1:
        #                 raise ValueError(
        #                     f"rssd_gt_label_tmp 不一致: {unique_labels.tolist()}"
        #                 )
        #             gt_label_index = int(gt_unique_labels[0])
        #             filtered_prompts[gt_label_index] = rssd_caption
                
        #         texts[batch_id] = filtered_prompts
                
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        
        # rssd_captions = batch_data_samples['RSSD_caption']  # 长度与 batch_size 一致
        
        # for batch_id, rssd_caption in enumerate(rssd_captions):
        #     if rssd_caption:
        #         bboxes_info_all = batch_data_samples['bboxes_labels']  # [num_boxes, 6]
        #         mask = (bboxes_info_all[:, 0] == batch_id)
        #         bboxes_info = bboxes_info_all[mask]  # 取到只属于这个 batch 的 box 信息
                
        #         if bboxes_info.shape[0] == 0:
        #             # 说明该 batch_id 没有标注框，可能是空的图像或其他原因
        #             # 如果这种情况合法，可以继续；如果不合法，可以 raise Exception
        #             continue

        #         gt_labels = bboxes_info[:, 1]
        #         unique_labels = torch.unique(gt_labels)
        #         if len(unique_labels) != 1:
        #             raise ValueError(
        #                 f"Batch {batch_id} 的 gt_labels 不一致: {unique_labels.tolist()}"
        #             )
                
        #         gt_label_index = int(unique_labels.item())
        #         texts[gt_label_index] = rssd_caption
            # # 1) 从 bboxes_labels 中筛选出属于当前 batch 的 bounding boxes
            # #    bboxes_info 形状同样是 [M, 6]，其中 M 是该 batch 的 bbox 数
            # bboxes_info_all = batch_data_samples['bboxes_labels']  # [num_boxes, 6]
            # mask = (bboxes_info_all[:, 0] == batch_id)
            # bboxes_info = bboxes_info_all[mask]  # 取到只属于这个 batch 的 box 信息
            
            # if bboxes_info.shape[0] == 0:
            #     # 说明该 batch_id 没有标注框，可能是空的图像或其他原因
            #     # 如果这种情况合法，可以继续；如果不合法，可以 raise Exception
            #     continue
            
            # # 取第 1 列作为 gt_labels
            # gt_labels = bboxes_info[:, 1]
            
            # # 2) 检查 gt_label 是否一致
            # unique_labels = torch.unique(gt_labels)
            # if len(unique_labels) != 1:
            #     raise ValueError(
            #         f"Batch {batch_id} 的 gt_labels 不一致: {unique_labels.tolist()}"
            #     )
            
            # # 3) 用唯一 gt_label 替换 txt_feats[batch_id, gt_label, :]
            # gt_label_index = int(unique_labels.item())  # 取出一个 int
               
            # # with torch.no_grad():
            # #     cap_emb = torch.tensor(self.sentence_bert.encode(rssd_caption), dtype=torch.float).unsqueeze(0).detach().to(txt_feats.device)
            # cap_emb = self.backbone.forward_text([[rssd_caption]])
            # # 6) 将二者在 channel 维度拼接，并用线性层映射回 [1, text_channel]
            # concat_emb = torch.cat([cap_emb[0], txt_feats[batch_id, gt_label_index, :].unsqueeze(0)], dim=1)  # [1, 2 * text_channel]
            # updated_feat = self.sentect_text_linear_merger(concat_emb)     # [1, text_channel]
            # txt_feats[batch_id, gt_label_index, :] = updated_feat
        
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
