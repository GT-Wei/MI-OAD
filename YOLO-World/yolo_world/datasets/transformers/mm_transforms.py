# Copyright (c) Tencent Inc. All rights reserved.
import json
import random
from typing import Tuple

import numpy as np
from mmyolo.registry import TRANSFORMS


# @TRANSFORMS.register_module()
# class RandomLoadText:

#     def __init__(self,
#                  text_path: str = None,
#                  prompt_format: str = '{}',
#                  num_neg_samples: Tuple[int, int] = (80, 80),
#                  max_num_samples: int = 80,
#                  padding_to_max: bool = False,
#                  padding_value: str = '') -> None:
#         self.prompt_format = prompt_format
#         self.num_neg_samples = num_neg_samples
#         self.max_num_samples = max_num_samples
#         self.padding_to_max = padding_to_max
#         self.padding_value = padding_value
#         if text_path is not None:
#             with open(text_path, 'r') as f:
#                 self.class_texts = json.load(f)

#     def __call__(self, results: dict) -> dict:
#         assert 'texts' in results or hasattr(self, 'class_texts'), (
#             'No texts found in results.')
#         class_texts = results.get(
#             'texts',
#             getattr(self, 'class_texts', None))

#         num_classes = len(class_texts)
#         if 'gt_labels' in results:
#             gt_label_tag = 'gt_labels'
#         elif 'gt_bboxes_labels' in results:
#             gt_label_tag = 'gt_bboxes_labels'
#         else:
#             raise ValueError('No valid labels found in results.')
#         positive_labels = set(results[gt_label_tag])

#         if len(positive_labels) > self.max_num_samples:
#             positive_labels = set(random.sample(list(positive_labels),
#                                   k=self.max_num_samples))

#         # num_neg_samples = min(
#         #     min(num_classes, self.max_num_samples) - len(positive_labels),
#         #     random.randint(*self.num_neg_samples))
#         rssd_caption = results.get('RSSD_caption', None)
#         if rssd_caption:
#             num_neg_samples = 0
#         else:
#             num_neg_samples = min(
#                 min(num_classes, self.max_num_samples) - len(positive_labels),
#             random.randint(*self.num_neg_samples))
#         candidate_neg_labels = []
#         for idx in range(num_classes):
#             # if idx not in positive_labels:
#             #     candidate_neg_labels.append(idx)
#             if idx not in positive_labels:
#                 candidate_neg_labels.append(idx)
#         negative_labels = random.sample(
#             candidate_neg_labels, k=num_neg_samples)

#         sampled_labels = list(positive_labels) + list(negative_labels)
#         random.shuffle(sampled_labels)

#         label2ids = {label: i for i, label in enumerate(sampled_labels)}

#         gt_valid_mask = np.zeros(len(results['gt_bboxes']), dtype=bool)
#         for idx, label in enumerate(results[gt_label_tag]):
#             if label in label2ids:
#                 gt_valid_mask[idx] = True
#                 results[gt_label_tag][idx] = label2ids[label]
#         results['gt_bboxes'] = results['gt_bboxes'][gt_valid_mask]
#         results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]

#         if 'instances' in results:
#             retaged_instances = []
#             for idx, inst in enumerate(results['instances']):
#                 label = inst['bbox_label']
#                 if label in label2ids:
#                     inst['bbox_label'] = label2ids[label]
#                     retaged_instances.append(inst)
#             results['instances'] = retaged_instances

#         texts = []
#         for label in sampled_labels:
#             cls_caps = class_texts[label]
#             assert len(cls_caps) > 0
#             cap_id = random.randrange(len(cls_caps))
#             sel_cls_cap = self.prompt_format.format(cls_caps[cap_id])
#             texts.append(sel_cls_cap)

#         if self.padding_to_max:
#             num_valid_labels = len(positive_labels) + len(negative_labels)
#             num_padding = self.max_num_samples - num_valid_labels
#             if num_padding > 0:
#                 texts += [self.padding_value] * num_padding

#         results['texts'] = texts

#         return results


# @TRANSFORMS.register_module()
# class LoadText:

#     def __init__(self,
#                  text_path: str = None,
#                  prompt_format: str = '{}',
#                  multi_prompt_flag: str = '/') -> None:
#         self.prompt_format = prompt_format
#         self.multi_prompt_flag = multi_prompt_flag
#         if text_path is not None:
#             with open(text_path, 'r') as f:
#                 self.class_texts = json.load(f)

#     def __call__(self, results: dict) -> dict:
#         assert 'texts' in results or hasattr(self, 'class_texts'), (
#             'No texts found in results.')
#         class_texts = results.get(
#             'texts',
#             getattr(self, 'class_texts', None))

#         texts = []
#         for idx, cls_caps in enumerate(class_texts):
#             assert len(cls_caps) > 0
#             sel_cls_cap = cls_caps[0]
#             sel_cls_cap = self.prompt_format.format(sel_cls_cap)
#             texts.append(sel_cls_cap)

#         results['texts'] = texts

#         return results



@TRANSFORMS.register_module()
class RandomLoadText:
    """Data‑augment text sampler driven by per‑sample meta‑info.

    Required keys in ``results``
    ---------------------------------
    - original_class : Sequence[str]
        所属数据集的全部类别名称（本 sample 出现 or 未出现）。
    - original_idx   : Sequence[int]
        对应每个 ``original_class`` 在全局 ``class_texts`` 中的索引。
        *必须*与 ``original_class`` 一一对应。
    - pos_class      : Sequence[str]   —— 正样本类别（图像中确实存在的目标）。
    - neg_class      : Sequence[str]   —— 该 sample 的候选负样本类别
                                     （= original_class – pos_class）。
    - gt_labels / gt_bboxes_labels    —— 现有 GT 标签（全局索引）。

    Optional keys
    ---------------------------------
    - text (Sequence[List[str]])
        若在 pipeline 前端就已放进来，可直接复用；否则从 ``text_path`` 读取。

    Notes
    -----
    - 重新映射标签时仅保留 *被采样* 到的新类别，其他目标一律过滤掉；
      逻辑与旧版保持一致。
    """

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = True,
                 padding_value: str = '#') -> None:

        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value

        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:

        # ---------- 0. 基础取值 ----------
        class_texts = results.get('texts',
                                  getattr(self, 'class_texts', None))
        if class_texts is None:
            raise RuntimeError('`class_texts` not found in results and '
                               '`text_path` is None.')

        # per‑sample 元信息
        orig_cls = results['original_class']
        orig_idx = results['original_idx']
        pos_cls = results['pos_class']
        neg_cls = results['neg_class']

        # gt 标签字段名
        if 'gt_labels' in results:
            gt_tag = 'gt_labels'
        elif 'gt_bboxes_labels' in results:
            gt_tag = 'gt_bboxes_labels'
        else:
            raise ValueError('No gt label field found.')
        
        # if len(results[gt_tag]) == 0:
        #     print('here!')
            
        # mapping : class‑name → global‑idx
        name2gid = {n: i for n, i in zip(orig_cls, orig_idx)}

        # ---------- 1. 处理正样本 ----------
        pos_gids = [name2gid[c] for c in pos_cls]
        # 若正样本超过上限，则随机裁剪
        if len(pos_gids) > self.max_num_samples:
            pos_gids = random.sample(pos_gids, self.max_num_samples)

        # ---------- 2. 处理负样本 ----------
        # 可抽取的负样本候选 (already given)
        neg_candidates = [name2gid[c] for c in neg_cls]

        # 负样本数量区间
        max_neg_by_cap = self.max_num_samples - len(pos_gids)
        wanted_neg = random.randint(*self.num_neg_samples)
        num_neg = min(len(neg_candidates), max_neg_by_cap, wanted_neg)
        
        # num_neg = len(neg_candidates)
        
        neg_gids = random.sample(neg_candidates, k=num_neg)

        # ---------- 3. 组装最终类别序列 ----------
        sampled_gids = pos_gids + neg_gids
        random.shuffle(sampled_gids)  # 打乱顺序
        gid2newid = {gid: i for i, gid in enumerate(sampled_gids)}

        # ---------- 4. 过滤 / 重新映射 GT ----------
        gt_valid = np.zeros(len(results['gt_bboxes']), dtype=bool)
        for i, g in enumerate(results[gt_tag]):
            if g in gid2newid:
                gt_valid[i] = True
                results[gt_tag][i] = gid2newid[g]

        results['gt_bboxes']   = results['gt_bboxes'][gt_valid]
        results[gt_tag]        = results[gt_tag][gt_valid]
        
        if 'instances' in results:
            new_inst = []
            for inst in results['instances']:
                lab = inst['bbox_label']
                if lab in gid2newid:
                    inst['bbox_label'] = gid2newid[lab]
                    new_inst.append(inst)
            results['instances'] = new_inst

        # ---------- 5. 生成文本描述 ----------
        MIAOD_caption_text = results.get('MIAOD_caption')  # str 或 None
        MIAOD_FLAG = MIAOD_caption_text is not None
        
        texts = []
        unique_cls_gids = list(set(results[gt_tag]))
        if MIAOD_FLAG:
            if len(unique_cls_gids) > 0:
                if len(unique_cls_gids) != 1:
                    raise ValueError('MIAOD: caption 应对应唯一类别')
                gid_only = unique_cls_gids[0]
            else:
                gid_only = -1  # 图片GT=0
                
        for idx, gid in enumerate(sampled_gids):
            if MIAOD_FLAG and idx == gid_only:
                c_t = MIAOD_caption_text.replace('.', '')
            else:
                c_t = class_texts[gid][0]
            texts.append(self.prompt_format.format(c_t))

        # 可选：padding 到固定长度
        if self.padding_to_max and len(texts) < self.max_num_samples:
            pad_num = self.max_num_samples - len(texts)
            texts.extend([self.padding_value] * pad_num)

        results['texts'] = tuple(texts)
        return results


@TRANSFORMS.register_module()
class LoadText:
    """根据类别筛选文本并写回 `results['text']`.

    Args:
        text_path (str): JSON 列表文件，每元素为该类别的 prompt。
        prompt_format (str): 输出包装格式，如 'a photo of {}'。
        multi_prompt_flag (str): 一条 prompt 出现多候选时的分隔符。
    """

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 multi_prompt_flag: str = '/') -> None:
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag

        if text_path is not None:
            with open(text_path, 'r', encoding='utf-8') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        class_texts = results.get('texts', getattr(self, 'class_texts', None))
        if class_texts is None:
            raise ValueError('`class_texts` is None – 检查输入或 text_path.')

        orig_cls = results['original_class']          # 当前数据集全部类
        texts     = ['#'] * len(class_texts)           # 先用填充字符

        # ---------------------------------------------------
        # MIAOD_caption ：仅 1 个正类 + 一个 caption 字符串
        # ---------------------------------------------------
        if results.get('MIAOD_caption', False):
            pos_cls = results['pos_class']
            if len(pos_cls) != 1:
                raise ValueError(
                    f'MIAOD 样本必须且只能有 1 个正类，实际 {pos_cls}；'
                    f'ann={results.get("ann_path", "N/A")}'
                )

            target_cls   = pos_cls[0]
            caption_text = results['MIAOD_caption']          # 单条 caption 字符串
            
            for i, cls_prompt in enumerate(class_texts):
                if cls_prompt[0] == target_cls:
                    texts[i] = caption_text.replace('.', '')
                    break 

        # ---------------------------------------------------
        # 普通分支：orig_cls 内的类全部填入各自 prompt
        # ---------------------------------------------------
        else:
            for i, cls_prompt in enumerate(class_texts):
                if cls_prompt[0] in orig_cls:
                    texts[i] = self.prompt_format.format(cls_prompt[0])

        results['texts'] = texts
        return results
