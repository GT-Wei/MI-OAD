# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import Tuple
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes

try:
    from transformers import AutoTokenizer
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

import random
import re

import numpy as np


def clean_name(name):
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    name = name.lower()
    return name


def check_for_positive_overflow(gt_bboxes, gt_labels, text, tokenizer,
                                max_tokens):
    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_list = np.unique(gt_labels).tolist()
    # random shuffule so we can sample different annotations
    # at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(text[str(label)]) + '. '

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_tokens:
            break
        else:
            kept_lables.append(label)

    keep_box_index = []
    keep_gt_labels = []
    for i in range(len(gt_labels)):
        if gt_labels[i] in kept_lables:
            keep_box_index.append(i)
            keep_gt_labels.append(gt_labels[i])

    return gt_bboxes[keep_box_index], np.array(
        keep_gt_labels, dtype=np.long), length


def generate_senetence_given_labels(positive_label_list, negative_label_list,
                                    text):
    label_to_positions = {}

    label_list = negative_label_list + positive_label_list

    random.shuffle(label_list)

    pheso_caption = ''

    label_remap_dict = {}
    for index, label in enumerate(label_list):

        start_index = len(pheso_caption)

        pheso_caption += clean_name(text[str(label)])

        end_index = len(pheso_caption)

        if label in positive_label_list:
            label_to_positions[index] = [[start_index, end_index]]
            label_remap_dict[int(label)] = index

        # if index != len(label_list) - 1:
        #     pheso_caption += '. '
        pheso_caption += '. '

    return label_to_positions, pheso_caption, label_remap_dict


@TRANSFORMS.register_module()
class RandomSamplingNegPos(BaseTransform):

    def __init__(self,
                 tokenizer_name,
                 num_sample_negative=85,
                 max_tokens=256,
                 full_sampling_prob=0.5,
                 label_map_file=None):
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_sample_negative = num_sample_negative
        self.full_sampling_prob = full_sampling_prob
        self.max_tokens = max_tokens
        self.label_map = None
        if label_map_file:
            with open(label_map_file, 'r') as file:
                self.label_map = json.load(file)

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            return self.vg_aug(results)
        else:
            return self.od_aug(results)

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text'].lower().strip()
        if not text.endswith('.'):
            text = text + '. '

        phrases = results['phrases']
        # TODO: add neg
        positive_label_list = np.unique(gt_labels).tolist()
        label_to_positions = {}
        for label in positive_label_list:
            label_to_positions[label] = phrases[label]['tokens_positive']

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = text
        results['tokens_positive'] = label_to_positions
        return results

    def od_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']

        if 'text' not in results:
            assert self.label_map is not None
            text = self.label_map
        else:
            text = results['text']

        original_box_num = len(gt_labels)
        # If the category name is in the format of 'a/b' (in object365),
        # we randomly select one of them.
        for key, value in text.items():
            if '/' in value:
                text[key] = random.choice(value.split('/')).strip()

        gt_bboxes, gt_labels, positive_caption_length = \
            check_for_positive_overflow(gt_bboxes, gt_labels,
                                        text, self.tokenizer, self.max_tokens)

        if len(gt_bboxes) < original_box_num:
            print('WARNING: removed {} boxes due to positive caption overflow'.
                  format(original_box_num - len(gt_bboxes)))

        valid_negative_indexes = list(text.keys())

        positive_label_list = np.unique(gt_labels).tolist()
        full_negative = self.num_sample_negative

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            # c. probability_full: add both all positive and all negatives
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            for i in np.random.choice(
                    valid_negative_indexes, size=num_negatives, replace=False):
                if i not in positive_label_list:
                    negative_label_list.add(i)

        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)
        random.shuffle(negative_label_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_label_list = []

        for negative_label in negative_label_list:
            label_text = clean_name(text[str(negative_label)]) + '. '

            tokenized = self.tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_label_list.append(negative_label)
            else:
                break
        negative_label_list = screened_negative_label_list
        label_to_positions, pheso_caption, label_remap_dict = \
            generate_senetence_given_labels(positive_label_list,
                                            negative_label_list, text)

        # label remap
        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions

        return results


@TRANSFORMS.register_module()
class LoadTextAnnotations(BaseTransform):

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            tokens_positive = [
                phrase['tokens_positive']
                for phrase in results['phrases'].values()
            ]
            results['tokens_positive'] = tokens_positive
        else:
            text = results['text']
            results['text'] = list(text.values())
        return results

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
        class_texts = results.get('text',
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
                c_t = class_texts[gid]
            texts.append(self.prompt_format.format(c_t))

        # 可选：padding 到固定长度
        if self.padding_to_max and len(texts) < self.max_num_samples:
            pad_num = self.max_num_samples - len(texts)
            texts.extend([self.padding_value] * pad_num)

        results['text'] = tuple(texts)
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
        class_texts = results.get('text', getattr(self, 'class_texts', None))
        if class_texts is None:
            raise ValueError('`class_texts` is None – 检查输入或 text_path.')

        orig_cls = results['original_class']          # 当前数据集全部类
        texts     = ['#'] * len(class_texts)           # 先填空

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
            caption_text = results['MIAOD_caption'].replace('.', '')        # 单条 caption 字符串
            
            for i, cls_prompt in enumerate(class_texts):
                if cls_prompt == target_cls:
                    texts[i] = caption_text
                    break 

        # ---------------------------------------------------
        # 普通分支：orig_cls 内的类全部填入各自 prompt
        # ---------------------------------------------------
        else:
            for i, cls_prompt in enumerate(class_texts):
                if cls_prompt in orig_cls:
                    texts[i] = self.prompt_format.format(cls_prompt)

        results['text'] = texts
        return results
