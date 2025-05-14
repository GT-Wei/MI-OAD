<div align="center">

# From Word to Sentence: A Large‑Scale Multi‑Instance Dataset for **Open‑Set Aerial Detection**

**Dataset Name:** **MI‑OAD**  
**Annotation Method:** **OS‑W2S Label Engine**

</div>

---

## ✨ Abstract
In recent years, language-guided open-world aerial object detection has gained significant attention due to its better alignment with real-world application needs. However, due to limited datasets, most existing language-guided methods primarily focus on vocabulary, which fails to meet the demands of more fine-grained open-world detection. To address this limitation, we propose constructing a large-scale language-guided open-set aerial detection dataset, encompassing three levels of language guidance: from words to phrases, and ultimately to sentences. Centered around an open-source large vision-language model and integrating image-operation-based preprocessing with BERT-based postprocessing, we present the OS-W2S Label Engine, an automatic annotation pipeline capable of handling diverse scene annotations for aerial images. Using this label engine, we expand existing aerial detection datasets with rich textual annotations and construct a novel benchmark dataset, called Multi-instance Open-set Aerial Dataset (MI-OAD), addressing the limitations of current remote sensing grounding data and enabling effective open-set aerial detection. Specifically, MI-OAD contains 163,023 images and 2 million image-caption pairs, approximately 40 times larger than the comparable datasets.
We also employ state-of-the-art open-set methods from the natural image domain, trained on our proposed dataset, to validate the model’s open-set detection capabilities.

---

## 🔧 Quick Links
| | |
|---|---|
| **MMDetection 3.3.0 Install Guide** | <https://mmdetection.readthedocs.io/zh-cn/latest/get_started.html> |
| **YOLO‑World Repository** | <https://github.com/AILab-CVC/YOLO-World> |
| MMDetection configs (MI‑OAD) | `mmdetection/projects/MI-OAD` |
| YOLO‑World configs (MI‑OAD) | `YOLO-World_RSSD/configs/MI-OAD` |

---

## ⚙️ Installation

> For full details, follow the official links above. Below is a minimal recipe.

```bash
# 1. Environment
conda create -n openmm python=3.10 -y
conda activate openmm

# 2. MMDetection 3.3.0 (+ MMEngine & MMCV)
pip install openmim
mim install "mmcv>=2.0.0"
cd mmdetection && git checkout v3.3.0
pip install fairscale transformers
pip install -e .
cd ..

# 3. YOLO‑World
conda create -n yolo-world python=3.9 -y
conda activate yolo-world
cd YOLO-World
pip install -r requirements.txt
pip install -e .
```
---

## 🚀 Getting Started

```bash
# MMDetection example
bash mmdetection/tools/MIOAD_Pretrian.sh


# YOLO‑World example
bash YOLO-World/tools/MIOAD_Pretrain.sh
```

---

## 📁 Dataset Structure
```
MI-OAD/
  Caption/ 
  Detection/
  datasets_categories_list/
  images/
```
> Download links will be provided upon release.

