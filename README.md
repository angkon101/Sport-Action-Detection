# Human–Object Interaction-Aware Multimodal CNN–LSTM Framework for Sports Action Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

A Deep Learning framework for Fine-Grained Sports Action Recognition across **82 Sports Categories** (9,151 videos) using a **4-Stream Multimodal Architecture** (Visual, Pose, Object, and HOI Kinematics) with a **2-Layer Bidirectional LSTM** and **Temporal Attention Pooling**.

---

## 📌 Framework Architecture

```
                       Raw Sports Video (16 sampled frames)
                                       │
        ┌───────────────────┬──────────┴──────────┬───────────────────┐
        ▼                   ▼                     ▼                   ▼
1. Visual Stream     2. Posture Stream     3. Object Stream     4. HOI Kinematics
   [ResNet-50]       [Keypoint R-CNN]       [Faster R-CNN]      [Geometric Engine]
  Scene context       17 skeletal joints      Sports gear        Wrist/ankle-to-gear
   & background       + joint velocities      & centroids        distances & vectors
     (2048-d)              (85-d)                (7-d)                 (16-d)
        │                   │                     │                     │
        └───────────────────┼─────────────────────┼─────────────────────┘
                            ▼
          [LayerNorm + GELU Linear Projections]  (Mapped to 512-d joint embedding)
                            │
                            ▼
                [2-Layer Bidirectional LSTM]     (Captures forward & backward motion)
                            │
                            ▼
             [Temporal Attention Pooling Head]   (Learns to focus on key action moments)
                            │
                            ▼
               [82-Class Sports Classifier]
```

---

## 🚀 Execution Environments (3 Mediums)

Detailed, step-by-step instructions for all three platforms are in [**RUN_GUIDE.md**](RUN_GUIDE.md):

| Platform | Runner / Notebook | Description |
| :--- | :--- | :--- |
| **💻 Local Machine** | [`run_local.py`](run_local.py) & [`notebooks/3_Local_Machine_HOI_BiLSTM.ipynb`](notebooks/3_Local_Machine_HOI_BiLSTM.ipynb) | Runs locally on Windows / VS Code with NVIDIA GPU (GTX 1650+). |
| **☁️ Google Colab** | [`notebooks/1_Google_Colab_HOI_BiLSTM.ipynb`](notebooks/1_Google_Colab_HOI_BiLSTM.ipynb) | Cloud execution with T4 / V100 GPU and automated Google Drive backup. |
| **⚡ Kaggle** | [`notebooks/2_Kaggle_HOI_BiLSTM.ipynb`](notebooks/2_Kaggle_HOI_BiLSTM.ipynb) | Cloud execution with Tesla P100 GPU and output zip download. |

---

## 📁 Repository Structure

```text
├── Model Data/                      # 70/30 Stratified Split per sports class
│   ├── Train/                       # 6,408 videos across 82 categories
│   └── Test/                        # 2,743 videos across 82 categories
├── notebooks/
│   ├── 1_Google_Colab_HOI_BiLSTM.ipynb     # Google Colab Master Notebook
│   ├── 2_Kaggle_HOI_BiLSTM.ipynb           # Kaggle P100 Notebook
│   └── 3_Local_Machine_HOI_BiLSTM.ipynb    # Local Machine Interactive Notebook
├── src/
│   ├── models/
│   │   └── hoi_bilstm.py            # HOIBiLSTMClassifier & TemporalAttentionPooling
│   ├── extractors/
│   │   ├── hoi_feature_extractor.py # ResNet-50 + Keypoint R-CNN + Faster R-CNN
│   │   └── batch_extract_features.py# Resumable batch feature extractor
│   ├── datasets/
│   │   ├── hoi_dataset.py           # PyTorch Dataset for cached .pt features
│   │   └── data_splitter.py         # Stratified 70/30 splitter & tensor verifier
│   └── utils/
│       ├── metrics.py               # Top-k accuracy and AverageMeter
│       └── visualizer.py            # 5 thesis publication diagram generators
├── data/
│   ├── scripts/
│   │   └── populate_model_data_split.py # Physical folder splitter for Model Data
│   └── splits/                      # Stratified split JSON index files
├── run_local.py                     # Dedicated Local Machine CLI Runner
├── run_all.py                       # All-in-one CLI pipeline
├── RUN_GUIDE.md                     # Comprehensive execution guide for all 3 mediums
└── requirements.txt
```

---

## ⚡ Quickstart (Local PowerShell)

```powershell
# 1. Quick 2-Minute Smoke Test (2 videos per class)
python run_local.py --extract --data-dir "Model Data" --features-dir "data/features" --max-per-class 2
python run_local.py --verify --features-dir "data/features"
python run_local.py --train --features-dir "data/features" --model-mode M7 --epochs 5 --batch-size 8

# 2. Full Dataset Execution (All 9,151 videos)
python run_local.py --extract --data-dir "Model Data" --features-dir "data/features"
python run_local.py --train --features-dir "data/features" --model-mode M7 --epochs 40 --batch-size 32
```

---

## 📊 Evaluation & Publication Diagrams

Training generates 5 thesis publication-quality diagrams in `diagrams/M7/`:
1. `loss_curves.png`: Training & validation cross-entropy loss curves.
2. `accuracy_curves.png`: Top-1 & Top-5 accuracy progression.
3. `confusion_matrix_82.png`: High-resolution normalized 82-class confusion matrix.
4. `per_class_accuracy.png`: Ranked horizontal bar chart of sports performance.
5. `temporal_attention_weights.png`: Temporal attention heatmap across video keyframes.
6. `evaluation_report.json`: Per-class Precision, Recall, and F1-score report.
