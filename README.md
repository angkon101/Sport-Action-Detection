# CMAT-HOI: Cross-Modal Attention Transformer for Human–Object Interaction Sports Action Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Top-1 Accuracy](https://img.shields.io/badge/Top--1%20Accuracy-90.19%25-brightgreen.svg)]()
[![Top-5 Accuracy](https://img.shields.io/badge/Top--5%20Accuracy-99.02%25-brightgreen.svg)]()

A State-of-the-Art Deep Learning Framework for Fine-Grained Sports Action Recognition across **82 Sports Categories** (9,151 videos) using a **4-Stream Multimodal Architecture** (Visual, Pose, Object, and HOI Kinematics) powered by the **Cross-Modal Attention Transformer (CMAT-HOI)**.

---

## 🏆 Key Performance Metrics (82 Sports Classes)

| Metric | Result | Benchmark Context |
| :--- | :---: | :--- |
| **Top-1 Test Accuracy** | **90.19%** | Evaluated on 2,743 strictly disjoint test videos |
| **Top-5 Test Accuracy** | **99.02%** | Near-perfect top-5 sport retrieval |
| **Weighted Average Precision** | **93.40%** | Minimizes false positive sport detections |
| **Weighted Average Recall** | **90.19%** | Robust recognition across rare and frequent classes |
| **Weighted Average F1-Score** | **90.19%** | Harmonic mean reflecting balanced classification |
| **Test Categorical Loss** | **0.4397** | Rapid, stable convergence with Focal Loss & Label Smoothing |

---

## 📌 Framework Architecture: CMAT-HOI

```
                           Raw Sports Video (16 sampled frames)
                                            │
            ┌───────────────────────┬───────┴───────┬───────────────────────┐
            ▼                       ▼               ▼                       ▼
    1. Visual Stream         2. Pose Stream   3. Object Stream        4. HOI Stream
       [ResNet-50]          [Keypoint R-CNN]   [Faster R-CNN]       [Kinematic Engine]
      Scene context        17 skeletal joints   Sports gear &       Wrist/ankle-to-gear
       & background         + velocities (85d)  centroids (7d)      vectors & dist (16d)
         (2048-d)                   │               │                      │
            │                       └───────┬───────┘                      │
            ▼                               ▼                              ▼
    [Modality Stem 1]           [Bi-Directional Cross-Attn]         [Modality Stem 4]
    (Conv1D + LayerNorm)        (Pose queries Object;               (Conv1D + LayerNorm)
            │                    Object queries Pose)                      │
            │                               │                              │
            └───────────────────────┬───────┴──────────────────────────────┘
                                    ▼
                     [Gated Multimodal Fusion Unit (GMFU)]
                     (Dynamic softmax gates prevent visual dominance)
                                    │
                                    ▼
                     [Temporal Positional Encoding + [CLS] Token]
                                    │
                                    ▼
                     [3-Layer Pre-LN Transformer Encoder]
                     (8 Attention heads, GELU, d_model=256)
                                    │
                                    ▼
                     [Multi-Scale 4-Way Temporal Pooling]
                     ([CLS] Token + Attention Pooling + Max + Avg = 1024-d)
                                    │
                                    ▼
                     [3-Layer MLP Classification Head]
                                    │
                                    ▼
                     [82-Class Softmax Probability Distribution]
```

---

## 🚀 Execution Environments (3 Mediums)

Full step-by-step instructions for all three platforms are in [**RUN_GUIDE.md**](RUN_GUIDE.md):

| Platform | Runner / Notebook | Description |
| :--- | :--- | :--- |
| **💻 Local Machine** | [`run_local.py`](run_local.py) & [`notebooks/3_Local_Machine_HOI_BiLSTM.ipynb`](notebooks/3_Local_Machine_HOI_BiLSTM.ipynb) | Runs locally on Windows with NVIDIA GPU (GTX 1650+). |
| **☁️ Google Colab** | [`notebooks/1_Google_Colab_HOI_BiLSTM.ipynb`](notebooks/1_Google_Colab_HOI_BiLSTM.ipynb) | Cloud execution with T4 / V100 GPU and automated Google Drive backup. |
| **⚡ Kaggle** | [`notebooks/2_Kaggle_HOI_BiLSTM.ipynb`](notebooks/2_Kaggle_HOI_BiLSTM.ipynb) | Cloud execution with Tesla P100 GPU and output zip download. |

---

## 🎥 Real-Time Validation & Video Uploader Environments

We provide **4 complete environments** to test custom sports videos or scan live webcams:

| Interface | Launch Command | Target Audience / Environment |
| :--- | :--- | :--- |
| **🌐 Streamlit Web App** | `streamlit run app_streamlit.py` | Browser-based interactive dashboard with drag-and-drop video upload, side-by-side video playback, and Top-5 probability bars. |
| **🖥️ Native Desktop GUI** | `python app_desktop.py` | Standalone native Windows Tkinter GUI app with file picker, instant video analysis, and looping OpenCV preview. |
| **☁️ Cloud Gradio App** | `python app_web.py --share` | Colab / Kaggle cloud app with a public shareable URL and direct webcam recording in browser. |
| **📹 Real-Time Webcam Scanner** | `python demo_realtime.py --camera 0` | Asynchronous multi-threaded live webcam scanner running at **30+ FPS** without UI freezing. |

---

## 📊 Publication-Quality Thesis Outcome Diagrams

All generated figures are stored in `diagrams/M7/`:

| Diagram | Description | File Path |
| :--- | :--- | :--- |
| **1. Loss Learning Curves** | Convergence curves across training epochs | [loss_curves.png](diagrams/M7/loss_curves.png) |
| **2. Accuracy Curves** | Top-1 (90.19%) and Top-5 (99.02%) accuracy curves | [accuracy_curves.png](diagrams/M7/accuracy_curves.png) |
| **3. 82-Class Confusion Matrix** | High-resolution normalized 82x82 confusion matrix | [confusion_matrix_82.png](diagrams/M7/confusion_matrix_82.png) |
| **4. Per-Class Accuracy Rankings** | Top 10 highest vs 10 lowest performing sports | [per_class_accuracy.png](diagrams/M7/per_class_accuracy.png) |
| **5. Temporal Attention Weights** | Heatmap of frame-level attention allocation | [temporal_attention_weights.png](diagrams/M7/temporal_attention_weights.png) |
| **6. Metric Distributions** | Statistical histogram of Precision, Recall, and F1 | [metric_distribution_82.png](diagrams/M7/metric_distribution_82.png) |
| **7. Multimodal Ablation Study** | Performance progression comparing M1 through M7 | [multimodal_ablation_comparison.png](diagrams/M7/multimodal_ablation_comparison.png) |
| **8. Full Metrics JSON Report** | Complete per-class precision, recall, F1, and support | [evaluation_report.json](diagrams/M7/evaluation_report.json) |

---

## 🔬 Multimodal Stream Ablation Study (M1 – M7)

| Ablation | Visual (ResNet-50) | Pose (Keypoint R-CNN) | Object (Faster R-CNN) | HOI Kinematics | Fusion / Attention | Top-1 Test Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | ✅ | ❌ | ❌ | ❌ | Concatenation | 48.2% |
| **M2** | ❌ | ✅ | ❌ | ❌ | Concatenation | 28.6% |
| **M3** | ❌ | ❌ | ✅ | ❌ | Concatenation | 14.5% |
| **M4** | ✅ | ✅ | ❌ | ❌ | Linear Projection | 62.4% |
| **M5** | ✅ | ✅ | ✅ | ❌ | Linear Projection | 71.8% |
| **M6** | ✅ | ✅ | ✅ | ✅ | Linear Projection | 81.3% |
| **M7 (CMAT-HOI)** | ✅ | ✅ | ✅ | ✅ | **Cross-Attention + GMFU + Transformer** | **90.19%** |

---

## ⚡ Quickstart (Local Windows PowerShell)

```powershell
# 1. Run inference on any custom video:
python demo_realtime.py --video "path/to/video.mp4"

# 2. Launch the Streamlit Web Dashboard:
streamlit run app_streamlit.py

# 3. Launch the Native Windows GUI:
python app_desktop.py

# 4. Evaluate the trained CMAT-HOI model on the test split:
python run_local.py --test --checkpoint checkpoints/M7/best_model.pth

# 5. Train CMAT-HOI from scratch:
python run_local.py --train --model-mode M7 --arch transformer --epochs 45 --batch-size 64 --lr 0.0004
```

---

## 📁 Repository Structure

```text
├── Model Data/                      # 70/30 Stratified Split per sports class
│   ├── Train/                       # 6,408 videos across 82 categories
│   └── Test/                        # 2,743 videos across 82 categories
├── data/
│   ├── features/                    # Pre-extracted 4-stream multimodal .pt files (0 NaNs)
│   │   ├── Train/                   # 6,408 tensor files
│   │   └── Test/                    # 2,743 tensor files
│   └── splits/                      # Stratified split JSON index files
├── diagrams/
│   └── M7/                          # 7 thesis publication figures + metrics report
├── checkpoints/
│   └── M7/                          # Best trained CMAT-HOI model (best_model.pth)
├── src/
│   ├── models/
│   │   ├── hoi_transformer.py       # CMAT-HOI: Cross-Modal Attention Transformer (Proposed)
│   │   └── hoi_bilstm.py            # Legacy BiLSTM + Temporal Attention baseline
│   ├── extractors/
│   │   ├── hoi_feature_extractor.py # ResNet-50 + Keypoint R-CNN + Faster R-CNN
│   │   └── batch_extract_features.py# Resumable batch feature extractor
│   ├── datasets/
│   │   └── hoi_dataset.py           # Preloaded PyTorch Dataset (0ms RAM seek)
│   ├── utils/
│   │   ├── losses.py                # FocalLossWithSmoothing & SoftTargetCrossEntropy
│   │   ├── augmentation.py          # MultimodalAugmentor (Mixup, Jitter, Masking)
│   │   ├── metrics.py               # Top-1/Top-5 accuracy & AverageMeter
│   │   └── visualizer.py            # Publication diagram generators
│   └── inference.py                 # Multi-stream video inference & HUD annotator
├── app_streamlit.py                 # Interactive Streamlit Web Dashboard
├── app_desktop.py                   # Standalone Tkinter Native Windows GUI
├── app_web.py                       # Cloud Gradio Web App with webcam recording
├── demo_realtime.py                 # Threaded real-time webcam scanner (30+ FPS)
├── run_local.py                     # Dedicated Local Machine CLI Runner
├── RUN_GUIDE.md                     # Step-by-step master execution guide
└── requirements.txt
```

---

## 📜 Citation & License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
