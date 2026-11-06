# Sports Action Recognition & Detection

Deep learning framework for Sports Action Recognition and Detection using the UCF benchmark datasets (UCF101 / UCF-Sports).

---

## 📌 Project Overview

This repository provides an end-to-end pipeline for:
- Video ingestion, frame extraction, and temporal sampling.
- Spatial and temporal data augmentations for action recognition.
- Baseline and state-of-the-art deep learning architectures (e.g., 3D CNNs like R(2+1)D / ResNet3D, 2D CNN + Temporal Encoders, Video Transformers).
- Experiment tracking, validation metrics (Top-1 / Top-5 Accuracy, mAP, Confusion Matrices), and inference demos.

---

## 📁 Repository Structure

```text
├── configs/                 # YAML configuration files for experiments
├── data/                    # Dataset directory (raw downloads, splits, annotations)
│   ├── splits/              # Train / val / test split txt or csv files
│   └── scripts/             # Download and preprocessing scripts
├── src/
│   ├── datasets/            # Video dataset loaders and transform pipelines
│   ├── models/              # Model architectures and temporal backbones
│   ├── utils/               # Metrics, visualization, video I/O helpers
│   ├── train.py             # Model training entrypoint
│   ├── evaluate.py          # Validation / test evaluation pipeline
│   └── inference.py         # Single video & webcam action recognition demo
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate       # On Linux / macOS
# or
.venv\Scripts\activate          # On Windows

pip install -r requirements.txt
```

> **Note on CUDA**: For GPU acceleration on Windows, install the PyTorch build with CUDA support matching your system:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 📊 Datasets

- **UCF101**: 13,320 videos spanning 101 action categories across 5 broad types (including sport actions such as Basketball, Biking, Diving, Golf Swing, Tennis, etc.).
- **UCF-Sports**: 150 sports video sequences across 10 actions with spatial bounding box annotations for action localization and detection.
