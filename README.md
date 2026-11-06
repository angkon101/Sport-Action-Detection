# Sports Action Recognition on UCF101 Sports Dataset

A modular Deep Learning framework for Sports Action Recognition using the **UCF101 Sports Action Subset** with factorized 3D Convolutional Neural Networks (**R(2+1)D**, **R3D**, **MC3**, **S3D**).

---

## 📌 Project Overview

This repository is designed for thesis research on video-based sports action recognition. It provides:
- **Curated Sports Action Taxonomy**: 45 sports-related action categories extracted from the UCF101 dataset.
- **Efficient Temporal Video Loading**: OpenCV-based frame extraction with uniform and random temporal stride sampling.
- **Factorized 3D Spatio-Temporal Architectures**: R(2+1)D-18 and 3D ResNet models pretrained on Kinetics-400 for strong transfer learning.
- **Memory-Efficient Training**: Mixed Precision (FP16) and gradient accumulation tuned to fit comfortably within 4GB VRAM (GTX 1650).
- **Thesis-Ready Evaluation**: Top-1 and Top-5 accuracy, per-class Precision/Recall/F1 metrics, and high-resolution confusion matrix generation.
- **Inference & Visualization**: Live inference script with confidence score banners overlaid on video clips.

---

## 📁 Repository Structure

```text
├── configs/
│   └── default.yaml                 # Experiment hyperparameters and model settings
├── data/
│   ├── raw/                         # Raw UCF101 video files (e.g. data/raw/<Class>/<video>.avi)
│   ├── splits/                      # Train/test split files and sports_classes.txt
│   └── scripts/
│       ├── download_ucf101.py       # Helper script to download UCF101 splits
│       └── prepare_sports_subset.py # Filters official UCF101 splits into sports subset
├── src/
│   ├── datasets/
│   │   ├── sports_classes.py        # 45 sports class definitions and mapping
│   │   ├── transforms.py            # Video temporal & spatial augmentations
│   │   └── ucf_sports_dataset.py    # PyTorch Dataset for video clip loading
│   ├── models/
│   │   └── action_classifier.py     # 3D CNN backbones (R(2+1)D, R3D, MC3, S3D)
│   ├── utils/
│   │   └── metrics.py               # Top-k accuracy, confusion matrix plotting
│   ├── train.py                     # Main training pipeline with mixed precision
│   ├── evaluate.py                  # Evaluation, classification report, and CM plot
│   └── inference.py                 # Single video inference and demo annotation
├── requirements.txt
└── README.md
```

---

## 🏋️‍♂️ 45 Sports Action Classes

The sports action subset covers:
> *Archery, BaseballPitch, Basketball, BasketballDunk, BenchPress, Biking, Billiards, Bowling, BoxingPunchingBag, BoxingSpeedBag, CleanAndJerk, CricketBowling, CricketShot, Diving, Fencing, FrisbeeCatch, GolfSwing, HorseRace, HorseRiding, IceDancing, JavelinThrow, JumpRope, Kayaking, PoleVault, PullUps, PushUps, RockClimbingIndoor, RopeClimbing, Rowing, Shotput, Skateboarding, Skiing, Skijet, SkyDiving, SoccerJuggling, SoccerPenalty, SumoWrestling, Surfing, Swing, TableTennisShot, TaiChi, TennisSwing, ThrowDiscus, TrampolineJumping, VolleyballSpiking.*

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Optional: create virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Windows

# Install dependencies
pip install -r requirements.txt
```

> **CUDA Support**: For NVIDIA GPU acceleration (e.g. GTX 1650):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Dataset Setup

1. **Download UCF101 Splits**:
   ```bash
   python data/scripts/download_ucf101.py
   ```
2. **Generate Sports Action Splits**:
   ```bash
   python data/scripts/prepare_sports_subset.py
   ```
3. **Download UCF101 Videos**:
   Download the UCF101 dataset archive (`UCF101.rar`) from [CRCV UCF101 Official Source](https://www.crcv.ucf.edu/data/UCF101.php) and extract the video folders into `data/raw/` (e.g. `data/raw/Basketball/v_Basketball_g01_c01.avi`).

### 3. Model Training

Train the factorized 3D CNN (R(2+1)D-18) initialized with Kinetics-400 weights:

```bash
python src/train.py --config configs/default.yaml
```

Useful overrides:
```bash
python src/train.py --batch-size 8 --epochs 25 --lr 0.0003 --backbone r2plus1d_18
```

Checkpoints will be saved in `checkpoints/best_model.pth`.

### 4. Evaluation & Metrics

Evaluate the best trained model on the test split:

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```
Outputs:
- `reports/classification_report.json`: Precision, Recall, and F1 per sport action.
- `reports/confusion_matrix.png`: Normalized confusion matrix visualization.

### 5. Video Inference & Demo

Run inference on any sports video clip:

```bash
python src/inference.py --video path/to/sample_video.mp4 --checkpoint checkpoints/best_model.pth --output demo_annotated.mp4
```
