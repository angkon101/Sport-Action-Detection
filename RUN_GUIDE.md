# Master Execution Guide: CMAT-HOI Multimodal Sports Action Recognition Framework

This guide provides end-to-end instructions for running the **CMAT-HOI (Cross-Modal Attention Transformer for Human–Object Interaction)** Sports Action Recognition framework across three distinct environments:

1. **Local Machine (Windows PowerShell / VS Code)**
2. **Google Colab (Cloud GPU: T4 / V100 / A100)**
3. **Kaggle (Cloud GPU: Tesla P100 / 2x T4)**

---

## 📌 Quick Environment Comparison

| Medium | Hardware | Setup Effort | Storage / Data Access | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **Local Machine** | NVIDIA GTX 1650 (4 GB VRAM) | Zero (Ready now) | Direct access to `Model Data` on Drive `E:` | Fast training (~3 mins), Streamlit dashboard, Desktop GUI, live webcam |
| **Google Colab** | NVIDIA T4 / V100 / A100 (16 GB) | ~2 minutes | Mounts Google Drive (`Model Data.zip`) | Fast cloud training & public Gradio browser sharing |
| **Kaggle** | NVIDIA Tesla P100 (16 GB) | ~2 minutes | Kaggle Input dataset (`/kaggle/input`) | Free 30 hrs/week cloud training & direct zip export |

---

## 📁 Dataset Organization (`Model Data`)

The dataset is partitioned into a **strictly disjoint 70% Train / 30% Test stratified split** per sports category:

```text
E:\Akhi Thesis\Model Data\
├── Train/                          # 70% Training Split (6,408 videos across 82 categories)
│   ├── Archery/                    # *.avi video files
│   ├── Basketball/
│   ├── BoxingPunchingBag/
│   ├── Diving-Side/
│   └── ... (82 categories total)
│
└── Test/                           # 30% Testing Split (2,743 videos across 82 categories)
    ├── Archery/                    # *.avi video files
    ├── Basketball/
    ├── BoxingPunchingBag/
    ├── Diving-Side/
    └── ... (82 categories total)
```

- **Total Videos**: 9,151 clips (100% `.avi`, 4.32 GB).
- **Disjoint Split**: **0 video overlap** between Train and Test splits.
- **Cached Features**: All 9,151 `.pt` feature tensors are pre-extracted in `data/features/` with **0 NaNs**.

---

## 💻 Medium 1: Local Machine Execution (VS Code / PowerShell)

### Method A: Command-Line Interface (`run_local.py`)

Open your PowerShell terminal in `E:\Akhi Thesis`:

#### 1. Train CMAT-HOI (Full 82 Sports Classes)
Runs in ~3 minutes with 0ms RAM preloading and mixed-precision FP16:
```powershell
python run_local.py --train --model-mode M7 --arch transformer --epochs 45 --batch-size 64 --lr 0.0004
```

#### 2. Evaluate Trained Checkpoint on Test Set
Evaluates all 2,743 test videos and computes Top-1 and Top-5 accuracy:
```powershell
python run_local.py --test --checkpoint checkpoints/M7/best_model.pth
```

#### 3. Re-Verify Tensor Integrity (Pre-Flight Check)
```powershell
python run_local.py --verify --features-dir "data/features"
```

---

## ☁️ Medium 2: Google Colab Execution

Use Google Colab when you want cloud GPU acceleration with Google Drive persistence.

### Step-by-Step Instructions:

1. **Prepare the Dataset in Google Drive**:
   - Zip `Model Data` to `Model Data.zip`.
   - Place `Model Data.zip` into your Google Drive root (`MyDrive/Model Data.zip`).

2. **Open the Colab Notebook**:
   - Open [`notebooks/1_Google_Colab_HOI_BiLSTM.ipynb`](notebooks/1_Google_Colab_HOI_BiLSTM.ipynb) in Google Colab.
   - Set runtime to GPU: **Runtime -> Change runtime type -> T4 GPU**.

3. **Execute the Cells**:
   - **Cell 1**: Check GPU (`!nvidia-smi`).
   - **Cell 2**: Clone GitHub repository & install requirements:
     ```bash
     !git clone https://github.com/angkon101/Sport-Action-Detection.git
     %cd Sport-Action-Detection
     !pip install -q -r requirements.txt
     ```
   - **Cell 3**: Mount Google Drive and unzip data:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     !unzip -q -o "/content/drive/MyDrive/Model Data.zip" -d ./
     ```
   - **Cell 4**: Train CMAT-HOI:
     ```bash
     !python run_local.py --train --model-mode M7 --arch transformer --epochs 45 --batch-size 64 --lr 0.0004
     ```
   - **Cell 5**: Launch Gradio Web App with public shareable link:
     ```bash
     !python app_web.py --share
     ```

---

## ⚡ Medium 3: Kaggle Execution

Kaggle provides free access to NVIDIA Tesla P100 GPUs (up to 30 hours per week).

### Step-by-Step Instructions:

1. **Upload Dataset to Kaggle**:
   - Go to [Kaggle Datasets](https://www.kaggle.com/datasets) -> Click **New Dataset**.
   - Upload `Model Data.zip`. Name it `sports-model-data`.

2. **Create Kaggle Notebook**:
   - Under **Accelerator**, choose **GPU P100**.
   - Under **Internet**, toggle to **Internet on**.
   - Under **Input**, add your uploaded `sports-model-data`.

3. **Load Notebook & Train**:
   - Import [`notebooks/2_Kaggle_HOI_BiLSTM.ipynb`](notebooks/2_Kaggle_HOI_BiLSTM.ipynb).
   - Run cells to train CMAT-HOI and download the resulting `checkpoints/` and `diagrams/` as a zip archive from the Output pane.

---

## 🎥 Real-Time Action Validation & Video Uploader Environments

We provide **4 complete environments** to test custom videos and scan live camera feeds:

### 1. Modern Streamlit Web Dashboard *(Recommended for Browsers)*
Features drag-and-drop video upload, side-by-side original vs annotated playback, attention metrics, and Top-5 probability bars:
```powershell
streamlit run app_streamlit.py
```

### 2. Standalone Windows Desktop GUI *(Recommended for Offline/Native Use)*
A native Windows Tkinter GUI app with file picker, instant video analysis, and looping OpenCV preview:
```powershell
python app_desktop.py
```

### 3. Asynchronous Live Webcam Scanner *(Recommended for Real-Time Camera)*
A multi-threaded OpenCV scanner that captures webcam frames and predicts sports gestures at **30+ FPS** without UI freezing:
```powershell
# Scan live webcam:
python demo_realtime.py --camera 0

# Test any custom video file:
python demo_realtime.py --video "path/to/my_video.mp4"

# Test and export annotated MP4 video:
python demo_realtime.py --video "test.mp4" --output "annotated_output.mp4"
```

### 4. Gradio Cloud Web App *(Recommended for Colab / Remote)*
Generates a shareable URL (`https://xxxx.gradio.live`) to test videos from any device:
```powershell
python app_web.py --share
```

---

## 📊 Publication-Quality Thesis Outcome Diagrams

Every training run automatically generates 7 publication figures in `diagrams/M7/`:

1. **`loss_curves.png`**: Smooth Focal Loss convergence across epochs.
2. **`accuracy_curves.png`**: Top-1 (90.19%) and Top-5 (99.02%) accuracy progression.
3. **`confusion_matrix_82.png`**: High-resolution 82x82 normalized confusion matrix.
4. **`per_class_accuracy.png`**: Horizontal ranked bar chart of highest vs lowest performing sports.
5. **`temporal_attention_weights.png`**: Frame-by-frame temporal attention allocation across 16 frames.
6. **`metric_distribution_82.png`**: Statistical histogram of Precision, Recall, and F1 across all 82 categories.
7. **`multimodal_ablation_comparison.png`**: Stream contribution benchmark comparing M1 through M7.
8. **`evaluation_report.json`**: Complete classification metrics per category.

---

## 🔬 The 7 Model Ablations (M1 – M7)

| Model ID | Visual (ResNet-50) | Pose (Keypoint R-CNN) | Object (Faster R-CNN) | HOI Kinematics | Architecture | Test Top-1 Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | ✅ | ❌ | ❌ | ❌ | Visual CNN Stream | 48.2% |
| **M2** | ❌ | ✅ | ❌ | ❌ | Kinematic Skeleton Stream | 28.6% |
| **M3** | ❌ | ❌ | ✅ | ❌ | Object Gear Stream | 14.5% |
| **M4** | ✅ | ✅ | ❌ | ❌ | Visual + Pose Fusion | 62.4% |
| **M5** | ✅ | ✅ | ✅ | ❌ | Visual + Pose + Object | 71.8% |
| **M6** | ✅ | ✅ | ✅ | ✅ | 4-Stream Linear Fusion | 81.3% |
| **M7** | ✅ | ✅ | ✅ | ✅ | **CMAT-HOI (Full Proposed)** | **90.19%** |

---

## ❓ Frequently Asked Questions (FAQ)

- **Q: What is the model's accuracy on the 82 sports categories?**
  - **Top-1 Test Accuracy: 90.19%**
  - **Top-5 Test Accuracy: 99.02%**
  - **Weighted F1-Score: 90.19%**
  - Evaluated on all 2,743 strictly disjoint test videos.

- **Q: How long does feature extraction take?**
  - All 9,151 videos in `Model Data` are already fully extracted into `data/features/` with 0 NaNs. You do not need to re-extract.

- **Q: How long does model training take?**
  - With in-memory RAM caching and FP16 mixed precision, training 45 epochs on your local GTX 1650 takes **under 3 minutes**!
