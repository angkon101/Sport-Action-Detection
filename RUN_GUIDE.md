# Master Execution Guide: Human–Object Interaction (HOI) Multimodal BiLSTM Framework

This guide provides end-to-end instructions for running the **Human–Object Interaction (HOI) Multimodal BiLSTM Sports Action Recognition framework** across three distinct environments:

1. **Local Machine (VS Code / Windows PowerShell)**
2. **Google Colab (Cloud GPU: T4 / V100 / A100)**
3. **Kaggle (Cloud GPU: Tesla P100 / 2x T4)**

---

## 📌 Quick Environment Comparison

| Medium | Hardware | Setup Effort | Storage / Data Access | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **Local Machine** | NVIDIA GTX 1650 (4 GB VRAM) | Zero (Ready now) | Direct access to `Model Data` on Drive `E:` | Fast smoke tests, local debugging, direct file inspection |
| **Google Colab** | NVIDIA T4 / V100 / A100 (16 GB) | ~2 minutes | Mounts Google Drive (`Model Data.zip`) | Fast feature extraction & full 40-epoch training |
| **Kaggle** | NVIDIA Tesla P100 (16 GB) | ~2 minutes | Kaggle Input dataset (`/kaggle/input`) | Alternative free 30 hrs/week cloud training |

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
- **Overlap**: **0 videos** ($\text{Train} \cap \text{Test} = \emptyset$, mathematically verified).

---

## 💻 Medium 1: Local Machine Execution (VS Code / PowerShell)

### Method A: Command-Line Interface (`run_local.py`)

Open your VS Code Terminal (PowerShell) in `E:\Akhi Thesis`:

#### 1. Quick 2-Minute Smoke Test *(Recommended First)*
Extracts 2 videos per category, verifies integrity, trains M7 for 5 epochs, and generates all 5 thesis diagrams:
```powershell
# Step 1: Quick test extraction
python run_local.py --extract --data-dir "Model Data" --features-dir "data/features" --max-per-class 2

# Step 2: Pre-flight tensor integrity check
python run_local.py --verify --features-dir "data/features"

# Step 3: Train M7 (Full Multimodal + Attention)
python run_local.py --train --features-dir "data/features" --model-mode M7 --epochs 5 --batch-size 8
```

#### 2. Full Dataset Run *(All 9,151 Videos)*
```powershell
# Step 1: Extract all 9,151 videos (takes ~1.5 - 2 hrs on GTX 1650, resumable if paused)
python run_local.py --extract --data-dir "Model Data" --features-dir "data/features"

# Step 2: Pre-flight verification
python run_local.py --verify --features-dir "data/features"

# Step 3: Full 40-Epoch Training & Thesis Diagrams Generation
python run_local.py --train --features-dir "data/features" --model-mode M7 --epochs 40 --batch-size 32 --lr 0.0005
```

### Method B: Interactive Local Notebook
Open [`notebooks/3_Local_Machine_HOI_BiLSTM.ipynb`](notebooks/3_Local_Machine_HOI_BiLSTM.ipynb) in VS Code, select your Python 3.11 kernel, and run cells sequentially.

---

## ☁️ Medium 2: Google Colab Execution

Use Google Colab when you want faster feature extraction and training using Colab's high-memory GPUs (T4 or A100).

### Step-by-Step Instructions:

1. **Prepare the Dataset in Google Drive**:
   - Compress `Model Data` into a zip file named `Model Data.zip` (or upload `ucf action.zip`).
   - Place `Model Data.zip` into the root of your Google Drive (`MyDrive/Model Data.zip`).

2. **Open the Colab Notebook**:
   - Go to [Google Colab](https://colab.research.google.com/).
   - Click **File -> Open Notebook -> GitHub** (or upload [`notebooks/1_Google_Colab_HOI_BiLSTM.ipynb`](notebooks/1_Google_Colab_HOI_BiLSTM.ipynb)).
   - Set the runtime to GPU: **Runtime -> Change runtime type -> T4 GPU**.

3. **Execute the Cells in Order**:
   - **Cell 1**: Check GPU (`!nvidia-smi`).
   - **Cell 2**: Clones the GitHub repository:
     ```bash
     !git clone https://github.com/angkon101/Sport-Action-Detection.git
     %cd Sport-Action-Detection
     !pip install -q -r requirements.txt
     ```
   - **Cell 3**: Mounts Google Drive and auto-unzips `Model Data.zip`:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     !unzip -q -o "/content/drive/MyDrive/Model Data.zip" -d ./
     ```
   - **Cell 4**: Extracts features with GPU acceleration:
     ```bash
     !python run_local.py --extract --data-dir "Model Data" --features-dir "data/features"
     !python run_local.py --verify --features-dir "data/features"
     ```
   - **Cell 5**: Trains the model:
     ```bash
     !python run_local.py --train --features-dir "data/features" --model-mode M7 --epochs 40 --batch-size 32 --lr 0.0005
     ```
   - **Cell 6**: Displays all 5 thesis outcome diagrams inline in the notebook.
   - **Cell 7**: Auto-backs up `checkpoints/` and `diagrams/` back to your Google Drive (`/content/drive/MyDrive/Sports_Action_Results/`).

---

## ⚡ Medium 3: Kaggle Execution

Kaggle provides free access to NVIDIA Tesla P100 GPUs (up to 30 hours per week).

### Step-by-Step Instructions:

1. **Upload Dataset to Kaggle**:
   - Go to [Kaggle Datasets](https://www.kaggle.com/datasets) -> Click **New Dataset**.
   - Upload `Model Data.zip` (or `ucf action.zip`). Name it `sports-model-data`.

2. **Create a Kaggle Notebook**:
   - Click **New Notebook**.
   - In the right-hand panel:
     - Under **Accelerator**, choose **GPU P100** or **GPU T4 x 2**.
     - Under **Internet**, toggle to **Internet on**.
     - Under **Input**, click **+ Add Data** and select your uploaded `sports-model-data`.

3. **Load the Kaggle Notebook**:
   - Click **File -> Import Notebook** and upload [`notebooks/2_Kaggle_HOI_BiLSTM.ipynb`](notebooks/2_Kaggle_HOI_BiLSTM.ipynb).
   - Run all cells sequentially.

4. **Retrieve Results**:
   - After training completes, the notebook automatically saves all diagrams and `best_model.pth` to `/kaggle/working/Sports_Action_Results.zip`.
   - Download the zip directly from Kaggle's right-hand **Output** pane.

---

## 🎥 Real-Time Action Validation & Live Camera Scanning

You can validate that the model is actively detecting human postures, sports gear, and actions on **new custom videos** or **live webcams** across all three platforms:

### 1. Local Machine Validation (GUI & Webcam)
Using [`demo_realtime.py`](demo_realtime.py):

```powershell
# A. Scan live camera / webcam in real time:
python demo_realtime.py --camera 0

# B. Upload and test ANY custom video file:
python demo_realtime.py --video "C:\path\to\my_sports_video.mp4"

# C. Test and export an annotated video:
python demo_realtime.py --video "test.mp4" --output "results/annotated_test.mp4"

# D. Test a random sample from the dataset:
python demo_realtime.py --sample
```

**What you will see in the real-time window:**
- **Neon Green Bones & Dots**: The 17 skeletal joint posture estimated by Keypoint R-CNN.
- **Neon Yellow Box**: The detected sports equipment tracked by Faster R-CNN.
- **Purple Connecting Vectors**: The geometric HOI lines connecting the athlete's wrists and ankles to the sports gear.
- **Top HUD Banner**: Displays the detected sport action with confidence score and Top-3 runners-up.

---

### 2. Google Colab & Kaggle Validation (Interactive Web App)
Cloud platforms do not have a physical desktop window, so we provide an interactive **Gradio Web App** (`app_web.py`):

In your Colab or Kaggle notebook, run:
```bash
!pip install -q gradio
!python app_web.py --share
```

**Features in the Cloud Web App:**
1. **Public URL**: Generates a live link (`https://xxxx.gradio.live`) accessible from any browser or phone.
2. **Drag & Drop Upload**: Test any MP4, AVI, or MOV video clip.
3. **Record from Webcam**: Click "Record from Webcam" directly in the browser to capture a live action and test it immediately.
4. **Interactive Dashboard**: Plays the annotated video with skeleton/HOI overlays and displays an interactive Top-5 probability bar chart.

---

## 🔬 The 7 Model Ablations (M1 – M7)

You can benchmark any of the 7 ablation models by setting `--model-mode <ID>`:

| Model ID | Visual (ResNet-50) | Pose (Keypoint R-CNN) | Object (Faster R-CNN) | HOI Kinematics | Temporal Attention | Purpose in Thesis |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M1** | ✅ | ❌ | ❌ | ❌ | ❌ | Baseline Visual CNN + BiLSTM |
| **M2** | ❌ | ✅ | ❌ | ❌ | ❌ | Pure Skeleton Motion Classifier |
| **M3** | ❌ | ❌ | ✅ | ❌ | ❌ | Sports Equipment Kinematics Only |
| **M4** | ✅ | ✅ | ❌ | ❌ | ❌ | Scene Context + Body Articulation |
| **M5** | ✅ | ✅ | ✅ | ❌ | ❌ | Adding Equipment Tracking |
| **M6** | ✅ | ✅ | ✅ | ✅ | ❌ | Multimodal Fusion without Attention |
| **M7** | ✅ | ✅ | ✅ | ✅ | ✅ | **Full Proposed Framework (Best Accuracy)** |

Example command:
```powershell
python run_local.py --train --model-mode M7 --epochs 40
```

---

## 📊 Generated Thesis Outcome Diagrams

Every training run automatically generates high-resolution figures in `diagrams/<model_mode>/`:

1. **`loss_curves.png`**: Epoch-by-epoch training and validation loss curves (proves smooth convergence).
2. **`accuracy_curves.png`**: Top-1 and Top-5 accuracy progression over training.
3. **`confusion_matrix_82.png`**: 82x82 normalized confusion matrix showing inter-class distinctions.
4. **`per_class_accuracy.png`**: Horizontal ranked bar chart highlighting top-performing sports actions.
5. **`temporal_attention_weights.png`**: Frame-by-frame heatmap showing which video frames the attention mechanism highlighted.
6. **`evaluation_report.json`**: Precision, Recall, and F1-score for each of the 82 categories.

---

## ❓ Troubleshooting & FAQs

- **Q: How long does feature extraction take?**
  - On your local GTX 1650: ~1.5 to 2 hours for all 9,151 videos.
  - On Colab T4 / Kaggle P100: ~35 to 50 minutes.
  - *Extraction only needs to run once!* All features are cached as `.pt` files.
- **Q: Can I stop extraction and resume later?**
  - Yes! The script automatically checks if a `.pt` file exists. If you stop and restart, it skips already-extracted videos and resumes immediately.
- **Q: How long does model training take?**
  - Because features are pre-cached, training 40 epochs on your local GPU takes **only ~3 to 5 minutes**!

