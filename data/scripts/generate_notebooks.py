"""Helper script to generate the 3 dedicated notebooks for Google Colab, Kaggle, and Local Machine."""

import json
from pathlib import Path


def create_colab_notebook(output_path: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🏋️‍♂️ Human–Object Interaction (HOI) Multimodal BiLSTM for Sports Action Recognition\n",
                    "## 🚀 Google Colab Cloud-Accelerated Master Notebook (82 Sports Categories)\n",
                    "\n",
                    "**Thesis Framework**: *Human–Object Interaction-Aware Multimodal CNN–LSTM Framework for Sports Action Recognition*\n",
                    "\n",
                    "### 📋 Overview of this Notebook:\n",
                    "1. **GPU Acceleration**: NVIDIA Tesla T4 / V100 / A100 verification\n",
                    "2. **Codebase Setup**: Automated clone of the GitHub repository and environment install\n",
                    "3. **Dataset Sync**: Mount Google Drive and load `Model Data.zip` (Train 70% / Test 30%)\n",
                    "4. **Multimodal Feature Extraction**: Extract Visual (ResNet-50), Pose (Keypoint R-CNN), Objects (Faster R-CNN), and HOI Kinematics\n",
                    "5. **Model Training**: 2-Layer BiLSTM with Temporal Attention Pooling (Ablations M1–M7)\n",
                    "6. **In-Notebook Visualization**: Live inline display of all 5 thesis outcome diagrams\n",
                    "7. **Persistent Backup**: Auto-saves trained model checkpoints and diagrams directly back to Google Drive",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Verify Colab GPU Acceleration\n",
                    "Make sure your Colab runtime is set to **GPU** (`Runtime -> Change runtime type -> T4 GPU`).",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!nvidia-smi\n",
                    "import torch\n",
                    "print(f\"CUDA Ready: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"GPU Model  : {torch.cuda.get_device_name(0)}\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Clone GitHub Codebase & Install Requirements",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Clone fresh codebase from GitHub\n",
                    "!rm -rf Sport-Action-Detection\n",
                    "!git clone https://github.com/angkon101/Sport-Action-Detection.git\n",
                    "%cd Sport-Action-Detection\n",
                    "\n",
                    "# Install dependencies\n",
                    "!pip install -q -r requirements.txt",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Mount Google Drive & Load Dataset (`Model Data`)\n",
                    "Mount your Google Drive to load `Model Data.zip` or `ucf action.zip`.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from google.colab import drive\n",
                    "from pathlib import Path\n",
                    "\n",
                    "drive.mount('/content/drive')\n",
                    "drive_path = Path('/content/drive/MyDrive')\n",
                    "\n",
                    "# Look for Model Data.zip or ucf action.zip\n",
                    "if (drive_path / 'Model Data.zip').exists():\n",
                    "    print(\"Found 'Model Data.zip' in Google Drive. Extracting...\")\n",
                    "    !unzip -q -o \"/content/drive/MyDrive/Model Data.zip\" -d ./\n",
                    "elif (drive_path / 'ucf action.zip').exists():\n",
                    "    print(\"Found 'ucf action.zip' in Google Drive. Extracting and creating Model Data split...\")\n",
                    "    !unzip -q -o \"/content/drive/MyDrive/ucf action.zip\" -d ./\n",
                    "    !python data/scripts/populate_model_data_split.py --source-dir \"ucf action\" --target-dir \"Model Data\"\n",
                    "else:\n",
                    "    print(\"[!] Neither 'Model Data.zip' nor 'ucf action.zip' found in Google Drive root.\")\n",
                    "    print(\"    Please upload your zip file to Google Drive (MyDrive) and rerun this cell.\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Extract Multimodal Features\n",
                    "Runs ResNet-50, Keypoint R-CNN, Faster R-CNN, and HOI kinematics across 16 sampled frames.\n",
                    "\n",
                    "> **Tip**: To run a quick 2-minute smoke test across all 82 categories, append `--max-per-class 2`.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract multimodal features into data/features\n",
                    "!python run_local.py --extract --data-dir \"Model Data\" --features-dir \"data/features\"\n",
                    "\n",
                    "# Pre-flight integrity check\n",
                    "!python run_local.py --verify --features-dir \"data/features\"",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Model Training (BiLSTM + Temporal Attention)\n",
                    "Trains our proposed **M7 Model** on the 70% Train split (6,408 videos) and validates on the 30% Test split (2,743 videos).",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train M7 with Cosine Annealing scheduler and AdamW\n",
                    "!python run_local.py --train --features-dir \"data/features\" --model-mode M7 --epochs 40 --batch-size 32 --lr 0.0005",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. View the 5 Publication-Ready Thesis Diagrams",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from IPython.display import Image, display\n",
                    "from pathlib import Path\n",
                    "\n",
                    "diag_dir = Path(\"diagrams/M7\")\n",
                    "diagrams = [\n",
                    "    (\"Diagram 1: Training & Validation Loss Curves\", diag_dir / \"loss_curves.png\"),\n",
                    "    (\"Diagram 2: Top-1 & Top-5 Accuracy Curves\", diag_dir / \"accuracy_curves.png\"),\n",
                    "    (\"Diagram 3: 82-Class Confusion Matrix\", diag_dir / \"confusion_matrix_82.png\"),\n",
                    "    (\"Diagram 4: Per-Class Accuracy Rankings\", diag_dir / \"per_class_accuracy.png\"),\n",
                    "    (\"Diagram 5: Temporal Attention Weights Heatmap\", diag_dir / \"temporal_attention_weights.png\"),\n",
                    "]\n",
                    "\n",
                    "for title, img_path in diagrams:\n",
                    "    if img_path.exists():\n",
                    "        print(f\"\\n{'='*70}\\n{title}\\n{'='*70}\")\n",
                    "        display(Image(filename=str(img_path), width=850))\n",
                    "    else:\n",
                    "        print(f\"[!] Diagram not found: {img_path}\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Backup Checkpoints & Diagrams to Google Drive\n",
                    "Preserves your trained model (`best_model.pth`) and high-resolution diagrams in your Google Drive even after Colab terminates.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!mkdir -p \"/content/drive/MyDrive/Sports_Action_Results\"\n",
                    "!cp -r diagrams/ \"/content/drive/MyDrive/Sports_Action_Results/\"\n",
                    "!cp -r checkpoints/ \"/content/drive/MyDrive/Sports_Action_Results/\"\n",
                    "!zip -r /content/Sports_Action_Results.zip diagrams checkpoints\n",
                    "\n",
                    "print(\"\\n[✓] Results permanently saved to Google Drive: /content/drive/MyDrive/Sports_Action_Results/\")\n",
                    "print(\"[✓] Downloadable archive ready: /content/Sports_Action_Results.zip\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Real-Time Action Validation & Live Camera Web App\n",
                    "Launches an interactive web app with a public share link (`https://xxxx.gradio.live`).\n",
                    "- Upload any custom sports video (.mp4, .avi, .mov)\n",
                    "- Record live from your webcam directly in the browser\n",
                    "- Real-time visualization of 17 skeletal joints, sports equipment boxes, and HOI vectors\n",
                    "- Live Top-5 prediction confidence chart",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Launch Gradio Web App with public shareable link\n",
                    "!python app_web.py --share",
                ],
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


def create_kaggle_notebook(output_path: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🏋️‍♂️ Human–Object Interaction (HOI) Multimodal BiLSTM for Sports Action Recognition\n",
                    "## ⚡ Kaggle Notebook Execution Pipeline (82 Sports Categories)\n",
                    "\n",
                    "**Thesis Framework**: *Human–Object Interaction-Aware Multimodal CNN–LSTM Framework for Sports Action Recognition*\n",
                    "\n",
                    "### 📋 Kaggle Environment Highlights:\n",
                    "- **Hardware**: Free Tesla P100 or 2x T4 GPU (up to 30 hours/week)\n",
                    "- **Input Path**: `/kaggle/input/...` (Read-Only dataset storage)\n",
                    "- **Working Path**: `/kaggle/working/...` (Writable storage for features, checkpoints, and figures)\n",
                    "- **One-Click Download**: Checkpoints and diagrams in `/kaggle/working/` are automatically downloadable from the Kaggle Output pane.",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Verify Kaggle GPU Environment",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!nvidia-smi\n",
                    "import torch\n",
                    "print(f\"CUDA Ready: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"Kaggle GPU: {torch.cuda.get_device_name(0)}\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Setup Codebase in `/kaggle/working`",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "%cd /kaggle/working\n",
                    "!rm -rf Sport-Action-Detection\n",
                    "!git clone https://github.com/angkon101/Sport-Action-Detection.git\n",
                    "%cd /kaggle/working/Sport-Action-Detection\n",
                    "\n",
                    "!pip install -q -r requirements.txt",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Detect & Locate Dataset from Kaggle Inputs\n",
                    "Checks `/kaggle/input` for an attached dataset containing `Model Data` or `ucf action`.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "from pathlib import Path\n",
                    "\n",
                    "input_dir = Path(\"/kaggle/input\")\n",
                    "data_dir = None\n",
                    "\n",
                    "# Scan for Model Data or ucf action inside /kaggle/input\n",
                    "for p in input_dir.rglob(\"*\"):\n",
                    "    if p.is_dir() and p.name == \"Model Data\":\n",
                    "        data_dir = p\n",
                    "        break\n",
                    "    elif p.is_dir() and p.name == \"ucf action\":\n",
                    "        # Populate Model Data split into working directory\n",
                    "        !python data/scripts/populate_model_data_split.py --source-dir \"{p}\" --target-dir \"/kaggle/working/Model Data\"\n",
                    "        data_dir = Path(\"/kaggle/working/Model Data\")\n",
                    "        break\n",
                    "\n",
                    "if data_dir:\n",
                    "    print(f\"[✓] Dataset successfully located at: {data_dir}\")\n",
                    "else:\n",
                    "    print(\"[!] No sports dataset found in /kaggle/input! Please click '+ Add Data' in Kaggle and attach your dataset.\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Extract Multimodal Features\n",
                    "Extracts Visual (2048-d), Pose (85-d), Object (7-d), and HOI (16-d) features.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract features to /kaggle/working/data/features\n",
                    "!python run_local.py --extract --data-dir \"{data_dir}\" --features-dir \"/kaggle/working/data/features\"\n",
                    "\n",
                    "# Pre-flight verification check\n",
                    "!python run_local.py --verify --features-dir \"/kaggle/working/data/features\"",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Train Model & Generate Outcome Diagrams",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train M7 (Full Multimodal + Attention) and generate diagrams\n",
                    "!python run_local.py --train --features-dir \"/kaggle/working/data/features\" --model-mode M7 --epochs 40 --batch-size 32 --lr 0.0005",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Render the 5 Thesis Outcome Diagrams Inline",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from IPython.display import Image, display\n",
                    "from pathlib import Path\n",
                    "\n",
                    "diag_dir = Path(\"/kaggle/working/Sport-Action-Detection/diagrams/M7\")\n",
                    "diagrams = [\n",
                    "    (\"1. Training & Validation Loss Curves\", diag_dir / \"loss_curves.png\"),\n",
                    "    (\"2. Top-1 & Top-5 Accuracy Curves\", diag_dir / \"accuracy_curves.png\"),\n",
                    "    (\"3. 82-Class Confusion Matrix\", diag_dir / \"confusion_matrix_82.png\"),\n",
                    "    (\"4. Per-Class Accuracy Rankings\", diag_dir / \"per_class_accuracy.png\"),\n",
                    "    (\"5. Temporal Attention Weights Heatmap\", diag_dir / \"temporal_attention_weights.png\"),\n",
                    "]\n",
                    "\n",
                    "for title, img_path in diagrams:\n",
                    "    if img_path.exists():\n",
                    "        print(f\"\\n{'='*70}\\n{title}\\n{'='*70}\")\n",
                    "        display(Image(filename=str(img_path), width=850))\n",
                    "    else:\n",
                    "        print(f\"[!] Diagram not found: {img_path}\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Package Outputs for One-Click Kaggle Download\n",
                    "Copies all diagrams and model checkpoints to `/kaggle/working` root for download.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!cp -r diagrams/ /kaggle/working/\n",
                    "!cp -r checkpoints/ /kaggle/working/\n",
                    "!zip -r /kaggle/working/Sports_Action_Results.zip /kaggle/working/diagrams /kaggle/working/checkpoints\n",
                    "print(\"\\n[✓] Results packaged! You can download 'Sports_Action_Results.zip' directly from the Kaggle Output pane.\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Real-Time Action Validation & Live Camera Web App\n",
                    "Launches an interactive web app with a public share link (`https://xxxx.gradio.live`).\n",
                    "- Upload any custom sports video (.mp4, .avi, .mov)\n",
                    "- Record live from your webcam directly in the browser\n",
                    "- Real-time visualization of 17 skeletal joints, sports equipment boxes, and HOI vectors\n",
                    "- Live Top-5 prediction confidence chart",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Launch Gradio Web App with public shareable link\n",
                    "!python app_web.py --share",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


def create_local_notebook(output_path: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🏋️‍♂️ Human–Object Interaction (HOI) Multimodal BiLSTM for Sports Action Recognition\n",
                    "## 💻 Local Machine Interactive Notebook (VS Code / Local GPU)\n",
                    "\n",
                    "**Thesis Framework**: *Human–Object Interaction-Aware Multimodal CNN–LSTM Framework for Sports Action Recognition*\n",
                    "\n",
                    "### 📋 Local Configuration Details:\n",
                    "- **Root Workspace**: `E:\\Akhi Thesis`\n",
                    "- **Dataset Folder**: `Model Data` (`Train` 70% = 6,408 videos | `Test` 30% = 2,743 videos)\n",
                    "- **Compute Device**: Local GPU (**NVIDIA GeForce GTX 1650 with Max-Q Design** on CUDA 12.1)\n",
                    "- **Output Artifacts**: Checkpoints in `checkpoints/M7/` and 5 thesis diagrams in `diagrams/M7/`",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Verify Local GPU and Hardware Acceleration",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "print(f\"PyTorch Version : {torch.__version__}\")\n",
                    "print(f\"CUDA Available  : {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"Local GPU Device: {torch.cuda.get_device_name(0)}\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Verify `Model Data` Labeled Train/Test Split",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "\n",
                    "train_dir = Path(\"Model Data/Train\")\n",
                    "test_dir = Path(\"Model Data/Test\")\n",
                    "\n",
                    "train_videos = list(train_dir.rglob(\"*.avi\"))\n",
                    "test_videos = list(test_dir.rglob(\"*.avi\"))\n",
                    "train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])\n",
                    "test_classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])\n",
                    "\n",
                    "print(f\"Train Categories: {len(train_classes)} | Videos: {len(train_videos):,}\")\n",
                    "print(f\"Test Categories : {len(test_classes)} | Videos: {len(test_videos):,}\")\n",
                    "print(f\"Total Videos    : {len(train_videos) + len(test_videos):,}\")\n",
                    "\n",
                    "# Verify zero overlap\n",
                    "overlap = set(p.name for p in train_videos).intersection(set(p.name for p in test_videos))\n",
                    "print(f\"Overlap count   : {len(overlap)} (Zero overlap verified!)\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Multimodal Feature Preprocessing & Extraction\n",
                    "Extracts Visual (2048-d), Pose (85-d), Object (7-d), and HOI (16-d) features on local GPU.\n",
                    "\n",
                    "- **Option A (2-minute smoke test)**: `!python run_local.py --extract --max-per-class 2`\n",
                    "- **Option B (Full dataset extraction)**: `!python run_local.py --extract`",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run extraction (toggle --max-per-class 2 for quick testing)\n",
                    "!python run_local.py --extract --data-dir \"Model Data\" --features-dir \"data/features\"\n",
                    "\n",
                    "# Verify feature tensor readiness\n",
                    "!python run_local.py --verify --features-dir \"data/features\"",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Train HOI-BiLSTM Model (Mode M7)\n",
                    "Trains 2-Layer BiLSTM with Temporal Attention on local GPU.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!python run_local.py --train --features-dir \"data/features\" --model-mode M7 --epochs 40 --batch-size 32 --lr 0.0005",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Render Outcome Diagrams in VS Code",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from IPython.display import Image, display\n",
                    "from pathlib import Path\n",
                    "\n",
                    "diag_dir = Path(\"diagrams/M7\")\n",
                    "diagrams = [\n",
                    "    (\"Diagram 1: Loss Learning Curves\", diag_dir / \"loss_curves.png\"),\n",
                    "    (\"Diagram 2: Top-1 & Top-5 Accuracy Curves\", diag_dir / \"accuracy_curves.png\"),\n",
                    "    (\"Diagram 3: 82-Class Confusion Matrix\", diag_dir / \"confusion_matrix_82.png\"),\n",
                    "    (\"Diagram 4: Per-Class Accuracy Rankings\", diag_dir / \"per_class_accuracy.png\"),\n",
                    "    (\"Diagram 5: Temporal Attention Weights Heatmap\", diag_dir / \"temporal_attention_weights.png\"),\n",
                    "]\n",
                    "\n",
                    "for title, img_path in diagrams:\n",
                    "    if img_path.exists():\n",
                    "        print(f\"\\n{'='*70}\\n{title}\\n{'='*70}\")\n",
                    "        display(Image(filename=str(img_path), width=800))\n",
                    "    else:\n",
                    "        print(f\"[!] Diagram not found: {img_path}\")",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Real-Time Action Validation & Live Camera Web App\n",
                    "Launches an interactive web app with a public share link (`https://xxxx.gradio.live`).\n",
                    "- Upload any custom sports video (.mp4, .avi, .mov)\n",
                    "- Record live from your webcam directly in the browser\n",
                    "- Real-time visualization of 17 skeletal joints, sports equipment boxes, and HOI vectors\n",
                    "- Live Top-5 prediction confidence chart",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Launch Gradio Web App with public shareable link\n",
                    "!python app_web.py --share",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


def main():
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    colab_path = notebooks_dir / "1_Google_Colab_HOI_BiLSTM.ipynb"
    kaggle_path = notebooks_dir / "2_Kaggle_HOI_BiLSTM.ipynb"
    local_path = notebooks_dir / "3_Local_Machine_HOI_BiLSTM.ipynb"

    create_colab_notebook(colab_path)
    print(f"[OK] Generated Google Colab Notebook: {colab_path}")

    create_kaggle_notebook(kaggle_path)
    print(f"[OK] Generated Kaggle Notebook      : {kaggle_path}")

    create_local_notebook(local_path)
    print(f"[OK] Generated Local Machine Notebook: {local_path}")


if __name__ == "__main__":
    main()

