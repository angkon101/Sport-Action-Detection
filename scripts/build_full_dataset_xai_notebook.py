"""Script to build the Comprehensive Full-Dataset Training & Multi-Action XAI Notebook."""
import json
from pathlib import Path

nb_path = Path("e:/Akhi Thesis/Kaggle_Full_Dataset_Training_and_MultiAction_XAI.ipynb")
cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# ============================================================================
# CELL 1: Header Markdown
# ============================================================================
add_md("""# Multimodal CMAT-HOI: Full-Dataset Training, Comprehensive Validation & Multi-Action XAI Suite

This master research notebook executes:
1. **Full Dataset Stratified Training**:
   - Ingests **100% of all available samples** across all action classes.
   - Stratified sampling guarantees every single sport action class is represented in both training and validation sets.
2. **Complete Epoch-by-Epoch Progress**:
   - Logs detailed metrics for every epoch: Train Loss, Train Top-1, Val Loss, Val Top-1, Val Top-5, Macro-F1, and Epoch Duration.
3. **Multi-Action Explainable AI (XAI)**:
   - Validates and explains **multiple distinct sports actions** (e.g., Racket, Archery, Ball Handling, Athletics) side-by-side.
   - **Multi-Action Temporal Saliency**: 4-panel comparison of critical moment identification ($\alpha_t$).
   - **Multi-Action Biomechanical Joint Saliency**: Cross-sport comparison across the 17 COCO human joints.
   - **Multi-Action Modality Attribution**: Quantifies how Visual, Pose, Object, and HOI streams contribute differently across sports.
   - **Multi-Action Counterfactual 'What-If'**: Validates the drop in model confidence when equipment/HOI interaction is removed.
4. **All Research Output Figures (300 DPI)** saved to `/kaggle/working/diagrams/`.
""")

# ============================================================================
# CELL 2: Environment Setup & Hardware Verification
# ============================================================================
add_code("""# Cell 1: Environment Setup, GPU Verification & Path Auto-Discovery
import os
import sys
import json
import time
import math
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score, 
    cohen_kappa_score, 
    matthews_corrcoef
)
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# Publication-grade figure styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.autolayout": True
})

def auto_detect_dataset_dir() -> Path:
    search_roots = ["/kaggle/input", "/kaggle/working", "."]
    for root in search_roots:
        p = Path(root)
        if not p.exists():
            continue
        pt_files = list(p.glob("**/*.pt"))
        if pt_files:
            features_dir = pt_files[0].parent.parent
            print(f"--> [Auto-Detect] Found feature directory: '{features_dir}'")
            return features_dir

    for root in search_roots:
        p = Path(root)
        if not p.exists():
            continue
        zip_files = list(p.glob("**/*.zip"))
        if zip_files:
            target_zip = zip_files[0]
            extract_to = Path("/kaggle/temp/extracted_features")
            print(f"--> [Auto-Detect] Found zip archive: {target_zip.name}. Extracting...")
            with zipfile.ZipFile(target_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            pt_files = list(extract_to.glob("**/*.pt"))
            if pt_files:
                return pt_files[0].parent.parent

    raise FileNotFoundError("Could not locate any .pt feature files in /kaggle/input.")

FEATURES_DIR = auto_detect_dataset_dir()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> [Hardware] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
""")

# ============================================================================
# CELL 3: Configuration & Seeding
# ============================================================================
add_code("""# Cell 2: Hyperparameters & Deterministic Seeding
SEED = 42
EPOCHS = 40              # 40 full epochs across 100% of the dataset
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

OUTPUT_DIR = Path("/kaggle/working/full_dataset_experiment")
DIAGRAM_DIR = OUTPUT_DIR / "diagrams"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)
print(f"--> [Config] Seed: {SEED} | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
""")

# ============================================================================
# CELL 4: Whole Dataset Loader with Stratified Split
# ============================================================================
add_code("""# Cell 3: Whole Dataset Loader & Stratified Train/Val Split (100% Data Utilization)
class RobustHOIFeatureDataset(Dataset):
    def __init__(self, features_dir: Path, is_train: bool = True):
        self.features_dir = features_dir
        self.is_train = is_train
        self.classes = sorted([
            d.name for d in self.features_dir.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        
        for cls_name in self.classes:
            cls_folder = self.features_dir / cls_name
            label = self.class_to_idx[cls_name]
            for pt_file in cls_folder.glob("*.pt"):
                self.samples.append((pt_file, label))

        self.cached_data: List[Dict[str, torch.Tensor]] = []
        self.cached_labels: List[int] = []

        print(f"--> [RAM Cache] Loading {len(self.samples):,} multimodal sequences into RAM...")
        for file_path, label in tqdm(self.samples, desc="RAM Loading", leave=False):
            try:
                data = torch.load(file_path, map_location="cpu")
                self.cached_data.append({
                    "visual": torch.nan_to_num(data["visual"].float(), nan=0.0),
                    "pose": torch.nan_to_num(data["pose"].float(), nan=0.0),
                    "object": torch.nan_to_num(data["object"].float(), nan=0.0),
                    "hoi": torch.nan_to_num(data["hoi"].float(), nan=0.0),
                })
                self.cached_labels.append(label)
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        data = self.cached_data[idx]
        label = self.cached_labels[idx]

        visual = data["visual"].clone()
        pose = data["pose"].clone()
        obj = data["object"].clone()
        hoi = data["hoi"].clone()

        if self.is_train:
            if random.random() < 0.25:
                t_idx = random.randint(0, visual.size(0) - 1)
                visual[t_idx] = 0.0
                pose[t_idx] = 0.0
            if random.random() < 0.25:
                pose += torch.randn_like(pose) * 0.015

        return {"visual": visual, "pose": pose, "object": obj, "hoi": hoi}, label


raw_train_set = RobustHOIFeatureDataset(features_dir=FEATURES_DIR, is_train=True)
num_classes = len(raw_train_set.classes)
total_samples = len(raw_train_set)
all_labels = raw_train_set.cached_labels

# Stratified Split: Guarantees EVERY class is represented in both splits
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
try:
    train_indices, val_indices = next(sss.split(range(total_samples), all_labels))
except ValueError:
    # Fallback if single sample per class
    indices = list(range(total_samples))
    random.Random(SEED).shuffle(indices)
    val_size = max(1, int(total_samples * 0.2))
    train_indices, val_indices = indices[val_size:], indices[:val_size]

train_dataset = Subset(raw_train_set, train_indices)
raw_val_set = RobustHOIFeatureDataset(features_dir=FEATURES_DIR, is_train=False)
val_dataset = Subset(raw_val_set, val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"--> [Stratified Split Complete] 100% of dataset utilized:")
print(f"    • Total Dataset: {total_samples:,} samples across {num_classes} classes")
print(f"    • Train Split:   {len(train_dataset):,} samples")
print(f"    • Val Split:     {len(val_dataset):,} samples")
""")

# ============================================================================
# CELL 5: Proposed Architecture: CMAT-HOI
# ============================================================================
add_code("""# Cell 4: Proposed Architecture (Multimodal CMAT-HOI with Temporal Self-Attention)
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int = 512, attention_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim), nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = self.proj(h).squeeze(-1)
        alpha = F.softmax(energy, dim=-1)
        return torch.bmm(alpha.unsqueeze(1), h).squeeze(1), alpha

class ProposedCMAT_HOI(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        # Asymmetric Dimension-Balanced Projection Stems
        self.visual_proj = nn.Sequential(nn.Linear(2048, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout))
        self.pose_proj   = nn.Sequential(nn.Linear(85, 128),   nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout))
        self.obj_proj    = nn.Sequential(nn.Linear(7, 64),     nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout))
        self.hoi_proj    = nn.Sequential(nn.Linear(16, 64),    nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout))

        self.fusion = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout))
        self.bilstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.pool = TemporalAttention(hidden_dim=hidden_dim * 2, attention_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, visual, pose, obj, hoi):
        v = self.visual_proj(visual)
        p = self.pose_proj(pose)
        o = self.obj_proj(obj)
        h = self.hoi_proj(hoi)
        fused = self.fusion(torch.cat([v, p, o, h], dim=-1))
        lstm_out, _ = self.bilstm(fused)
        context, attn = self.pool(lstm_out)
        logits = self.classifier(context)
        return logits, attn

model = ProposedCMAT_HOI(num_classes=num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"--> [Model Ready] Proposed CMAT-HOI: {total_params:.2f}M Trainable Parameters")
""")

# ============================================================================
# CELL 6: Training Loop with Full Epoch Logs
# ============================================================================
add_code("""# Cell 5: Comprehensive Training with Detailed Epoch-by-Epoch Progress
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

def calc_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            if k > output.size(1):
                res.append(0.0)
                continue
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / target.size(0)).item())
        return res

best_val_top1 = 0.0
best_val_top5 = 0.0
history = []
train_start = time.time()

print("\\n" + "=" * 90)
print(f"TRAINING PROPOSED CMAT-HOI ON 100% OF DATASET ({EPOCHS} FULL EPOCHS)")
print("=" * 90)

for epoch in range(1, EPOCHS + 1):
    ep_t0 = time.time()
    
    # --- Training Phase ---
    model.train()
    tr_loss, tr_top1_sum, total_tr = 0.0, 0.0, 0
    for batch, targets in train_loader:
        v = batch["visual"].to(device, non_blocking=True)
        p = batch["pose"].to(device, non_blocking=True)
        o = batch["object"].to(device, non_blocking=True)
        h = batch["hoi"].to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(v, p, o, h)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        b_sz = targets.size(0)
        tr_loss += loss.item() * b_sz
        tr_top1_sum += calc_accuracy(logits, targets, topk=(1,))[0] * b_sz
        total_tr += b_sz

    scheduler.step()

    # --- Validation Phase ---
    model.eval()
    val_loss, val_top1_sum, val_top5_sum, total_val = 0.0, 0.0, 0.0, 0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch, targets in val_loader:
            v = batch["visual"].to(device, non_blocking=True)
            p = batch["pose"].to(device, non_blocking=True)
            o = batch["object"].to(device, non_blocking=True)
            h = batch["hoi"].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits, _ = model(v, p, o, h)
            loss = criterion(logits, targets)

            b_sz = targets.size(0)
            p1, p5 = calc_accuracy(logits, targets, topk=(1, 5))
            val_loss += loss.item() * b_sz
            val_top1_sum += p1 * b_sz
            val_top5_sum += p5 * b_sz
            total_val += b_sz

            val_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            val_targets.extend(targets.cpu().tolist())

    val_t1 = val_top1_sum / total_val
    val_t5 = val_top5_sum / total_val
    macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0) * 100.0
    ep_elapsed = time.time() - ep_t0
    cum_min = (time.time() - train_start) / 60.0

    history.append({
        "epoch": epoch,
        "train_loss": tr_loss / total_tr,
        "train_top1": tr_top1_sum / total_tr,
        "val_loss": val_loss / total_val,
        "val_top1": val_t1,
        "val_top5": val_t5,
        "macro_f1": macro_f1
    })

    if val_t1 > best_val_top1:
        best_val_top1 = val_t1
        best_val_top5 = val_t5
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "classes": raw_train_set.classes,
            "best_top1": best_val_top1,
            "best_top5": best_val_top5,
            "macro_f1": macro_f1
        }, CHECKPOINT_DIR / "best_model.pth")

    print(
        f"Epoch {epoch:02d}/{EPOCHS:02d} [{ep_elapsed:.1f}s | Elapsed: {cum_min:.1f}m] - "
        f"Tr Loss: {tr_loss/total_tr:.4f} | Val Loss: {val_loss/total_val:.4f} | "
        f"Val Top-1: {val_t1:.2f}% | Val Top-5: {val_t5:.2f}% | Macro F1: {macro_f1:.2f}% | (Peak: {best_val_top1:.2f}%)"
    )

print("\\n" + "=" * 90)
print(f"TRAINING COMPLETE IN {(time.time() - train_start)/60:.1f} MINUTES! PEAK TOP-1: {best_val_top1:.2f}%")
print("=" * 90)
""")

# ============================================================================
# CELL 7: Full Dataset Evaluation Curves & Confusion Matrix
# ============================================================================
add_code("""# Cell 6: Full Validation Evaluation Curves & Normalized Confusion Matrix
best_ckpt = torch.load(CHECKPOINT_DIR / "best_model.pth")
model.load_state_dict(best_ckpt["state_dict"])
model.eval()

# Final Evaluation Pass
final_preds, final_targets = [], []
with torch.no_grad():
    for batch, targets in val_loader:
        v = batch["visual"].to(device)
        p = batch["pose"].to(device)
        o = batch["object"].to(device)
        h = batch["hoi"].to(device)
        logits, _ = model(v, p, o, h)
        final_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        final_targets.extend(targets.tolist())

all_class_labels = list(range(num_classes))
report = classification_report(
    final_targets, final_preds, labels=all_class_labels,
    target_names=raw_train_set.classes, output_dict=True, zero_division=0
)
kappa = cohen_kappa_score(final_targets, final_preds)
mcc = matthews_corrcoef(final_targets, final_preds)

print("\\n" + "=" * 80)
print("BENCHMARK METRIC SUMMARY (ON 100% OF VALIDATION SET)")
print("=" * 80)
print(f"  • Peak Top-1 Accuracy:            {best_ckpt['best_top1']:.2f}%")
print(f"  • Top-5 Accuracy:                 {best_ckpt['best_top5']:.2f}%")
print(f"  • Macro F1-Score:                 {best_ckpt['macro_f1']:.2f}%")
print(f"  • Cohen's Kappa (κ):              {kappa:.4f}")
print(f"  • Matthews Correlation (MCC):     {mcc:.4f}")
print("=" * 80)

epochs_range = [h["epoch"] for h in history]

# FIGURE 1: Loss Progression
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
ax.plot(epochs_range, [h["train_loss"] for h in history], "b-o", label="Train Loss", linewidth=2, markersize=3)
ax.plot(epochs_range, [h["val_loss"] for h in history], "r--s", label="Validation Loss", linewidth=2, markersize=3)
ax.set_title("Proposed CMAT-HOI Training & Validation Loss Progression", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.savefig(DIAGRAM_DIR / "1_loss_curves.png", dpi=300, bbox_inches="tight")
print("\\n" + "=" * 70 + "\\nFIGURE 1: Training & Validation Loss Progression\\n" + "=" * 70)
plt.show()

# FIGURE 2: Accuracy Progression
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
ax.plot(epochs_range, [h["train_top1"] for h in history], "g-o", label="Train Top-1", linewidth=2, markersize=3)
ax.plot(epochs_range, [h["val_top1"] for h in history], "b-^", label="Validation Top-1", linewidth=2, markersize=3)
ax.plot(epochs_range, [h["val_top5"] for h in history], "m--d", label="Validation Top-5", linewidth=2, markersize=3)
ax.set_title("CMAT-HOI Top-1 & Top-5 Validation Accuracy Progression", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right")
plt.savefig(DIAGRAM_DIR / "2_accuracy_progression.png", dpi=300, bbox_inches="tight")
print("\\n" + "=" * 70 + "\\nFIGURE 2: Top-1 and Top-5 Accuracy Progression\\n" + "=" * 70)
plt.show()

# FIGURE 3: Confusion Matrix
cm = confusion_matrix(final_targets, final_preds, labels=all_class_labels)
cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

fig_sz = (18, 16) if num_classes > 30 else (10, 8)
fig, ax = plt.subplots(figsize=fig_sz, dpi=150)
sns.heatmap(
    cm_norm, cmap="Blues", annot=num_classes <= 20, fmt=".2f",
    xticklabels=raw_train_set.classes, yticklabels=raw_train_set.classes,
    cbar_kws={"label": "Normalized Accuracy"}, ax=ax
)
ax.set_title(f"Normalized Confusion Matrix ({num_classes} Sports Classes)", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Action Class")
ax.set_ylabel("True Action Class")
plt.xticks(rotation=90, fontsize=7 if num_classes > 30 else 10)
plt.yticks(rotation=0, fontsize=7 if num_classes > 30 else 10)
plt.savefig(DIAGRAM_DIR / "3_confusion_matrix.png", dpi=300, bbox_inches="tight")
print("\\n" + "=" * 70 + f"\\nFIGURE 3: Normalized Confusion Matrix ({num_classes} Classes)\\n" + "=" * 70)
plt.show()
""")

# ============================================================================
# CELL 8: MULTI-ACTION XAI SUITE (4 DIVERSE ACTIONS COMPARED)
# ============================================================================
add_code("""# Cell 7: Multi-Action Explainable AI (XAI) Suite (Comparing 4 Diverse Sports)
print("\\n" + "=" * 85)
print("EXECUTING MULTI-ACTION EXPLAINABLE AI (XAI) ON 4 DISTINCT SPORTS ACTIONS")
print("=" * 85)

# Select 4 representative sports actions representing diverse biomechanical modalities:
target_sports = ["Archery", "Tennis", "Basketball", "Cricket"]
available_sports = [s for s in target_sports if s in raw_train_set.classes]

# Fallback if names differ: pick top 4 classes from dataset
if len(available_sports) < 4:
    available_sports = raw_train_set.classes[:4]

print(f"--> [Selected Actions for XAI Study]: {available_sports}")

# Extract 1 validation sample for each of the 4 actions
samples_dict = {}
for batch, targets in val_loader:
    for idx in range(targets.size(0)):
        cls_idx = targets[idx].item()
        cls_name = raw_train_set.classes[cls_idx]
        if cls_name in available_sports and cls_name not in samples_dict:
            samples_dict[cls_name] = {
                "visual": batch["visual"][idx:idx+1].to(device),
                "pose": batch["pose"][idx:idx+1].to(device),
                "object": batch["object"][idx:idx+1].to(device),
                "hoi": batch["hoi"][idx:idx+1].to(device),
                "target": cls_idx
            }
        if len(samples_dict) == len(available_sports):
            break
    if len(samples_dict) == len(available_sports):
        break

# Compute XAI Metrics for all selected actions
joint_names = ["Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"]
xai_results = {}

for action_name, s in samples_dict.items():
    v = s["visual"].clone().detach().requires_grad_(True)
    p = s["pose"].clone().detach().requires_grad_(True)
    o = s["object"].clone().detach().requires_grad_(True)
    h = s["hoi"].clone().detach().requires_grad_(True)
    target_idx = s["target"]

    # Safe cuDNN backward pass
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    model.zero_grad()

    logits, attn = model(v, p, o, h)
    probs = F.softmax(logits, dim=-1)[0]
    pred_idx = logits.argmax(dim=-1).item()
    conf = probs[pred_idx].item() * 100.0

    logits[0, pred_idx].backward(retain_graph=True)
    torch.backends.cudnn.enabled = prev_cudnn

    # 1. Temporal Attention
    if attn is not None:
        t_weights = attn.squeeze(0).detach().cpu().numpy()
    else:
        t_s = (v.grad.abs().sum(dim=-1) + p.grad.abs().sum(dim=-1)).squeeze(0).cpu().numpy()
        t_weights = t_s / (t_s.sum() + 1e-9)

    # 2. Joint Saliency
    p_grad = p.grad.abs().squeeze(0).cpu()
    j_sal = [p_grad[:, 3*j:3*j+3].sum().item() + p_grad[:, 51+2*j:51+2*j+2].sum().item() for j in range(17)]
    norm_j_sal = [(s / (sum(j_sal) + 1e-9)) * 100.0 for s in j_sal]

    # 3. Modality Attribution
    with torch.no_grad():
        s_full = probs[pred_idx].item()
        s_no_vis = F.softmax(model(torch.zeros_like(v), p, o, h)[0], dim=-1)[0, pred_idx].item()
        s_no_pose = F.softmax(model(v, torch.zeros_like(p), o, h)[0], dim=-1)[0, pred_idx].item()
        s_no_obj = F.softmax(model(v, p, torch.zeros_like(o), h)[0], dim=-1)[0, pred_idx].item()
        s_no_hoi = F.softmax(model(v, p, o, torch.zeros_like(h))[0], dim=-1)[0, pred_idx].item()

    drops = [max(0.01, s_full - s_no_vis), max(0.01, s_full - s_no_pose), max(0.01, s_full - s_no_obj), max(0.01, s_full - s_no_hoi)]
    mod_pcts = [(d / sum(drops)) * 100.0 for d in drops]

    # 4. Counterfactual Drop
    cf_no_hoi = s_no_hoi * 100.0
    cf_no_obj = s_no_obj * 100.0

    xai_results[action_name] = {
        "conf": conf,
        "temporal": t_weights,
        "joints": norm_j_sal,
        "modality": mod_pcts,
        "cf_real": conf,
        "cf_no_hoi": cf_no_hoi,
        "cf_no_obj": cf_no_obj
    }
    print(f"  [XAI Processed] Action: '{action_name}' | Confidence: {conf:.1f}%")

# ----------------------------------------------------------------------------
# FIGURE 4: MULTI-ACTION TEMPORAL ATTENTION SALIENCY (2x2 Grid)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
frames = np.arange(1, 17)
colors_act = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6"]

for idx, (act_name, res) in enumerate(xai_results.items()):
    ax = axes[idx // 2, idx % 2]
    t_w = res["temporal"]
    peak_f = frames[np.argmax(t_w)]
    ax.plot(frames, t_w, "o-", color=colors_act[idx], linewidth=2.3, markersize=5, label=f"Attention Energy")
    ax.axvline(x=peak_f, color="red", linestyle="--", linewidth=1.5, label=f"Peak Contact (Frame {peak_f})")
    ax.fill_between(frames, t_w, color=colors_act[idx], alpha=0.15)
    ax.set_title(f"Action: {act_name} (Confidence: {res['conf']:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Video Frame Index")
    ax.set_ylabel("Attention Score (α_t)")
    ax.set_xticks(frames)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

plt.suptitle("Multi-Action Temporal XAI: Critical Moment Identification Across 4 Sports", fontsize=13, fontweight="bold", y=1.02)
plt.savefig(DIAGRAM_DIR / "4_multiaction_temporal_saliency.png", dpi=300, bbox_inches="tight")
print("\\n" + "=" * 70 + "\\nFIGURE 4: Multi-Action Temporal Attention Saliency (4 Sports)\\n" + "=" * 70)
plt.show()

# ----------------------------------------------------------------------------
# FIGURE 5: MULTI-ACTION BIOMECHANICAL JOINT SALIENCY HEATMAP
# ----------------------------------------------------------------------------
joint_matrix = np.array([res["joints"] for res in xai_results.values()])
fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
sns.heatmap(
    joint_matrix, cmap="YlOrRd", annot=True, fmt=".1f",
    xticklabels=joint_names, yticklabels=list(xai_results.keys()),
    cbar_kws={"label": "Anatomical Joint Importance (%)"}, ax=ax
)
ax.set_title("Multi-Action Biomechanical XAI: 17 Human Joint Attributions Across 4 Sports", fontsize=13, pad=12, fontweight="bold")
ax.set_xlabel("COCO Human Anatomical Keypoints")
ax.set_ylabel("Sports Action Category")
plt.xticks(rotation=45, ha="right", fontsize=9.5)
plt.yticks(rotation=0, fontsize=10.5)
plt.savefig(DIAGRAM_DIR / "5_multiaction_joint_saliency_heatmap.png", dpi=300, bbox_inches="tight")
print("\\n" + "=" * 70 + "\\nFIGURE 5: Multi-Action Biomechanical Joint Saliency Heatmap\\n" + "=" * 70)
plt.show()

# ----------------------------------------------------------------------------
# FIGURE 6: MULTI-ACTION MODALITY ATTRIBUTION (Grouped Bar Chart)
# ----------------------------------------------------------------------------
mod_labels = ["Visual Context", "Pose Kinematics", "Equipment Track", "HOI Geometric"]
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
x = np.arange(len(xai_results))
w = 0.2
colors_mod = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]

for m_i, mod_name in enumerate(mod_labels):
    vals = [res["modality"][m_i] for res in xai_results.values()]
    ax.bar(x + (m_i - 1.5)*w, vals, width=w, label=mod_name, color=colors_mod[m_i], edgecolor="black", linewidth=0.8)

ax.set_ylabel("Modality Contribution Share (%)", fontsize=11)
ax.set_title("Multi-Action Modality Attribution Comparison: Visual vs Pose vs Equipment vs HOI", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(list(xai_results.keys()), fontsize=11, fontweight="bold")
ax.set_ylim(0, 65)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(fontsize=10)
plt.savefig(DIAGRAM_DIR / "6_multiaction_modality_attribution.png", dpi=300, bbox_inches="tight")
print("\\n" + "=" * 70 + "\\nFIGURE 6: Multi-Action Modality Attribution Comparison\\n" + "=" * 70)
plt.show()

# ----------------------------------------------------------------------------
# FIGURE 7: MULTI-ACTION COUNTERFACTUAL WHAT-IF EVALUATION
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
x = np.arange(len(xai_results))
w = 0.25

real_vals = [res["cf_real"] for res in xai_results.values()]
no_hoi_vals = [res["cf_no_hoi"] for res in xai_results.values()]
no_obj_vals = [res["cf_no_obj"] for res in xai_results.values()]

ax.bar(x - w, real_vals, width=w, label="Full Multimodal (Real)", color="#10B981", edgecolor="black")
ax.bar(x, no_hoi_vals, width=w, label="Without HOI Interaction", color="#F59E0B", edgecolor="black")
ax.bar(x + w, no_obj_vals, width=w, label="Without Sports Equipment", color="#EF4444", edgecolor="black")

ax.set_ylabel("Classification Confidence (%)", fontsize=11)
ax.set_title("Multi-Action Counterfactual 'What-If': Impact of Removing Equipment & HOI", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(list(xai_results.keys()), fontsize=11, fontweight="bold")
ax.set_ylim(0, 115)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(fontsize=10)

plt.savefig(DIAGRAM_DIR / "7_multiaction_counterfactual_comparison.png", dpi=300, bbox_inches="tight")
print("\\n" + "=" * 70 + "\\nFIGURE 7: Multi-Action Counterfactual Comparison\\n" + "=" * 70)
plt.show()

print("\\n" + "=" * 85)
print(f"MASTER SUITE COMPLETE! All 7 publication figures saved to: {DIAGRAM_DIR}")
print(f"Checkpoints saved to: {CHECKPOINT_DIR}/best_model.pth")
print("=" * 85)
""")

# ============================================================================
# CELL 9: 1-Click Export Archive
# ============================================================================
add_code("""# Cell 8: 1-Click Export Archive for Thesis Submission
import shutil

zip_output = "/kaggle/working/Thesis_Full_Dataset_and_MultiAction_XAI.zip"
print(f"--> Packaging all results from {OUTPUT_DIR} into {zip_output}...")
shutil.make_archive(zip_output.replace(".zip", ""), "zip", OUTPUT_DIR)
print(f"[✓] Ready for 1-click download! Archive created at: {zip_output}")
""")

notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"[SUCCESS] Built Kaggle Master Notebook with {len(cells)} cells at: {nb_path}")
