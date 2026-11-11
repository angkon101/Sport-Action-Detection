"""All-in-One Standalone Pipeline for Sports Action Recognition.

Complete Master Workflow:
1. Stratified 70% Train / 30% Test Dataset Splitting per category (--split)
2. Multimodal Feature Preprocessing & Extraction (--extract)
3. Pre-Flight Data Availability & Integrity Verification (--verify)
4. Model Training on 70% Train Split (--train)
5. Comprehensive Test Evaluation on 30% Test Split with 5 Thesis Diagrams (--evaluate)

Can be run locally or directly in Google Colab without GitHub!
"""

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import f1_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.models.detection as detection
import torchvision.transforms as transforms
from tqdm import tqdm

from src.datasets.data_splitter import create_stratified_split, verify_data_readiness
from src.utils.visualizer import (
    plot_learning_curves,
    plot_confusion_matrix_82,
    plot_per_class_accuracy,
    plot_temporal_attention_samples,
)


# ==============================================================================
# 1. UTILITY FUNCTIONS & METRICS
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class AverageMeter:
    """Computes and stores average and current value."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def compute_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k > output.size(1):
                res.append(0.0)
                continue
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# ==============================================================================
# 2. FEATURE EXTRACTOR (CNN + KEYPOINTS + OBJECTS + HOI)
# ==============================================================================
SPORTS_OBJECT_COCO_IDS = {
    2: "bicycle", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 34: "frisbee", 35: "skis", 36: "snowboard"
}


class SportsFeatureExtractor:
    """Extracts multimodal visual, pose, object, and HOI feature sequences."""

    def __init__(self, device: torch.device, num_frames: int = 16):
        self.device = device
        self.num_frames = num_frames

        print(f"--> Initializing ResNet-50, Keypoint R-CNN, and Faster R-CNN on {device}...")

        # 1. Visual CNN (ResNet-50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.visual_backbone = torch.nn.Sequential(*(list(resnet.children())[:-1])).to(self.device)
        self.visual_backbone.eval()

        self.visual_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Keypoint R-CNN (17 Joints)
        self.keypoint_model = detection.keypointrcnn_resnet50_fpn(
            weights=detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(self.device)
        self.keypoint_model.eval()

        # 3. Faster R-CNN (Sports Objects)
        self.object_model = detection.fasterrcnn_resnet50_fpn(
            weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(self.device)
        self.object_model.eval()

    def sample_video_frames(self, video_path: str | Path) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0 or total_frames < self.num_frames:
            raw = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                raw.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release()
            if len(raw) == 0:
                raise ValueError(f"Video {video_path} has 0 readable frames.")
            indices = np.linspace(0, len(raw) - 1, self.num_frames, dtype=int)
            return [raw[i] for i in indices]

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()
        return frames

    @torch.no_grad()
    def process_video(self, video_path: str | Path) -> Dict[str, torch.Tensor]:
        frames = self.sample_video_frames(video_path)
        t_frames = len(frames)

        # 1. Visual Stream [T, 2048]
        tensors = torch.stack([self.visual_transform(f) for f in frames]).to(self.device)
        visual_feats = self.visual_backbone(tensors).flatten(1).cpu()

        # 2. Pose Stream [T, 85]
        det_inputs = [transforms.functional.to_tensor(f).to(self.device) for f in frames]
        kp_preds = self.keypoint_model(det_inputs)
        raw_kpts = np.zeros((t_frames, 17, 3), dtype=np.float32)

        for t, pred in enumerate(kp_preds):
            kpts = pred["keypoints"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            if len(scores) > 0:
                raw_kpts[t] = kpts[np.argmax(scores)]

        norm_kpts = np.zeros_like(raw_kpts)
        for t in range(t_frames):
            h, w = frames[t].shape[:2]
            norm_kpts[t, :, 0] = raw_kpts[t, :, 0] / max(w, 1)
            norm_kpts[t, :, 1] = raw_kpts[t, :, 1] / max(h, 1)
            norm_kpts[t, :, 2] = raw_kpts[t, :, 2]

        velocities = np.zeros((t_frames, 17, 2), dtype=np.float32)
        for t in range(1, t_frames):
            velocities[t] = norm_kpts[t, :, :2] - norm_kpts[t - 1, :, :2]
        pose_feats = np.concatenate([norm_kpts.reshape(t_frames, -1), velocities.reshape(t_frames, -1)], axis=1)

        # 3. Object Stream [T, 7]
        obj_preds = self.object_model(det_inputs)
        obj_feats = np.zeros((t_frames, 7), dtype=np.float32)
        raw_boxes = np.zeros((t_frames, 4), dtype=np.float32)

        for t, pred in enumerate(obj_preds):
            h, w = frames[t].shape[:2]
            boxes = pred["boxes"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()

            best_idx = None
            best_score = 0.25
            for idx, (label, score) in enumerate(zip(labels, scores)):
                if label in SPORTS_OBJECT_COCO_IDS and score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                b = boxes[best_idx]
                raw_boxes[t] = b
                cx = ((b[0] + b[2]) / 2.0) / max(w, 1)
                cy = ((b[1] + b[3]) / 2.0) / max(h, 1)
                bw = (b[2] - b[0]) / max(w, 1)
                bh = (b[3] - b[1]) / max(h, 1)
                obj_feats[t] = [cx, cy, bw, bh, 0.0, 0.0, 1.0]

        for t in range(1, t_frames):
            if obj_feats[t, 6] == 1.0 and obj_feats[t - 1, 6] == 1.0:
                obj_feats[t, 4] = obj_feats[t, 0] - obj_feats[t - 1, 0]
                obj_feats[t, 5] = obj_feats[t, 1] - obj_feats[t - 1, 1]

        # 4. HOI Geometric Stream [T, 16]
        hoi_feats = np.zeros((t_frames, 16), dtype=np.float32)
        key_joints = [9, 10, 15, 16]  # L/R Wrist, L/R Ankle
        prev_dists = np.zeros(len(key_joints), dtype=np.float32)

        for t in range(t_frames):
            h, w = frames[t].shape[:2]
            if obj_feats[t, 6] == 0.0:
                prev_dists.fill(0.0)
                continue

            p_obj = np.array([obj_feats[t, 0], obj_feats[t, 1]], dtype=np.float32)
            cur_dists = []
            slot = 0
            for j_i, j_id in enumerate(key_joints):
                p_joint = np.array([raw_kpts[t, j_id, 0] / max(w, 1), raw_kpts[t, j_id, 1] / max(h, 1)], dtype=np.float32)
                dist = float(np.linalg.norm(p_joint - p_obj))
                cur_dists.append(dist)
                diff = p_obj - p_joint
                u_vec = diff / (np.linalg.norm(diff) + 1e-6)
                delta_d = dist - prev_dists[j_i] if t > 0 else 0.0

                hoi_feats[t, slot:slot+4] = [dist, float(u_vec[0]), float(u_vec[1]), delta_d]
                slot += 4
            prev_dists = np.array(cur_dists, dtype=np.float32)

        return {
            "visual": visual_feats,
            "pose": torch.from_numpy(pose_feats).float(),
            "object": torch.from_numpy(obj_feats).float(),
            "hoi": torch.from_numpy(hoi_feats).float(),
        }


# ==============================================================================
# 3. SPLIT-AWARE DATASET CLASS
# ==============================================================================
class SplitAwareHOIDataset(Dataset):
    """Loads features corresponding to a specific split (70% train or 30% test)."""

    def __init__(self, features_dir: Path, split_file: Path, class_mapping_file: Path):
        self.features_dir = features_dir
        with open(class_mapping_file, "r", encoding="utf-8") as f:
            self.class_to_idx = json.load(f)
        self.classes = sorted(list(self.class_to_idx.keys()))

        with open(split_file, "r", encoding="utf-8") as f:
            self.split_data = json.load(f)

        self.samples: List[Tuple[Path, int]] = []
        for cls_name, rel_paths in self.split_data.items():
            if cls_name not in self.class_to_idx:
                continue
            lbl = self.class_to_idx[cls_name]
            for rel_p in rel_paths:
                p = Path(rel_p)
                # Feature file path is: features_dir / cls_name / f"{stem}.pt"
                feat_path = self.features_dir / cls_name / f"{p.stem}.pt"
                if feat_path.exists():
                    self.samples.append((feat_path, lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = torch.load(path, map_location="cpu")
        return {
            "visual": data["visual"].float(),
            "pose": data["pose"].float(),
            "object": data["object"].float(),
            "hoi": data["hoi"].float(),
            "stem": path.stem,
        }, label


# ==============================================================================
# 4. HOI-BiLSTM MODEL WITH ATTENTION POOLING
# ==============================================================================
class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = self.projection(h).squeeze(-1)  # [B, T]
        alpha = F.softmax(energy, dim=-1)        # [B, T]
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # [B, D]
        return context, alpha


class HOIBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 82,
        visual_in_dim: int = 2048,
        pose_in_dim: int = 85,
        object_in_dim: int = 7,
        hoi_in_dim: int = 16,
        bilstm_hidden: int = 256,
        bilstm_layers: int = 2,
        dropout: float = 0.3,
        use_visual: bool = True,
        use_pose: bool = True,
        use_object: bool = True,
        use_hoi: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_visual = use_visual
        self.use_pose = use_pose
        self.use_object = use_object
        self.use_hoi = use_hoi
        self.use_attention = use_attention

        fusion_dim = 0
        if use_visual:
            self.visual_branch = nn.Sequential(
                nn.Linear(visual_in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout)
            )
            fusion_dim += 256
        if use_pose:
            self.pose_branch = nn.Sequential(
                nn.Linear(pose_in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout)
            )
            fusion_dim += 128
        if use_object:
            self.object_branch = nn.Sequential(
                nn.Linear(object_in_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout)
            )
            fusion_dim += 64
        if use_hoi:
            self.hoi_branch = nn.Sequential(
                nn.Linear(hoi_in_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout)
            )
            fusion_dim += 64

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout)
        )

        self.bilstm = nn.LSTM(
            input_size=512, hidden_size=bilstm_hidden, num_layers=bilstm_layers,
            bidirectional=True, batch_first=True, dropout=dropout if bilstm_layers > 1 else 0.0
        )
        total_hidden = bilstm_hidden * 2

        if use_attention:
            self.attention = TemporalAttentionPooling(total_hidden, attention_dim=128)

        self.classifier = nn.Sequential(
            nn.Linear(total_hidden, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, visual=None, pose=None, obj=None, hoi=None):
        tokens = []
        if self.use_visual: tokens.append(self.visual_branch(visual))
        if self.use_pose: tokens.append(self.pose_branch(pose))
        if self.use_object: tokens.append(self.object_branch(obj))
        if self.use_hoi: tokens.append(self.hoi_branch(hoi))

        fused = torch.cat(tokens, dim=-1)
        z = self.fusion(fused)

        out, (h_n, _) = self.bilstm(z)

        if self.use_attention:
            context, alpha = self.attention(out)
        else:
            context = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            alpha = None

        logits = self.classifier(context)
        return logits, alpha


# ==============================================================================
# 5. PIPELINE EXECUTION FUNCTIONS
# ==============================================================================
def run_feature_extraction(data_dir: Path, output_dir: Path, num_frames: int = 16, max_per_class: Optional[int] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SportsFeatureExtractor(device=device, num_frames=num_frames)

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"\n--> Extracting features from {len(class_dirs)} categories into '{output_dir}'...")

    total_extracted, total_cached = 0, 0
    t0 = time.time()

    for idx, cls_folder in enumerate(class_dirs, 1):
        cls_out = output_dir / cls_folder.name
        cls_out.mkdir(parents=True, exist_ok=True)

        videos = []
        for ext in ["*.avi", "*.mp4", "*.mkv", "*.mov"]:
            videos.extend(list(cls_folder.rglob(ext)))

        if max_per_class is not None:
            videos = videos[:max_per_class]

        for v_path in tqdm(videos, desc=f"[{idx}/{len(class_dirs)}] {cls_folder.name}", leave=False):
            save_path = cls_out / f"{v_path.stem}.pt"
            if save_path.exists():
                total_cached += 1
                continue
            try:
                feats = extractor.process_video(v_path)
                torch.save(feats, save_path)
                total_extracted += 1
            except Exception as e:
                print(f"[!] Error on {v_path.name}: {e}")

    print(f"\nDone! Extracted {total_extracted} videos ({total_cached} cached) in {time.time()-t0:.1f}s.")


def run_training_and_evaluation(
    features_dir: Path,
    splits_dir: Path,
    model_mode: str = "M7",
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.0005,
    save_dir: str = "checkpoints",
    diagrams_dir: str = "diagrams",
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--> Training Mode [{model_mode}] on Device: {device}")

    train_split_file = splits_dir / "train_split_70.json"
    test_split_file = splits_dir / "test_split_30.json"
    class_map_file = splits_dir / "class_indices_82.json"

    if not train_split_file.exists() or not test_split_file.exists():
        raise FileNotFoundError("Split files not found! Please run with --split first.")

    train_dataset = SplitAwareHOIDataset(features_dir, train_split_file, class_map_file)
    test_dataset = SplitAwareHOIDataset(features_dir, test_split_file, class_map_file)

    num_classes = len(train_dataset.classes)
    print(f"--> Train Split (70%): {len(train_dataset)} samples")
    print(f"--> Test Split (30%) : {len(test_dataset)} samples across {num_classes} classes")

    num_workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    modes = {
        "M1": dict(use_visual=True,  use_pose=False, use_object=False, use_hoi=False, use_attention=False),
        "M2": dict(use_visual=False, use_pose=True,  use_object=False, use_hoi=False, use_attention=False),
        "M3": dict(use_visual=False, use_pose=False, use_object=True,  use_hoi=False, use_attention=False),
        "M4": dict(use_visual=True,  use_pose=True,  use_object=False, use_hoi=False, use_attention=False),
        "M5": dict(use_visual=True,  use_pose=True,  use_object=True,  use_hoi=False, use_attention=False),
        "M6": dict(use_visual=True,  use_pose=True,  use_object=True,  use_hoi=True,  use_attention=False),
        "M7": dict(use_visual=True,  use_pose=True,  use_object=True,  use_hoi=True,  use_attention=True),
    }

    model = HOIBiLSTMClassifier(num_classes=num_classes, **modes[model_mode]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_ckpt_dir = Path(save_dir) / model_mode
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history = []

    print("=" * 65)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_m, t1_m, t5_m = AverageMeter(), AverageMeter(), AverageMeter()
        t0 = time.time()

        for b, y in train_loader:
            v, p, o, h, y = b["visual"].to(device), b["pose"].to(device), b["object"].to(device), b["hoi"].to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(visual=v, pose=p, obj=o, hoi=h)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            p1, p5 = compute_accuracy(logits, y, (1, 5))
            loss_m.update(loss.item(), y.size(0))
            t1_m.update(p1, y.size(0))
            t5_m.update(p5, y.size(0))

        # Validation on Test Split
        model.eval()
        v_loss, v_t1, v_t5 = AverageMeter(), AverageMeter(), AverageMeter()
        preds, tgts = [], []
        with torch.no_grad():
            for b, y in test_loader:
                v, p, o, h, y = b["visual"].to(device), b["pose"].to(device), b["object"].to(device), b["hoi"].to(device), y.to(device)
                logits, _ = model(visual=v, pose=p, obj=o, hoi=h)
                l = criterion(logits, y)
                p1, p5 = compute_accuracy(logits, y, (1, 5))
                v_loss.update(l.item(), y.size(0))
                v_t1.update(p1, y.size(0))
                v_t5.update(p5, y.size(0))
                preds.extend(logits.argmax(-1).cpu().tolist())
                tgts.extend(y.cpu().tolist())

        scheduler.step()
        f1 = f1_score(tgts, preds, average="macro", zero_division=0) * 100.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{time.time()-t0:.1f}s] "
            f"Train Loss: {loss_m.avg:.4f} | Test Loss: {v_loss.avg:.4f} | "
            f"Test Top-1: {v_t1.avg:.2f}% | Test Top-5: {v_t5.avg:.2f}% | Macro-F1: {f1:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "train_loss": loss_m.avg, "train_top1": t1_m.avg,
            "val_loss": v_loss.avg, "val_top1": v_t1.avg, "val_top5": v_t5.avg, "val_f1": f1
        })

        if v_t1.avg > best_acc:
            best_acc = v_t1.avg
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "classes": train_dataset.classes,
                "model_mode": model_mode,
            }, out_ckpt_dir / "best_model.pth")
            print(f" [★] New Best Test Top-1: {best_acc:.2f}% (Saved to {out_ckpt_dir}/best_model.pth)")

    # ==========================================================================
    # 6. POST-TRAINING EVALUATION & DIAGRAM GENERATION
    # ==========================================================================
    print("\n" + "=" * 65)
    print("GENERATING THESIS EVALUATION DIAGRAMS & FINAL METRICS")
    print("=" * 65)
    diag_path = Path(diagrams_dir) / model_mode
    diag_path.mkdir(parents=True, exist_ok=True)

    # 1. Loss & Accuracy curves
    plot_learning_curves(history, diag_path)

    # 2. Final Evaluation with Best Model
    best_checkpoint = torch.load(out_ckpt_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["state_dict"])
    model.eval()

    all_preds, all_tgts = [], []
    sample_attns = []

    with torch.no_grad():
        for b, y in test_loader:
            v, p, o, h = b["visual"].to(device), b["pose"].to(device), b["object"].to(device), b["hoi"].to(device)
            logits, alpha = model(visual=v, pose=p, obj=o, hoi=h)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_tgts.extend(y.tolist())

            if alpha is not None and len(sample_attns) < 5:
                for idx_in_batch in range(min(5 - len(sample_attns), y.size(0))):
                    cls_name = train_dataset.classes[y[idx_in_batch].item()]
                    sample_attns.append((cls_name, alpha[idx_in_batch].cpu().numpy()))

    # 3. Confusion Matrix
    plot_confusion_matrix_82(all_tgts, all_preds, train_dataset.classes, diag_path)

    # 4. Per-Class Accuracy Rankings
    plot_per_class_accuracy(all_tgts, all_preds, train_dataset.classes, diag_path)

    # 5. Temporal Attention Weights
    if sample_attns:
        plot_temporal_attention_samples(sample_attns, diag_path)

    # 6. Classification Report JSON
    report = classification_report(all_tgts, all_preds, target_names=train_dataset.classes, output_dict=True, zero_division=0)
    with open(diag_path / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[✓] Loss & Accuracy Learning Curves: {diag_path / 'loss_curves.png'}, {diag_path / 'accuracy_curves.png'}")
    print(f"[✓] Normalized Confusion Matrix    : {diag_path / 'confusion_matrix_82.png'}")
    print(f"[✓] Per-Class Accuracy Chart       : {diag_path / 'per_class_accuracy.png'}")
    if sample_attns:
        print(f"[✓] Temporal Attention Weights Plot: {diag_path / 'temporal_attention_weights.png'}")
    print(f"[✓] Complete Evaluation JSON Report : {diag_path / 'evaluation_report.json'}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Master HOI-BiLSTM Sports Action Recognition Pipeline")
    parser.add_argument("--split", action="store_true", help="Generate stratified 70/30 train/test split per category")
    parser.add_argument("--extract", action="store_true", help="Run multimodal feature extraction")
    parser.add_argument("--verify", action="store_true", help="Run pre-flight data verification")
    parser.add_argument("--train", action="store_true", help="Run model training and diagram evaluation")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: split -> extract -> verify -> train -> evaluate")
    parser.add_argument("--data-dir", type=str, default="ucf action", help="Path to raw video dataset")
    parser.add_argument("--features-dir", type=str, default="data/features", help="Path to cached .pt features")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Directory for split JSONs")
    parser.add_argument("--model-mode", type=str, default="M7", choices=["M1", "M2", "M3", "M4", "M5", "M6", "M7"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional max videos per class for smoke testing")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    feat_path = Path(args.features_dir)
    splits_path = Path(args.splits_dir)

    if args.all:
        args.split = True
        args.extract = True
        args.verify = True
        args.train = True

    if args.split:
        create_stratified_split(data_path, splits_path, train_ratio=0.7)

    if args.extract:
        run_feature_extraction(data_path, feat_path, max_per_class=args.max_per_class)

    if args.verify:
        is_ready = verify_data_readiness(data_path, feat_path)
        if not is_ready and args.train:
            print("[!] Halting training because data verification failed.")
            return

    if args.train:
        run_training_and_evaluation(
            features_dir=feat_path,
            splits_dir=splits_path,
            model_mode=args.model_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    if not (args.split or args.extract or args.verify or args.train):
        print("Please specify a pipeline stage:")
        print("  python run_all.py --split")
        print("  python run_all.py --extract")
        print("  python run_all.py --verify")
        print("  python run_all.py --train")
        print("  python run_all.py --all")


if __name__ == "__main__":
    main()
