"""Dedicated Local Machine Runner for Sports Action Recognition.

Specifically configured for local execution on Windows (VS Code / PowerShell)
using the labeled dataset at 'Model Data/Train' and 'Model Data/Test'.

Workflow:
1. Feature Extraction: Extracts 4-stream multimodal features into 'data/features/Train' and 'data/features/Test'
2. Data Verification: Pre-flight tensor integrity, dimension, and NaN checking
3. Model Training: Trains BiLSTM + Temporal Attention on GTX 1650 GPU (or CPU)
4. Diagram Evaluation: Generates 5 thesis publication-quality diagrams
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.hoi_dataset import HOIFeatureDataset
from src.extractors.hoi_feature_extractor import SportsFeatureExtractor
from src.models.hoi_bilstm import HOIBiLSTMClassifier
from src.utils.metrics import AverageMeter, accuracy
from src.utils.visualizer import (
    plot_learning_curves,
    plot_confusion_matrix_82,
    plot_per_class_accuracy,
    plot_temporal_attention_samples,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


# ==============================================================================
# 1. FEATURE EXTRACTION FOR MODEL DATA (TRAIN & TEST)
# ==============================================================================
def extract_split_features(
    data_dir: Path,
    output_dir: Path,
    num_frames: int = 16,
    max_per_class: Optional[int] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Initializing Extractor on device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    extractor = SportsFeatureExtractor(device=device, num_frames=num_frames)

    for split in ["Train", "Test"]:
        split_in = data_dir / split
        split_out = output_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        if not split_in.exists():
            print(f"[!] Warning: Folder '{split_in}' does not exist! Skipping.")
            continue

        class_dirs = sorted([d for d in split_in.iterdir() if d.is_dir() and not d.name.startswith(".")])
        print(f"\n=================================================================")
        print(f"Extracting [{split.upper()}]: {len(class_dirs)} categories -> '{split_out}'")
        print(f"=================================================================")

        total_extracted, total_cached = 0, 0
        t0 = time.time()

        for idx, cls_dir in enumerate(class_dirs, start=1):
            cls_out = split_out / cls_dir.name
            cls_out.mkdir(parents=True, exist_ok=True)

            videos = sorted(list(cls_dir.rglob("*.avi")))
            if max_per_class is not None:
                videos = videos[:max_per_class]

            for v_path in tqdm(videos, desc=f"[{idx}/{len(class_dirs)}] {cls_dir.name}", leave=False):
                save_path = cls_out / f"{v_path.stem}.pt"
                if save_path.exists():
                    total_cached += 1
                    continue
                try:
                    feats = extractor.process_video(v_path)
                    torch.save(feats, save_path)
                    total_extracted += 1
                except Exception as e:
                    print(f"  [!] Error processing {v_path.name}: {e}")

        elapsed = time.time() - t0
        print(f"[OK] [{split}] Extracted {total_extracted} videos ({total_cached} cached) in {elapsed:.1f}s.")


# ==============================================================================
# 2. PRE-FLIGHT VERIFICATION
# ==============================================================================
def verify_dataset_features(features_dir: Path) -> bool:
    print("\n" + "=" * 65)
    print("PRE-FLIGHT DATA & TENSOR INTEGRITY CHECK")
    print("=" * 65)

    train_dir = features_dir / "Train"
    test_dir = features_dir / "Test"

    if not train_dir.exists() or not test_dir.exists():
        print(f"[ERROR] Features folders not found at: {features_dir}")
        print("Please run extraction first: python run_local.py --extract")
        return False

    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    test_classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

    print(f"--> Found {len(train_classes)} categories in Train features.")
    print(f"--> Found {len(test_classes)} categories in Test features.")

    if len(train_classes) == 0:
        print("[ERROR] Train features directory is empty!")
        return False

    # Check sample files and tensor shapes
    expected_shapes = {
        "visual": (16, 2048),
        "pose": (16, 85),
        "object": (16, 7),
        "hoi": (16, 16),
    }

    train_files = list(train_dir.rglob("*.pt"))
    test_files = list(test_dir.rglob("*.pt"))

    print(f"--> Total .pt feature files in Train : {len(train_files):,}")
    print(f"--> Total .pt feature files in Test  : {len(test_files):,}")

    if len(train_files) == 0:
        print("[ERROR] No .pt files in Train features!")
        return False

    print("--> Checking tensor shapes and verifying no NaNs in sample tensors...")
    sample_files = train_files[:20] + test_files[:20]
    for p in sample_files:
        try:
            feats = torch.load(p, map_location="cpu")
            for key, shape in expected_shapes.items():
                if key not in feats:
                    print(f"[ERROR] Missing stream '{key}' in {p.name}")
                    return False
                if tuple(feats[key].shape) != shape:
                    print(f"[ERROR] Incorrect shape for '{key}' in {p.name}: got {feats[key].shape}, expected {shape}")
                    return False
                if torch.isnan(feats[key]).any() or torch.isinf(feats[key]).any():
                    print(f"[ERROR] NaN/Inf found in '{key}' in {p.name}")
                    return False
        except Exception as e:
            print(f"[ERROR] Corrupted file {p.name}: {e}")
            return False

    print("[SUCCESS] All sample feature tensors are valid, well-formed, and non-corrupt.")
    print("=" * 65)
    return True


# ==============================================================================
# 3. TRAINING & EVALUATION PIPELINE
# ==============================================================================
def run_local_training(
    features_dir: Path,
    model_mode: str = "M7",
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.0005,
    save_dir: Path = Path("checkpoints"),
    diagrams_dir: Path = Path("diagrams"),
) -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=================================================================")
    print(f"Training Model Mode: [{model_mode}]")
    print(f"Device             : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs             : {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f"=================================================================")

    train_feat_dir = features_dir / "Train"
    test_feat_dir = features_dir / "Test"

    # Load datasets
    train_dataset = HOIFeatureDataset(train_feat_dir)
    test_dataset = HOIFeatureDataset(test_feat_dir, classes=train_dataset.classes)

    num_classes = len(train_dataset.classes)
    print(f"--> Training Samples   : {len(train_dataset):,} across {num_classes} classes")
    print(f"--> Testing Samples    : {len(test_dataset):,} across {num_classes} classes")

    if len(train_dataset) == 0:
        print("[!] No training samples found. Please extract features first.")
        return

    # Windows-safe DataLoader with pinned memory for direct GPU DMA transfers
    num_workers = 0 if os.name == "nt" else 2
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # Ablation configuration
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
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ckpt_path = save_dir / model_mode
    ckpt_path.mkdir(parents=True, exist_ok=True)
    diag_path = diagrams_dir / model_mode
    diag_path.mkdir(parents=True, exist_ok=True)

    history = []
    best_acc = 0.0

    print("\n" + "-" * 75)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc (Top-1)':<15} | {'Time':<6}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = AverageMeter("Loss")
        train_acc = AverageMeter("Top-1")

        for batch, targets in train_loader:
            v = batch["visual"].to(device, non_blocking=True)
            p = batch["pose"].to(device, non_blocking=True)
            o = batch["object"].to(device, non_blocking=True)
            h = batch["hoi"].to(device, non_blocking=True)
            y = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits, _ = model(visual=v, pose=p, obj=o, hoi=h)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            acc1, _ = accuracy(logits, y, topk=(1, 5))
            acc1_val = acc1.item() if hasattr(acc1, "item") else float(acc1)
            train_loss.update(loss.item(), y.size(0))
            train_acc.update(acc1_val, y.size(0))

        scheduler.step()

        # Fast Validation on test split
        model.eval()
        val_loss = AverageMeter("Val Loss")
        val_acc1 = AverageMeter("Val Top-1")
        val_acc5 = AverageMeter("Val Top-5")

        with torch.inference_mode():
            for batch, targets in test_loader:
                v = batch["visual"].to(device, non_blocking=True)
                p = batch["pose"].to(device, non_blocking=True)
                o = batch["object"].to(device, non_blocking=True)
                h = batch["hoi"].to(device, non_blocking=True)
                y = targets.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits, _ = model(visual=v, pose=p, obj=o, hoi=h)
                    loss = criterion(logits, y)
                acc1, acc5 = accuracy(logits, y, topk=(1, 5))
                acc1_val = acc1.item() if hasattr(acc1, "item") else float(acc1)
                acc5_val = acc5.item() if hasattr(acc5, "item") else float(acc5)

                val_loss.update(loss.item(), y.size(0))
                val_acc1.update(acc1_val, y.size(0))
                val_acc5.update(acc5_val, y.size(0))

        elapsed = time.time() - t0
        print(f"{epoch:<6} | {train_loss.avg:<10.4f} | {train_acc.avg:<10.2f}% | {val_loss.avg:<10.4f} | {val_acc1.avg:<15.2f}% | {elapsed:<5.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss.avg,
            "train_top1": train_acc.avg,
            "train_acc1": train_acc.avg,
            "val_loss": val_loss.avg,
            "val_top1": val_acc1.avg,
            "val_acc1": val_acc1.avg,
            "val_top5": val_acc5.avg,
        })

        if val_acc1.avg > best_acc:
            best_acc = val_acc1.avg
            ckpt_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "classes": train_dataset.classes,
                "best_acc": best_acc,
                "model_mode": model_mode,
            }
            torch.save(ckpt_dict, ckpt_path / "best_model.pth")
            torch.save(ckpt_dict, ckpt_path / "best_model.pt")

    print("-" * 75)
    print(f"[OK] Training Complete! Best Validation Accuracy: {best_acc:.2f}%")
    print(f"     Checkpoint saved to: {ckpt_path / 'best_model.pth'}")

    # ==============================================================================
    # 4. GENERATING 5 THESIS OUTCOME DIAGRAMS
    # ==============================================================================
    print("\n" + "=" * 65)
    print("GENERATING 5 PUBLICATION-QUALITY THESIS DIAGRAMS")
    print("=" * 65)

    # Load best checkpoint for diagram generation
    try:
        best_ckpt = torch.load(ckpt_path / "best_model.pth", map_location=device, weights_only=False)
    except TypeError:
        best_ckpt = torch.load(ckpt_path / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    # 1 & 2. Loss & Accuracy Learning Curves
    plot_learning_curves(history, diag_path)

    all_preds, all_targets = [], []
    sample_attentions = []

    with torch.no_grad():
        for batch, targets in test_loader:
            v = batch["visual"].to(device)
            p = batch["pose"].to(device)
            o = batch["object"].to(device)
            h = batch["hoi"].to(device)
            logits, alpha = model(visual=v, pose=p, obj=o, hoi=h)

            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_targets.extend(targets.tolist())

            if alpha is not None and len(sample_attentions) < 5:
                for b_idx in range(min(5 - len(sample_attentions), targets.size(0))):
                    cls_name = train_dataset.classes[targets[b_idx].item()]
                    sample_attentions.append((cls_name, alpha[b_idx].cpu().numpy()))

    # 3. 82-Class Confusion Matrix
    plot_confusion_matrix_82(all_targets, all_preds, train_dataset.classes, diag_path)

    # 4. Per-Class Accuracy Rankings
    plot_per_class_accuracy(all_targets, all_preds, train_dataset.classes, diag_path)

    # 5. Temporal Attention Weights Heatmap
    if sample_attentions:
        plot_temporal_attention_samples(sample_attentions, diag_path)

    # 6. Full Classification Metrics JSON Report
    report = classification_report(all_targets, all_preds, target_names=train_dataset.classes, output_dict=True, zero_division=0)
    with open(diag_path / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[✓] Diagram 1: Loss Learning Curves       -> {diag_path / 'loss_curves.png'}")
    print(f"[✓] Diagram 2: Accuracy Curves           -> {diag_path / 'accuracy_curves.png'}")
    print(f"[✓] Diagram 3: 82-Class Confusion Matrix -> {diag_path / 'confusion_matrix_82.png'}")
    print(f"[✓] Diagram 4: Per-Class Accuracy Bar    -> {diag_path / 'per_class_accuracy.png'}")
    if sample_attentions:
        print(f"[✓] Diagram 5: Temporal Attention Maps   -> {diag_path / 'temporal_attention_weights.png'}")
    print(f"[✓] Report   : Full Metrics JSON         -> {diag_path / 'evaluation_report.json'}")
    print("=" * 65)


def run_local_evaluation(
    features_dir: Path,
    model_mode: str = "M7",
    checkpoint: Optional[str] = None,
    batch_size: int = 64,
) -> None:
    """Evaluates a trained model checkpoint on the test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print(f"Evaluating Model Mode: [{model_mode}]")
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 65)

    test_feat_dir = features_dir / "Test"
    if not test_feat_dir.exists():
        print(f"[!] Test features directory not found at: {test_feat_dir}")
        return

    test_dataset = HOIFeatureDataset(test_feat_dir, preload=True)
    if len(test_dataset) == 0:
        print("[!] No test samples found.")
        return

    num_classes = len(test_dataset.classes)
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

    # Resolve checkpoint
    ckpt_path = Path(checkpoint) if checkpoint else (PROJECT_ROOT / "checkpoints" / model_mode / "best_model.pth")
    if not ckpt_path.exists():
        alt_path = ckpt_path.with_suffix(".pt") if ckpt_path.suffix == ".pth" else ckpt_path.with_suffix(".pth")
        if alt_path.exists():
            ckpt_path = alt_path

    if not ckpt_path.exists():
        print(f"[!] Checkpoint not found at: {ckpt_path}")
        return

    print(f"--> Loading weights from: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    remapped = {}
    for k, v in state.items():
        nk = k.replace("fusion.", "fusion_layer.").replace("attention.", "temporal_pooling.")
        remapped[nk] = v
    model.load_state_dict(remapped)
    model.eval()

    num_workers = 0 if os.name == "nt" else 2
    pin_mem = (device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    criterion = nn.CrossEntropyLoss()

    val_loss = AverageMeter("Test Loss")
    val_acc1 = AverageMeter("Top-1")
    val_acc5 = AverageMeter("Top-5")
    all_preds, all_targets = [], []

    print("\n--> Running Fast Evaluation on Test Split...")
    with torch.inference_mode():
        for batch, targets in tqdm(test_loader, desc="Evaluating"):
            v = batch["visual"].to(device, non_blocking=True)
            p = batch["pose"].to(device, non_blocking=True)
            o = batch["object"].to(device, non_blocking=True)
            h = batch["hoi"].to(device, non_blocking=True)
            y = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits, _ = model(visual=v, pose=p, obj=o, hoi=h)
                loss = criterion(logits, y)

            acc1, acc5 = accuracy(logits, y, topk=(1, 5))
            acc1_val = acc1.item() if hasattr(acc1, "item") else float(acc1)
            acc5_val = acc5.item() if hasattr(acc5, "item") else float(acc5)

            val_loss.update(loss.item(), y.size(0))
            val_acc1.update(acc1_val, y.size(0))
            val_acc5.update(acc5_val, y.size(0))

            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_targets.extend(targets.tolist())

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(f"Test Loss        : {val_loss.avg:.4f}")
    print(f"Top-1 Accuracy   : {val_acc1.avg:.2f}%")
    print(f"Top-5 Accuracy   : {val_acc5.avg:.2f}%")
    print(f"Total Evaluated  : {len(all_targets):,} samples across {num_classes} classes")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Sports Action Recognition Local Runner")
    parser.add_argument("--extract", action="store_true", help="Extract multimodal features from Model Data")
    parser.add_argument("--verify", action="store_true", help="Verify features tensor readiness")
    parser.add_argument("--train", action="store_true", help="Train model & generate thesis diagrams")
    parser.add_argument("--test", "--eval", dest="test", action="store_true", help="Evaluate trained model on test split")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: extract -> verify -> train")
    parser.add_argument("--data-dir", type=str, default="Model Data", help="Path to Model Data folder")
    parser.add_argument("--features-dir", type=str, default="data/features", help="Path to cache .pt features")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for testing/evaluation")
    parser.add_argument("--model-mode", type=str, default="M7", choices=["M1", "M2", "M3", "M4", "M5", "M6", "M7"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional max videos per class for smoke testing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    feat_dir = Path(args.features_dir)

    if args.all:
        args.extract = True
        args.verify = True
        args.train = True

    if args.extract:
        extract_split_features(data_dir, feat_dir, max_per_class=args.max_per_class)

    if args.verify:
        ready = verify_dataset_features(feat_dir)
        if not ready and args.train:
            print("[!] Halting training due to verification failure.")
            return

    if args.train:
        run_local_training(
            features_dir=feat_dir,
            model_mode=args.model_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    if args.test:
        run_local_evaluation(
            features_dir=feat_dir,
            model_mode=args.model_mode,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
        )

    if not (args.extract or args.verify or args.train or args.test):
        print("Please specify a command:")
        print("  python run_local.py --extract               # Extract all features from Model Data")
        print("  python run_local.py --extract --max-per-class 2  # Quick 2-video test extraction")
        print("  python run_local.py --verify                # Verify feature tensors")
        print("  python run_local.py --train                 # Train M7 & generate diagrams")
        print("  python run_local.py --test                  # Evaluate model on test split")
        print("  python run_local.py --all                   # Complete pipeline")


if __name__ == "__main__":
    main()

