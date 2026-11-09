"""Training script for Human-Object Interaction Multimodal BiLSTM (HOI-BiLSTM)."""

import argparse
import json
from pathlib import Path
import random
import sys
import time
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.hoi_dataset import HOIFeatureDataset
from src.models.hoi_bilstm import HOIBiLSTMClassifier
from src.utils.metrics import AverageMeter, accuracy


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    loss_meter = AverageMeter("Train Loss")
    top1_meter = AverageMeter("Top-1")
    top5_meter = AverageMeter("Top-5")

    for batch, targets in loader:
        visual = batch["visual"].to(device)
        pose = batch["pose"].to(device)
        obj = batch["object"].to(device)
        hoi = batch["hoi"].to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(visual=visual, pose=pose, obj=obj, hoi=hoi)
        loss = criterion(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
        loss_meter.update(loss.item(), targets.size(0))
        top1_meter.update(prec1, targets.size(0))
        top5_meter.update(prec5, targets.size(0))

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    loss_meter = AverageMeter("Val Loss")
    top1_meter = AverageMeter("Val Top-1")
    top5_meter = AverageMeter("Val Top-5")

    all_preds = []
    all_targets = []

    for batch, targets in loader:
        visual = batch["visual"].to(device)
        pose = batch["pose"].to(device)
        obj = batch["object"].to(device)
        hoi = batch["hoi"].to(device)
        targets = targets.to(device)

        logits, _ = model(visual=visual, pose=pose, obj=obj, hoi=hoi)
        loss = criterion(logits, targets)

        prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
        loss_meter.update(loss.item(), targets.size(0))
        top1_meter.update(prec1, targets.size(0))
        top5_meter.update(prec5, targets.size(0))

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().tolist())

    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return loss_meter.avg, top1_meter.avg, top5_meter.avg, macro_f1 * 100.0


def main():
    parser = argparse.ArgumentParser(description="Train HOI-BiLSTM Sports Action Classifier")
    parser.add_argument("--features-dir", type=str, default="data/features", help="Path to pre-extracted .pt features")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--model-mode", type=str, default="M7", choices=["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
                        help="Ablation model mode from thesis plan (M1=Visual only, M7=Full Proposed HOI-BiLSTM)")
    parser.add_argument("--save-dir", type=str, default="checkpoints/hoi_bilstm", help="Directory to save checkpoints")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using compute device: {device}")
    if device.type == "cuda":
        print(f"--> GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Dataset
    print(f"--> Loading feature dataset from '{args.features_dir}'...")
    full_dataset = HOIFeatureDataset(features_dir=args.features_dir)
    num_classes = len(full_dataset.classes)
    total_samples = len(full_dataset)
    print(f"--> Total samples: {total_samples} across {num_classes} sport classes")

    if total_samples == 0:
        print("[!] No extracted features found. Please run 'python src/extractors/batch_extract_features.py' first.")
        return

    # Train / Val Split (80% train, 20% val)
    indices = list(range(total_samples))
    rng = random.Random(42)
    rng.shuffle(indices)

    val_size = max(1, int(total_samples * 0.2))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"--> Split: {len(train_set)} train samples | {len(val_set)} val samples")

    # 2. Configure Ablation Mode (M1 through M7)
    modes = {
        "M1": dict(use_visual=True,  use_pose=False, use_object=False, use_hoi=False, use_attention=False),
        "M2": dict(use_visual=False, use_pose=True,  use_object=False, use_hoi=False, use_attention=False),
        "M3": dict(use_visual=False, use_pose=False, use_object=True,  use_hoi=False, use_attention=False),
        "M4": dict(use_visual=True,  use_pose=True,  use_object=False, use_hoi=False, use_attention=False),
        "M5": dict(use_visual=True,  use_pose=True,  use_object=True,  use_hoi=False, use_attention=False),
        "M6": dict(use_visual=True,  use_pose=True,  use_object=True,  use_hoi=True,  use_attention=False),
        "M7": dict(use_visual=True,  use_pose=True,  use_object=True,  use_hoi=True,  use_attention=True),
    }

    config = modes[args.model_mode]
    print(f"\n--> Configuring Model [{args.model_mode}]: {config}")

    model = HOIBiLSTMClassifier(
        num_classes=num_classes,
        bilstm_hidden=256,
        bilstm_layers=2,
        dropout=0.3,
        **config,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir) / args.model_mode
    save_dir.mkdir(parents=True, exist_ok=True)

    best_top1 = 0.0
    history = []

    print("\n" + "=" * 65)
    print(f"STARTING TRAINING [{args.model_mode}] FOR {args.epochs} EPOCHS")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_t1, tr_t5 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_t1, val_t5, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] "
            f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Top-1: {val_t1:.2f}% | Val Top-5: {val_t5:.2f}% | Macro-F1: {val_f1:.2f}%"
        )

        record = {
            "epoch": epoch,
            "train_loss": tr_loss, "train_top1": tr_t1,
            "val_loss": val_loss, "val_top1": val_t1, "val_top5": val_t5, "val_macro_f1": val_f1
        }
        history.append(record)

        if val_t1 > best_top1:
            best_top1 = val_t1
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "classes": full_dataset.classes,
                "best_top1": best_top1,
                "model_mode": args.model_mode,
            }, save_dir / "best_model.pth")
            print(f" [★] New Best Val Top-1: {best_top1:.2f}% -> Saved best_model.pth")

    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete for [{args.model_mode}]. Peak Val Top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()
