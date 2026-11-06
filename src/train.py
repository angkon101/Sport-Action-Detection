"""Training Pipeline for Sports Action Recognition."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Optional, Tuple
import yaml

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.sports_classes import UCF101_SPORTS_CLASSES
from src.datasets.ucf_sports_dataset import UCFSportsActionDataset
from src.models.action_classifier import build_model
from src.utils.metrics import AverageMeter, accuracy


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, float, float]:
    model.train()
    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top-1")
    top5_meter = AverageMeter("Top-5")

    optimizer.zero_grad()
    num_batches = len(dataloader)

    for i, (videos, targets) in enumerate(dataloader):
        videos = videos.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(videos)
                loss = criterion(outputs, targets) / gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(videos)
            loss = criterion(outputs, targets) / gradient_accumulation_steps
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == num_batches:
                optimizer.step()
                optimizer.zero_grad()

        # Metrics tracking
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item() * gradient_accumulation_steps, videos.size(0))
        top1_meter.update(prec1, videos.size(0))
        top5_meter.update(prec5, videos.size(0))

        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(
                f"Epoch [{epoch}] Batch [{i+1}/{num_batches}] "
                f"Loss: {loss_meter.avg:.4f} | Top-1: {top1_meter.avg:.2f}% | Top-5: {top5_meter.avg:.2f}%"
            )

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    loss_meter = AverageMeter("Val Loss")
    top1_meter = AverageMeter("Val Top-1")
    top5_meter = AverageMeter("Val Top-5")

    for videos, targets in dataloader:
        videos = videos.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(videos)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item(), videos.size(0))
        top1_meter.update(prec1, videos.size(0))
        top5_meter.update(prec5, videos.size(0))

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def main():
    parser = argparse.ArgumentParser(description="Train Sports Action Recognition Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone architecture")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset root path")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Overrides
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.data_dir:
        cfg["dataset"]["data_dir"] = args.data_dir

    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using compute device: {device}")
    if device.type == "cuda":
        print(f"--> GPU: {torch.cuda.get_device_name(0)}")

    # Target classes
    target_classes = sorted(UCF101_SPORTS_CLASSES)
    num_classes = len(target_classes)

    print(f"--> Initializing datasets with {num_classes} sports classes...")
    data_dir = cfg["dataset"]["data_dir"]
    train_split = cfg["dataset"].get("train_split", "data/splits/trainlist01.txt")
    val_split = cfg["dataset"].get("val_split", "data/splits/testlist01.txt")

    train_dataset = UCFSportsActionDataset(
        data_dir=data_dir,
        split_file=train_split if Path(train_split).exists() else None,
        num_frames=cfg["dataset"].get("num_frames", 16),
        frame_interval=cfg["dataset"].get("frame_interval", 2),
        is_training=True,
        target_classes=target_classes,
    )

    val_dataset = UCFSportsActionDataset(
        data_dir=data_dir,
        split_file=val_split if Path(val_split).exists() else None,
        num_frames=cfg["dataset"].get("num_frames", 16),
        frame_interval=cfg["dataset"].get("frame_interval", 2),
        is_training=False,
        target_classes=target_classes,
    )

    print(f"--> Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    if len(train_dataset) == 0:
        print("[!] Warning: No training video samples found.")
        print(f"    Please place videos under '{data_dir}/<SportClass>/' or supply split files in 'data/splits/'.")
        return

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Build model
    print(f"--> Building {cfg['model']['backbone']} with Kinetics-400 pretrained weights...")
    model = build_model(
        num_classes=num_classes,
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.5),
        freeze_backbone=cfg["model"].get("freeze_backbone", False),
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = cfg["training"].get("mixed_precision", True) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)

    checkpoint_dir = Path(cfg["checkpoint"].get("save_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_top1 = 0.0
    history = []

    print(f"\n{'='*60}\nStarting Training for {epochs} Epochs\n{'='*60}")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_t1, train_t5 = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            gradient_accumulation_steps=grad_accum,
        )

        val_loss, val_t1, val_t5 = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Train Top-1: {train_t1:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Top-1: {val_t1:.2f}%, Val Top-5: {val_t5:.2f}%"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_t1,
            "train_top5": train_t5,
            "val_loss": val_loss,
            "val_top1": val_t1,
            "val_top5": val_t5,
        }
        history.append(record)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_top1": best_top1,
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "classes": target_classes,
        }
        torch.save(checkpoint, checkpoint_dir / "latest_checkpoint.pth")

        if val_t1 > best_top1:
            best_top1 = val_t1
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print(f" [★] New best Top-1 validation accuracy: {best_top1:.2f}% (Saved best_model.pth)")

    # Save training history
    with open(checkpoint_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best Validation Top-1 Accuracy: {best_top1:.2f}%")


if __name__ == "__main__":
    main()
