"""Evaluation and Metrics Generation Pipeline for Sports Action Recognition."""

import argparse
import json
from pathlib import Path
import sys
import yaml

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.sports_classes import UCF101_SPORTS_CLASSES
from src.datasets.ucf_sports_dataset import UCFSportsActionDataset
from src.models.action_classifier import build_model
from src.utils.metrics import (
    AverageMeter,
    accuracy,
    compute_classification_metrics,
    plot_and_save_confusion_matrix,
)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    output_dir: Path,
) -> dict:
    model.eval()
    loss_meter = AverageMeter("Test Loss")
    top1_meter = AverageMeter("Top-1")
    top5_meter = AverageMeter("Top-5")

    all_preds: list[int] = []
    all_targets: list[int] = []

    for videos, targets in dataloader:
        videos = videos.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(videos)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item(), videos.size(0))
        top1_meter.update(prec1, videos.size(0))
        top5_meter.update(prec5, videos.size(0))

        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().tolist())

    output_dir.mkdir(parents=True, exist_ok=True)

    # Classification report
    report = compute_classification_metrics(all_targets, all_preds, class_names)
    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(all_targets, all_preds, class_names, cm_path)

    results = {
        "loss": loss_meter.avg,
        "top1_acc": top1_meter.avg,
        "top5_acc": top5_meter.avg,
        "num_evaluated_samples": len(all_targets),
        "confusion_matrix_path": str(cm_path),
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Top-1 Accuracy: {top1_meter.avg:.2f}%")
    print(f"Overall Top-5 Accuracy: {top5_meter.avg:.2f}%")
    print(f"Average Loss:           {loss_meter.avg:.4f}")
    print(f"Report saved to:        {output_dir / 'classification_report.json'}")
    print(f"Confusion Matrix saved: {cm_path}")
    print("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sports Action Recognition Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint file path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save evaluation artifacts")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    # Load checkpoint
    print(f"--> Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    target_classes = checkpoint.get("classes", sorted(UCF101_SPORTS_CLASSES))
    num_classes = len(target_classes)

    # Build model
    backbone = cfg["model"].get("backbone", "r2plus1d_18")
    model = build_model(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])

    # Load test dataset
    data_dir = cfg["dataset"]["data_dir"]
    test_split = cfg["dataset"].get("test_split", "data/splits/testlist01.txt")

    test_dataset = UCFSportsActionDataset(
        data_dir=data_dir,
        split_file=test_split if Path(test_split).exists() else None,
        num_frames=cfg["dataset"].get("num_frames", 16),
        frame_interval=cfg["dataset"].get("frame_interval", 2),
        is_training=False,
        target_classes=target_classes,
    )

    if len(test_dataset) == 0:
        print("[!] No test samples found. Check your dataset paths.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    criterion = nn.CrossEntropyLoss()
    evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        class_names=target_classes,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
