"""Comprehensive Visualizer & Diagram Generator for Model Evaluation.

Generates 5 publication-ready diagrams for thesis presentation:
1. Training & Validation Loss curves (loss_curves.png)
2. Top-1 and Top-5 Accuracy curves (accuracy_curves.png)
3. Normalized Confusion Matrix (confusion_matrix_82.png)
4. Per-Class Accuracy Rankings: Top 10 vs Bottom 10 sports (per_class_accuracy.png)
5. Temporal Attention Weights visualization across 16 frames (temporal_attention_weights.png)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_learning_curves(history: List[Dict], output_dir: Path) -> None:
    """Plots and saves loss and accuracy learning curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    # 1. Loss Curve
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [h["train_loss"] for h in history], "b-o", label="Train Loss", linewidth=2, markersize=4)
    plt.plot(epochs, [h["val_loss"] for h in history], "r--s", label="Validation Loss", linewidth=2, markersize=4)
    plt.title("HOI-BiLSTM Training & Validation Loss Across Epochs", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=300)
    plt.close()

    # 2. Accuracy Curves
    plt.figure(figsize=(9, 5))
    if "train_top1" in history[0]:
        plt.plot(epochs, [h["train_top1"] for h in history], "g-o", label="Train Top-1 Acc", linewidth=2, markersize=4)
    plt.plot(epochs, [h["val_top1"] for h in history], "b-^", label="Validation Top-1 Acc", linewidth=2, markersize=4)
    if "val_top5" in history[0]:
        plt.plot(epochs, [h["val_top5"] for h in history], "m--d", label="Validation Top-5 Acc", linewidth=2, markersize=4)

    plt.title("HOI-BiLSTM Top-1 & Top-5 Accuracy Curves", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curves.png", dpi=300)
    plt.close()


def plot_confusion_matrix_82(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_dir: Path,
) -> None:
    """Plots a high-resolution normalized confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    plt.figure(figsize=(24, 20))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Accuracy"},
    )
    plt.title("Normalized Confusion Matrix (82 Sports Classes) - HOI-BiLSTM", fontsize=16, pad=15)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_82.png", dpi=300)
    plt.close()


def plot_per_class_accuracy(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_dir: Path,
) -> None:
    """Plots Top 10 best-performing vs Top 10 most challenging sports categories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    class_recalls = []
    for cls in class_names:
        if cls in report:
            class_recalls.append((cls, report[cls]["recall"] * 100.0))

    class_recalls.sort(key=lambda x: x[1])

    bottom_10 = class_recalls[:10]
    top_10 = class_recalls[-10:]
    combined = bottom_10 + top_10

    names = [c[0] for c in combined]
    scores = [c[1] for c in combined]
    colors = ["#e74c3c"] * 10 + ["#2ecc71"] * 10

    plt.figure(figsize=(12, 8))
    bars = plt.barh(names, scores, color=colors, height=0.6)
    plt.axvline(x=np.mean([c[1] for c in class_recalls]), color="black", linestyle="--", alpha=0.7, label="Mean Accuracy")
    plt.title("Sport Action Recognition: Top 10 Highest vs Lowest Performing Categories", fontsize=14, pad=15)
    plt.xlabel("Recognition Accuracy / Recall (%)", fontsize=12)
    plt.xlim(0, 105)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, score in zip(bars, scores):
        plt.text(score + 1.0, bar.get_y() + bar.get_height() / 2, f"{score:.1f}%", va="center", fontsize=9)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_accuracy.png", dpi=300)
    plt.close()


def plot_temporal_attention_samples(
    sample_attentions: List[Tuple[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """Plots temporal attention weights across the 16 frames for sample sports actions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(1, 17)

    plt.figure(figsize=(10, 6))
    for class_name, attn_weights in sample_attentions[:5]:
        plt.plot(frames, attn_weights, marker="o", linewidth=2.2, label=f"Action: {class_name}")

    plt.title("HOI-BiLSTM Temporal Attention Allocation Across 16 Frames", fontsize=14, pad=15)
    plt.xlabel("Video Frame Index (Temporal Progression)", fontsize=12)
    plt.ylabel("Attention Weight (α_t)", fontsize=12)
    plt.xticks(frames)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_attention_weights.png", dpi=300)
    plt.close()
