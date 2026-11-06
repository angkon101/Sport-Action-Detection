"""Inference and Visual Demo Pipeline for Sports Action Recognition."""

import argparse
from pathlib import Path
import sys
from typing import List, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.sports_classes import UCF101_SPORTS_CLASSES
from src.datasets.transforms import VideoTransform
from src.models.action_classifier import build_model


def predict_video(
    video_path: str | Path,
    model: torch.nn.Module,
    classes: List[str],
    device: torch.device,
    num_frames: int = 16,
    topk: int = 3,
) -> List[Tuple[str, float]]:
    """Predicts the top-k sport action classes for a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError("Video contains 0 readable frames.")

    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled = [frames[i] for i in indices]
    video_tensor = torch.from_numpy(np.array(sampled, dtype=np.uint8))  # [T, H, W, C]

    transform = VideoTransform(input_size=224, is_training=False)
    video_tensor = transform(video_tensor).unsqueeze(0).to(device)  # [1, C, T, H, W]

    with torch.no_grad():
        logits = model(video_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = probs.topk(min(topk, len(classes)))
    predictions = [
        (classes[idx.item()], prob.item() * 100.0)
        for idx, prob in zip(top_indices, top_probs)
    ]
    return predictions


def annotate_and_save_video(
    input_path: str | Path,
    output_path: str | Path,
    predictions: List[Tuple[str, float]],
) -> None:
    """Overlays the action prediction banner onto the original video and writes to file."""
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    top_class, top_conf = predictions[0]
    overlay_text = f"Action: {top_class} ({top_conf:.1f}%)"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw semi-transparent header overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw prediction label
        cv2.putText(
            frame,
            overlay_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)

    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sports Action Recognition Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint file path")
    parser.add_argument("--backbone", type=str, default="r2plus1d_18", help="Backbone architecture")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for annotated video")
    parser.add_argument("--topk", type=int, default=5, help="Top-K predicted actions to display")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"--> Loading trained checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        classes = checkpoint.get("classes", sorted(UCF101_SPORTS_CLASSES))
        num_classes = len(classes)
        model = build_model(num_classes=num_classes, backbone=args.backbone, pretrained=False).to(device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print(f"--> Checkpoint not found at '{checkpoint_path}'. Initializing pretrained backbone baseline...")
        classes = sorted(UCF101_SPORTS_CLASSES)
        num_classes = len(classes)
        model = build_model(num_classes=num_classes, backbone=args.backbone, pretrained=True).to(device)

    model.eval()

    print(f"\nAnalyzing video: {args.video} ...")
    predictions = predict_video(
        video_path=args.video,
        model=model,
        classes=classes,
        device=device,
        topk=args.topk,
    )

    print("\n" + "=" * 45)
    print("ACTION RECOGNITION PREDICTIONS")
    print("=" * 45)
    for rank, (cls_name, prob) in enumerate(predictions, start=1):
        bar = "█" * int(prob / 5)
        print(f" {rank}. {cls_name:<22} {prob:5.1f}%  {bar}")
    print("=" * 45)

    if args.output:
        annotate_and_save_video(args.video, args.output, predictions)


if __name__ == "__main__":
    main()
