"""Batch extraction script to extract and cache multimodal features for all sports videos."""

import argparse
from pathlib import Path
import sys
import time
from tqdm import tqdm
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.hoi_feature_extractor import SportsFeatureExtractor


def extract_all(
    data_dir: Path,
    output_dir: Path,
    num_frames: int = 16,
    max_videos_per_class: int = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Extracting multimodal features using device: {device}")
    if device.type == "cuda":
        print(f"--> GPU: {torch.cuda.get_device_name(0)}")

    extractor = SportsFeatureExtractor(device=device, num_frames=num_frames)

    # Discover all sport classes in data_dir
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"--> Found {len(class_dirs)} sport categories in '{data_dir}'")

    total_extracted = 0
    total_skipped = 0
    start_time = time.time()

    for class_idx, cls_folder in enumerate(class_dirs, start=1):
        cls_name = cls_folder.name
        cls_out_dir = output_dir / cls_name
        cls_out_dir.mkdir(parents=True, exist_ok=True)

        video_files = []
        for ext in ["*.avi", "*.mp4", "*.mkv", "*.mov"]:
            video_files.extend(list(cls_folder.rglob(ext)))

        if max_videos_per_class:
            video_files = video_files[:max_videos_per_class]

        print(f"\n[{class_idx}/{len(class_dirs)}] Processing '{cls_name}' ({len(video_files)} videos)...")

        for v_path in tqdm(video_files, desc=cls_name, leave=False):
            save_path = cls_out_dir / f"{v_path.stem}.pt"

            # Skip if already extracted
            if save_path.exists():
                total_skipped += 1
                continue

            try:
                features = extractor.process_video(v_path)
                torch.save(features, save_path)
                total_extracted += 1
            except Exception as e:
                print(f"[!] Error processing {v_path.name}: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total videos extracted : {total_extracted}")
    print(f"Total already cached   : {total_skipped}")
    print(f"Total time elapsed     : {elapsed:.2f} seconds")
    print(f"Saved directory        : {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch Multimodal Feature Extractor for Sports Videos")
    parser.add_argument("--data-dir", type=str, default="ucf action", help="Source video dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/features", help="Output directory for .pt features")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional max videos per class for testing")
    args = parser.parse_args()

    extract_all(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        num_frames=args.num_frames,
        max_videos_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()

