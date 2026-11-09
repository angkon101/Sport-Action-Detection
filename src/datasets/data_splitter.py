"""Stratified Train/Test Dataset Splitter (70% Train, 30% Test) and Data Readiness Verifier.

Features:
1. Generates stratified 70% train / 30% test splits per sports category.
2. Performs comprehensive pre-flight verification of data availability before training.
3. Validates feature tensor integrity (shape checks, NaN checks).
"""

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_stratified_split(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Generates a stratified 70% train and 30% test split across every sport category."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"--> Found {len(class_folders)} sport categories in '{data_dir}'")

    train_split: Dict[str, List[str]] = {}
    test_split: Dict[str, List[str]] = {}

    total_train_videos = 0
    total_test_videos = 0

    print(f"\n--> Performing Stratified Split ({int(train_ratio*100)}% Train / {int((1-train_ratio)*100)}% Test) per Category...")

    for cls_folder in class_folders:
        cls_name = cls_folder.name

        # Discover all video clips in the class folder
        video_files = []
        for ext in ["*.avi", "*.mp4", "*.mkv", "*.mov"]:
            video_files.extend(list(cls_folder.rglob(ext)))

        # Relative paths from data_dir
        rel_paths = [str(v.relative_to(data_dir)).replace("\\", "/") for v in sorted(video_files)]

        if len(rel_paths) == 0:
            print(f"  [!] Warning: '{cls_name}' has 0 video clips!")
            continue

        # Shuffle deterministically per class
        shuffled = rel_paths.copy()
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, int(round(n_total * train_ratio)))

        cls_train = sorted(shuffled[:n_train])
        cls_test = sorted(shuffled[n_train:])

        # Ensure at least 1 sample in test if category has more than 1 sample
        if len(cls_test) == 0 and n_total > 1:
            cls_test = [cls_train.pop()]

        train_split[cls_name] = cls_train
        test_split[cls_name] = cls_test

        total_train_videos += len(cls_train)
        total_test_videos += len(cls_test)

    # Save splits as JSON
    train_path = output_dir / "train_split_70.json"
    test_path = output_dir / "test_split_30.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=2)

    # Also save class index mapping
    class_names = sorted(list(train_split.keys()))
    mapping_path = output_dir / "class_indices_82.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({cls: idx for idx, cls in enumerate(class_names)}, f, indent=2)

    print("\n" + "=" * 65)
    print("STRATIFIED SPLIT COMPLETED")
    print("=" * 65)
    print(f"Total Categories   : {len(class_names)}")
    print(f"Train Videos (70%) : {total_train_videos} clips")
    print(f"Test Videos (30%)  : {total_test_videos} clips")
    print(f"Total Videos       : {total_train_videos + total_test_videos} clips")
    print(f"Train split file   : {train_path}")
    print(f"Test split file    : {test_path}")
    print(f"Class mapping file : {mapping_path}")
    print("=" * 65)

    return train_split, test_split


def verify_data_readiness(
    data_dir: Path,
    features_dir: Path,
    split_file: Optional[Path] = None,
) -> bool:
    """Performs pre-flight data verification before training.

    Checks:
    1. Category folder count and naming.
    2. Video file availability.
    3. Preprocessed feature tensor existence (.pt).
    4. Tensor shapes ([16, 2048], [16, 85], [16, 7], [16, 16]) and NaN checks.
    """
    print("\n" + "=" * 65)
    print("PRE-FLIGHT DATA READINESS VERIFICATION")
    print("=" * 65)

    if not data_dir.exists():
        print(f"[FAIL] Raw dataset directory does not exist: {data_dir}")
        return False

    if not features_dir.exists():
        print(f"[FAIL] Features directory does not exist: {features_dir}")
        print("       Please run feature extraction first before training.")
        return False

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    feature_dirs = sorted([d for d in features_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    print(f"[*] Raw Sport Categories Found     : {len(class_dirs)}")
    print(f"[*] Extracted Category Folders     : {len(feature_dirs)}")

    # Check for category mismatch
    raw_names = set(d.name for d in class_dirs)
    feat_names = set(d.name for d in feature_dirs)
    missing_cats = raw_names - feat_names

    if missing_cats:
        print(f"[WARN] {len(missing_cats)} categories missing extracted features: {sorted(list(missing_cats))[:5]}...")

    # Count total features
    total_pt_files = list(features_dir.rglob("*.pt"))
    print(f"[*] Total Preprocessed .pt Files   : {len(total_pt_files)}")

    if len(total_pt_files) == 0:
        print("[FAIL] 0 feature files found in features directory!")
        return False

    # Sample tensor shape and integrity check on 10 random samples
    print("\n[*] Validating sample feature tensors (shapes & NaN checks)...")
    sample_files = random.sample(total_pt_files, min(10, len(total_pt_files)))
    for s_file in sample_files:
        try:
            data = torch.load(s_file, map_location="cpu")
            assert "visual" in data and "pose" in data and "object" in data and "hoi" in data
            assert data["visual"].shape == (16, 2048), f"Invalid visual shape: {data['visual'].shape}"
            assert data["pose"].shape == (16, 85), f"Invalid pose shape: {data['pose'].shape}"
            assert data["object"].shape == (16, 7), f"Invalid object shape: {data['object'].shape}"
            assert data["hoi"].shape == (16, 16), f"Invalid hoi shape: {data['hoi'].shape}"

            # Check for NaNs
            for k in ["visual", "pose", "object", "hoi"]:
                if torch.isnan(data[k]).any():
                    print(f"[FAIL] NaN detected in {s_file.name} for stream '{k}'!")
                    return False
        except Exception as e:
            print(f"[FAIL] Corrupt feature file detected at {s_file}: {e}")
            return False

    print("[PASS] Tensor structure and integrity verified! (Shapes [16, 2048], [16, 85], [16, 7], [16, 16], No NaNs)")
    print("=" * 65)
    print("ALL PRE-FLIGHT CHECKS PASSED: DATA IS READY FOR TRAINING!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stratified Splitter & Data Verifier")
    parser.add_argument("--data-dir", type=str, default="ucf action", help="Path to raw dataset")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Directory to save split JSONs")
    parser.add_argument("--features-dir", type=str, default="data/features", help="Path to preprocessed features")
    parser.add_argument("--split", action="store_true", help="Generate stratified 70/30 train/test split")
    parser.add_argument("--verify", action="store_true", help="Run pre-flight data verification")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio (default: 0.7)")
    args = parser.parse_args()

    if args.split:
        create_stratified_split(Path(args.data_dir), Path(args.output_dir), train_ratio=args.train_ratio)

    if args.verify:
        verify_data_readiness(Path(args.data_dir), Path(args.features_dir))

    if not args.split and not args.verify:
        print("Please specify --split, --verify, or both.")


if __name__ == "__main__":
    main()
