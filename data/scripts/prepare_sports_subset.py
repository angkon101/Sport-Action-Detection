"""Filters UCF101 official splits to extract the Sports Action subset."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.sports_classes import (
    UCF101_SPORTS_CLASSES,
    filter_splits_for_sports,
    get_sports_class_to_idx,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare UCF101 Sports Action Subset Splits")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Path to UCF101 splits directory")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory for filtered splits")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sports class mapping
    class_to_idx = get_sports_class_to_idx(UCF101_SPORTS_CLASSES)
    mapping_file = output_dir / "sports_classes.txt"
    with open(mapping_file, "w", encoding="utf-8") as f:
        for cls_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
            f.write(f"{idx} {cls_name}\n")
    print(f"--> Saved {len(class_to_idx)} sports classes to {mapping_file}")

    # Process split files if they exist
    split_pairs = [
        ("trainlist01.txt", "sports_trainlist01.txt"),
        ("testlist01.txt", "sports_testlist01.txt"),
        ("trainlist02.txt", "sports_trainlist02.txt"),
        ("testlist02.txt", "sports_testlist02.txt"),
        ("trainlist03.txt", "sports_trainlist03.txt"),
        ("testlist03.txt", "sports_testlist03.txt"),
    ]

    found = 0
    for src_name, dst_name in split_pairs:
        src_file = splits_dir / src_name
        if src_file.exists():
            dst_file = output_dir / dst_name
            count = filter_splits_for_sports(src_file, dst_file, UCF101_SPORTS_CLASSES)
            print(f"--> Processed {src_name} -> {dst_name} ({count} sports video clips)")
            found += 1

    if found == 0:
        print("\nNote: Official UCF101 split files (trainlist01.txt, etc.) were not found in data/splits/.")
        print("You can run 'python data/scripts/download_ucf101.py' to download official split files automatically,")
        print("or place your downloaded video clips in 'data/raw/<SportClass>/'.")


if __name__ == "__main__":
    main()
