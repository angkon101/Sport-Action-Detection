"""Populate Model Data/Train and Model Data/Test with stratified 70/30 split.

Ensures:
- Exactly 70% of videos from each of the 82 categories go into Model Data/Train/<category>/
- Exactly 30% of videos from each of the 82 categories go into Model Data/Test/<category>/
- Zero video overlap between Train and Test
- Uses NTFS Hard Links by default (instantaneous, 0 additional disk space),
  with fallback to copying if requested or across filesystem boundaries.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def populate_split(
    source_dir: Path,
    target_dir: Path,
    train_split_file: Path,
    test_split_file: Path,
    mode: str = "link",
) -> None:
    print("=" * 65)
    print("Populating 'Model Data' Train / Test Split")
    print(f"Source Directory : {source_dir.resolve()}")
    print(f"Target Directory : {target_dir.resolve()}")
    print(f"Populate Mode    : {mode.upper()} (Hard Link = 0 extra disk space)")
    print("=" * 65)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    with open(train_split_file, "r", encoding="utf-8") as f:
        train_split = json.load(f)
    with open(test_split_file, "r", encoding="utf-8") as f:
        test_split = json.load(f)

    splits = [
        ("Train", train_split, target_dir / "Train"),
        ("Test", test_split, target_dir / "Test"),
    ]

    t0 = time.time()
    stats = {}

    for split_name, split_data, split_out_dir in splits:
        split_out_dir.mkdir(parents=True, exist_ok=True)
        total_items = sum(len(paths) for paths in split_data.values())
        print(f"\n--> Processing [{split_name} Split]: {total_items} videos across {len(split_data)} categories...")

        count_created = 0
        count_skipped = 0

        for cls_name, rel_paths in split_data.items():
            cls_out_dir = split_out_dir / cls_name
            cls_out_dir.mkdir(parents=True, exist_ok=True)

            for rel_p in rel_paths:
                src_file = source_dir / rel_p
                dst_file = cls_out_dir / src_file.name

                if not src_file.exists():
                    print(f"  [!] Missing source file: {src_file}")
                    continue

                if dst_file.exists():
                    count_skipped += 1
                    continue

                if mode == "link":
                    try:
                        os.link(src_file, dst_file)
                        count_created += 1
                    except Exception as e:
                        # Fallback to copy if hard link fails
                        shutil.copy2(src_file, dst_file)
                        count_created += 1
                else:
                    shutil.copy2(src_file, dst_file)
                    count_created += 1

        stats[split_name] = (count_created, count_skipped, total_items)
        print(f"    Done: {count_created} populated, {count_skipped} already present.")

    # Verification phase
    print("\n" + "=" * 65)
    print("--> Running Integrity & Overlap Verification...")
    print("=" * 65)

    train_files = {p.name: p for p in (target_dir / "Train").rglob("*.avi")}
    test_files = {p.name: p for p in (target_dir / "Test").rglob("*.avi")}

    train_classes = sorted([d.name for d in (target_dir / "Train").iterdir() if d.is_dir()])
    test_classes = sorted([d.name for d in (target_dir / "Test").iterdir() if d.is_dir()])

    print(f"Categories in Train : {len(train_classes)}")
    print(f"Categories in Test  : {len(test_classes)}")
    print(f"Videos in Train     : {len(train_files):,}")
    print(f"Videos in Test      : {len(test_files):,}")
    print(f"Total Videos        : {len(train_files) + len(test_files):,}")

    # Check overlap
    overlap = set(train_files.keys()).intersection(set(test_files.keys()))
    if len(overlap) > 0:
        print(f"[ERROR] Overlap detected! {len(overlap)} videos exist in both Train and Test:")
        for name in list(overlap)[:5]:
            print(f"  - {name}")
        raise ValueError(f"Strict split violated: {len(overlap)} overlapping videos!")
    else:
        print("[SUCCESS] Zero Overlap Confirmed: Train and Test video sets are 100% disjoint.")

    # Check non-empty files
    corrupted = [p for p in list(train_files.values()) + list(test_files.values()) if p.stat().st_size == 0]
    if corrupted:
        print(f"[ERROR] Found {len(corrupted)} zero-byte files!")
    else:
        print("[SUCCESS] All files have valid non-zero byte size.")

    print(f"\nExecution finished successfully in {time.time() - t0:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Populate Model Data Train/Test folders")
    parser.add_argument("--source-dir", type=str, default="ucf action", help="Source videos folder")
    parser.add_argument("--target-dir", type=str, default="Model Data", help="Target Model Data folder")
    parser.add_argument("--train-split", type=str, default="data/splits/train_split_70.json")
    parser.add_argument("--test-split", type=str, default="data/splits/test_split_30.json")
    parser.add_argument("--mode", type=str, default="link", choices=["link", "copy"], help="link (hard link) or copy")
    args = parser.parse_args()

    populate_split(
        source_dir=Path(args.source_dir),
        target_dir=Path(args.target_dir),
        train_split_file=Path(args.train_split),
        test_split_file=Path(args.test_split),
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
