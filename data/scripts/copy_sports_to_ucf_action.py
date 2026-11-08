"""Copies detected sports action folders from UCF101 to 'ucf action'."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def copy_sports_folders(
    source_dir: Path,
    target_dir: Path,
    taxonomy_file: Path,
    move: bool = False,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with open(taxonomy_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sports_classes = []
    for cat_name, items in meta["taxonomy"].items():
        for item in items:
            sports_classes.append(item["name"])

    print(f"Source Directory : {source_dir}")
    print(f"Target Directory : {target_dir}")
    print(f"Action to perform: {'MOVE' if move else 'COPY'}")
    print(f"Total sports folders to process: {len(sports_classes)}\n")

    start_time = time.time()
    processed_count = 0
    total_videos = 0

    for i, cls_name in enumerate(sorted(sports_classes), 1):
        src_cls_path = source_dir / cls_name
        dst_cls_path = target_dir / cls_name

        if not src_cls_path.exists():
            print(f"[{i:02d}/{len(sports_classes)}] Skipping {cls_name} (not found in source)")
            continue

        videos_in_src = list(src_cls_path.glob("*.avi")) + list(src_cls_path.glob("*.mp4"))
        count = len(videos_in_src)
        total_videos += count

        if move:
            shutil.move(str(src_cls_path), str(dst_cls_path))
            action_str = "Moved"
        else:
            if dst_cls_path.exists():
                print(f"[{i:02d}/{len(sports_classes)}] {cls_name} already exists in target. Skipping.")
                processed_count += 1
                continue
            shutil.copytree(str(src_cls_path), str(dst_cls_path))
            action_str = "Copied"

        processed_count += 1
        print(f"[{i:02d}/{len(sports_classes)}] {action_str} '{cls_name}' ({count} videos)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 55)
    print(f"Completed in {elapsed:.2f} seconds!")
    print(f"Folders processed : {processed_count}")
    print(f"Total videos      : {total_videos}")
    print(f"Target directory  : {target_dir}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Copy sports action folders to 'ucf action'")
    parser.add_argument("--source", type=str, default=r"E:\Akhi Thesis\UCF101\UCF-101", help="Source UCF101 directory")
    parser.add_argument("--target", type=str, default=r"E:\Akhi Thesis\ucf action", help="Target ucf action directory")
    parser.add_argument("--taxonomy", type=str, default="data/splits/detected_sports_classes.json", help="Path to taxonomy JSON")
    parser.add_argument("--move", action="store_true", help="Move folders instead of copying")
    args = parser.parse_args()

    copy_sports_folders(
        source_dir=Path(args.source),
        target_dir=Path(args.target),
        taxonomy_file=Path(args.taxonomy),
        move=args.move,
    )


if __name__ == "__main__":
    main()

