"""Helper script to package the 'ucf action' dataset into a zip archive for Google Drive upload."""

import argparse
from pathlib import Path
import shutil
import sys
import time

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_archive(source_dir: Path, output_zip: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    print("=" * 60)
    print("CREATING DATASET ZIP ARCHIVE FOR GOOGLE COLAB")
    print("=" * 60)
    print(f"Source folder : {source_dir}")
    print(f"Target archive: {output_zip}.zip")
    print("Packing files, this may take 1-2 minutes...")

    t0 = time.time()
    # Create zip without extension in base_name argument
    base_name = str(output_zip).replace(".zip", "")
    shutil.make_archive(base_name, "zip", root_dir=source_dir.parent, base_dir=source_dir.name)

    elapsed = time.time() - t0
    final_path = Path(f"{base_name}.zip")
    size_gb = final_path.stat().st_size / (1024**3)

    print("\n" + "=" * 60)
    print("ARCHIVE CREATED SUCCESSFULLY!")
    print(f"File path : {final_path}")
    print(f"File size : {size_gb:.2f} GB")
    print(f"Time taken: {elapsed:.1f} seconds")
    print("=" * 60)
    print("\nNext step: Upload this 'ucf_action.zip' file to your Google Drive root folder.")


def main():
    parser = argparse.ArgumentParser(description="Package ucf action dataset into zip for Google Colab")
    parser.add_argument("--source", type=str, default="ucf action", help="Source folder to zip")
    parser.add_argument("--output", type=str, default="ucf_action.zip", help="Output zip filename")
    args = parser.parse_args()

    create_archive(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()
