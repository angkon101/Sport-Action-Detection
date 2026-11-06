"""Script to download and unpack the UCF101 dataset and official train/test splits.

Official Source: University of Central Florida Center for Research in Computer Vision (CRCV)
- Videos: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
- Train/Test Splits: https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path


SPLITS_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"


def download_splits(destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    zip_path = destination_dir / "UCF101TrainTestSplits.zip"

    print(f"Downloading official splits to {zip_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(SPLITS_URL, headers=headers)
    with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    print("Extracting splits...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination_dir)

    print("Splits successfully downloaded and extracted.")


def main():
    parser = argparse.ArgumentParser(description="Download UCF101 official splits and metadata.")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Directory to store splits")
    parser.add_argument("--download-splits-only", action="store_true", default=True, help="Download train/test split files")
    args = parser.parse_args()

    download_splits(Path(args.splits_dir))


if __name__ == "__main__":
    main()

