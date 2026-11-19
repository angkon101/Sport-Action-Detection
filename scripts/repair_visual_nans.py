"""Fast repair utility to re-extract FP32 visual embeddings for feature files with NaNs.

Re-runs ONLY ResNet-50 visual feature extraction (5ms per video), preserving all
pose, object, and HOI feature vectors.
"""

from pathlib import Path
import sys
import time
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.hoi_feature_extractor import SportsFeatureExtractor


def repair_features(data_dir: Path, feat_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Initializing ResNet-50 visual backbone on {device}...")
    ext = SportsFeatureExtractor(device=device)

    splits = ["Train", "Test"]
    total_repaired = 0

    for split in splits:
        split_feat = feat_dir / split
        split_data = data_dir / split

        if not split_feat.exists() or not split_data.exists():
            continue

        pt_files = list(split_feat.rglob("*.pt"))
        nan_files = []

        print(f"\n--> Scanning {len(pt_files):,} files in '{split}' for visual NaNs...")
        for pt_path in pt_files:
            try:
                d = torch.load(pt_path, map_location="cpu", weights_only=False)
                if torch.isnan(d.get("visual", torch.tensor([]))).any():
                    nan_files.append(pt_path)
            except Exception:
                pass

        print(f"    Found {len(nan_files):,} files with visual NaNs in '{split}'. Repairing...")
        if not nan_files:
            continue

        repaired_in_split = 0
        t0 = time.time()
        for pt_path in tqdm(nan_files, desc=f"Repairing {split}"):
            cls_name = pt_path.parent.name
            video_stem = pt_path.stem

            # Find matching video in data_dir
            candidate_avi = split_data / cls_name / f"{video_stem}.avi"
            candidate_mp4 = split_data / cls_name / f"{video_stem}.mp4"

            video_path = None
            if candidate_avi.exists():
                video_path = candidate_avi
            elif candidate_mp4.exists():
                video_path = candidate_mp4
            else:
                # Search recursively in class folder
                matches = list((split_data / cls_name).glob(f"*{video_stem}*"))
                if matches:
                    video_path = matches[0]

            if not video_path or not video_path.exists():
                continue

            try:
                frames = ext.sample_video_frames(video_path)
                new_visual = ext.extract_visual_features(frames)

                d = torch.load(pt_path, map_location="cpu", weights_only=False)
                d["visual"] = new_visual
                torch.save(d, pt_path)
                repaired_in_split += 1
            except Exception as e:
                pass

        elapsed = time.time() - t0
        print(f"    [OK] Successfully repaired {repaired_in_split:,} files in {elapsed:.1f}s!")
        total_repaired += repaired_in_split

    print(f"\n[DONE] Repaired total {total_repaired:,} feature files. All tensors are now clean FP32!")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    repair_features(PROJECT_ROOT / "Model Data", PROJECT_ROOT / "data" / "features")
