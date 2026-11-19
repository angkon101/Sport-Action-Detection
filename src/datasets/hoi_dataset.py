"""Dataset loader for pre-extracted multimodal HOI features."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class HOIFeatureDataset(Dataset):
    """Loads pre-extracted multimodal features (visual, pose, object, hoi) for training."""

    def __init__(
        self,
        features_dir: str | Path,
        classes: Optional[List[str]] = None,
        preload: bool = True,
    ):
        self.features_dir = Path(features_dir)
        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")

        # Scan class directories
        if classes is not None:
            self.classes = sorted(classes)
        else:
            self.classes = sorted([
                d.name for d in self.features_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])

        self.class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        self._scan_samples()

        # In-Memory Preloading Cache
        self.preload = preload
        self.cached_data: List[Dict[str, torch.Tensor]] = []
        self.cached_labels: List[int] = []

        if self.preload and len(self.samples) > 0:
            print(f"--> Preloading {len(self.samples):,} feature tensors from '{self.features_dir.name}' into RAM...")
            for file_path, label in tqdm(self.samples, desc=f"RAM Cache ({self.features_dir.name})", leave=False):
                try:
                    try:
                        data = torch.load(file_path, map_location="cpu", weights_only=False)
                    except TypeError:
                        data = torch.load(file_path, map_location="cpu")
                    self.cached_data.append({
                        "visual": torch.nan_to_num(data["visual"].float(), nan=0.0),
                        "pose": torch.nan_to_num(data["pose"].float(), nan=0.0),
                        "object": torch.nan_to_num(data["object"].float(), nan=0.0),
                        "hoi": torch.nan_to_num(data["hoi"].float(), nan=0.0),
                    })
                    self.cached_labels.append(label)
                except Exception as e:
                    print(f"  [!] Corrupted feature file {file_path.name}: {e}")

            approx_mb = len(self.cached_data) * 0.138
            print(f"    [OK] Preloaded {len(self.cached_data):,} samples in RAM (~{approx_mb:.1f} MB). 0ms disk seek enabled!")

    def _scan_samples(self) -> None:
        for cls_name in self.classes:
            cls_folder = self.features_dir / cls_name
            if not cls_folder.is_dir():
                continue

            label = self.class_to_idx[cls_name]
            for pt_file in cls_folder.glob("*.pt"):
                self.samples.append((pt_file, label))

    def __len__(self) -> int:
        if self.preload and self.cached_data:
            return len(self.cached_data)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        if self.preload and self.cached_data:
            return self.cached_data[idx], self.cached_labels[idx]

        file_path, label = self.samples[idx]
        try:
            features = torch.load(file_path, map_location="cpu", weights_only=False)
        except TypeError:
            features = torch.load(file_path, map_location="cpu")

        sample_dict = {
            "visual": torch.nan_to_num(features["visual"].float(), nan=0.0),
            "pose": torch.nan_to_num(features["pose"].float(), nan=0.0),
            "object": torch.nan_to_num(features["object"].float(), nan=0.0),
            "hoi": torch.nan_to_num(features["hoi"].float(), nan=0.0),
        }

        return sample_dict, label

