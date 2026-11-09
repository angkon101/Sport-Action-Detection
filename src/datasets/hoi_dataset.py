"""Dataset loader for pre-extracted multimodal HOI features."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset


class HOIFeatureDataset(Dataset):
    """Loads pre-extracted multimodal features (visual, pose, object, hoi) for training."""

    def __init__(
        self,
        features_dir: str | Path,
        classes: Optional[List[str]] = None,
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

    def _scan_samples(self) -> None:
        for cls_name in self.classes:
            cls_folder = self.features_dir / cls_name
            if not cls_folder.is_dir():
                continue

            label = self.class_to_idx[cls_name]
            for pt_file in cls_folder.glob("*.pt"):
                self.samples.append((pt_file, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        file_path, label = self.samples[idx]

        # Load cached feature dictionary
        features = torch.load(file_path, map_location="cpu")

        sample_dict = {
            "visual": features["visual"].float(),  # [16, 2048]
            "pose": features["pose"].float(),      # [16, 85]
            "object": features["object"].float(),  # [16, 7]
            "hoi": features["hoi"].float(),        # [16, 16]
        }

        return sample_dict, label

