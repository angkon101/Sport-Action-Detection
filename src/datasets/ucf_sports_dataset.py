"""PyTorch Dataset for UCF Sports Action Recognition."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.sports_classes import UCF101_SPORTS_CLASSES, get_sports_class_to_idx
from src.datasets.transforms import get_video_transforms


class UCFSportsActionDataset(Dataset):
    """Dataset for loading and temporal sampling of UCF sport video clips.

    Supports both:
    1. Direct folder scan (data_dir/<class_name>/<video_file>)
    2. Official split files (e.g., trainlist01.txt with 'Class/video.avi <label>')
    """

    def __init__(
        self,
        data_dir: str | Path,
        split_file: Optional[str | Path] = None,
        num_frames: int = 16,
        frame_interval: int = 2,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        is_training: bool = True,
        target_classes: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split_file = Path(split_file) if split_file else None
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.is_training = is_training
        self.classes = sorted(target_classes or UCF101_SPORTS_CLASSES)
        self.class_to_idx = get_sports_class_to_idx(self.classes)

        if transform is None:
            train_t, val_t = get_video_transforms(input_size=224)
            self.transform = train_t if is_training else val_t
        else:
            self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Parses video files and maps them to their respective class labels."""
        if self.split_file and self.split_file.exists():
            with open(self.split_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    parts = line_str.split()
                    rel_path = parts[0]
                    cls_name = rel_path.split("/")[0] if "/" in rel_path else rel_path.split("\\")[0]

                    if cls_name in self.class_to_idx:
                        full_video_path = self.data_dir / rel_path
                        label = self.class_to_idx[cls_name]
                        self.samples.append((full_video_path, label))
        else:
            # Automatic directory scanning for target sport classes
            for cls_name in self.classes:
                cls_folder = self.data_dir / cls_name
                if not cls_folder.is_dir():
                    continue

                label = self.class_to_idx[cls_name]
                for ext in ["*.avi", "*.mp4", "*.mkv", "*.mov"]:
                    for video_path in cls_folder.glob(ext):
                        self.samples.append((video_path, label))

    def _load_video_frames(self, video_path: Path) -> np.ndarray:
        """Reads frames from video file using OpenCV and performs temporal sampling.

        Returns array of shape [T, H, W, 3] (RGB uint8).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            # Return dummy zero-tensor if video cannot be opened
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Read frame by frame to determine total count
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if len(frames) == 0:
                return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

            total_frames = len(frames)
            indices = self._sample_indices(total_frames)
            sampled = [frames[i] for i in indices]
            return np.array(sampled, dtype=np.uint8)

        # Sample indices
        indices = self._sample_indices(total_frames)
        sampled_frames = []

        # Seek and extract only needed frames for efficiency
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(frame_rgb)
            elif len(sampled_frames) > 0:
                sampled_frames.append(sampled_frames[-1])
            else:
                sampled_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()
        return np.array(sampled_frames, dtype=np.uint8)

    def _sample_indices(self, total_frames: int) -> np.ndarray:
        """Computes frame indices to sample from the video."""
        clip_span = self.num_frames * self.frame_interval

        if total_frames <= self.num_frames:
            # Pad by repeating indices
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            return indices

        if self.is_training:
            # Random temporal clip selection
            if total_frames > clip_span:
                start_frame = np.random.randint(0, total_frames - clip_span)
                indices = np.arange(start_frame, start_frame + clip_span, self.frame_interval)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Deterministic center temporal clip
            if total_frames > clip_span:
                start_frame = (total_frames - clip_span) // 2
                indices = np.arange(start_frame, start_frame + clip_span, self.frame_interval)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        return np.clip(indices, 0, total_frames - 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = self._load_video_frames(video_path)  # [T, H, W, C]
        video_tensor = torch.from_numpy(frames)

        if self.transform is not None:
            video_tensor = self.transform(video_tensor)  # Returns [C, T, H, W]

        return video_tensor, label
