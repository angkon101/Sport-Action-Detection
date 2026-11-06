"""Video Transformation and Augmentation Pipelines for Action Recognition."""

from typing import Tuple
import random
import torch
import torchvision.transforms.functional as F


# Kinetics / VideoResNet standard normalization statistics
VIDEO_MEAN = [0.43216, 0.394666, 0.37645]
VIDEO_STD = [0.22803, 0.22145, 0.216989]


class VideoTransform:
    """Base video transform wrapper applying consistent spatial transformations

    across all temporal frames in a video tensor [T, H, W, C] or [C, T, H, W].
    """

    def __init__(
        self,
        input_size: int = 224,
        is_training: bool = True,
        hflip_prob: float = 0.5,
        mean: Tuple[float, float, float] = tuple(VIDEO_MEAN),
        std: Tuple[float, float, float] = tuple(VIDEO_STD),
    ):
        self.input_size = input_size
        self.is_training = is_training
        self.hflip_prob = hflip_prob
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Args:

            video: Tensor of shape [T, H, W, C] with values in range [0, 255]
            (uint8 or float)

        Returns:
            Normalized tensor of shape [C, T, H, W] in float32
        """
        # Ensure float in range [0, 1]
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        elif video.max() > 1.0:
            video = video.float() / 255.0
        else:
            video = video.float()

        # Permute to [T, C, H, W] for frame-wise torchvision transforms
        # Video input is [T, H, W, C]
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

        # Determine spatial augmentation parameters for the entire clip (consistent across frames)
        do_hflip = self.is_training and (random.random() < self.hflip_prob)

        t_frames, c, h, w = video.shape
        transformed_frames = []

        # Resize dimensions
        resize_size = int(self.input_size * 1.14) if self.is_training else int(self.input_size * 1.14)

        for t in range(t_frames):
            frame = video[t]  # [C, H, W]

            # Resize shorter side
            frame = F.resize(frame, [resize_size, resize_size], antialias=True)

            # Crop
            if self.is_training:
                # Random crop
                crop_y = random.randint(0, resize_size - self.input_size)
                crop_x = random.randint(0, resize_size - self.input_size)
                frame = F.crop(frame, crop_y, crop_x, self.input_size, self.input_size)
            else:
                # Center crop
                frame = F.center_crop(frame, [self.input_size, self.input_size])

            if do_hflip:
                frame = F.hflip(frame)

            # Normalize
            frame = F.normalize(frame, mean=self.mean, std=self.std)
            transformed_frames.append(frame)

        # Stack back to [T, C, H, W] then permute to [C, T, H, W] as expected by 3D CNNs
        out_video = torch.stack(transformed_frames, dim=0)  # [T, C, H, W]
        out_video = out_video.permute(1, 0, 2, 3)  # [C, T, H, W]
        return out_video


def get_video_transforms(input_size: int = 224) -> Tuple[VideoTransform, VideoTransform]:
    """Returns (train_transform, val_transform)"""
    train_transform = VideoTransform(input_size=input_size, is_training=True)
    val_transform = VideoTransform(input_size=input_size, is_training=False)
    return train_transform, val_transform
