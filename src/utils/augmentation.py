"""Feature-Space Data Augmentation for Multimodal Action Recognition.

Enables robust generalization on fine-grained video features without raw video re-encoding.
Techniques:
1. Multimodal Mixup (Feature-space convex combination)
2. Temporal Frame Masking (Simulates sensor dropouts / occlusions)
3. Kinematic Perturbation (Subtle jitter on joint/object trajectories)
"""

from typing import Dict, Tuple
import numpy as np
import torch


class MultimodalAugmentor:
    """Applies stochastic feature-space augmentations during training."""

    def __init__(
        self,
        mixup_prob: float = 0.5,
        mixup_alpha: float = 0.4,
        drop_frame_prob: float = 0.25,
        jitter_std: float = 0.008,
    ):
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.drop_frame_prob = drop_frame_prob
        self.jitter_std = jitter_std

    def apply_temporal_mask(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Randomly masks 1 or 2 frames across the batch."""
        if np.random.rand() > self.drop_frame_prob:
            return batch

        B, T = batch["visual"].shape[:2]
        num_drops = np.random.randint(1, 3)
        drop_indices = np.random.choice(T, size=num_drops, replace=False)

        augmented = {}
        for key, tensor in batch.items():
            augmented_tensor = tensor.clone()
            for idx in drop_indices:
                # Mask with temporal neighbor interpolation or zero
                if idx > 0:
                    augmented_tensor[:, idx] = augmented_tensor[:, idx - 1]
                else:
                    augmented_tensor[:, idx] = augmented_tensor[:, min(idx + 1, T - 1)]
            augmented[key] = augmented_tensor

        return augmented

    def apply_kinematic_jitter(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds subtle zero-mean Gaussian jitter to pose and object trajectories."""
        augmented = {k: v.clone() for k, v in batch.items()}
        if self.jitter_std > 0:
            p = augmented["pose"]
            noise_p = torch.randn_like(p) * self.jitter_std
            augmented["pose"] = p + noise_p

            o = augmented["object"]
            noise_o = torch.randn_like(o) * self.jitter_std
            augmented["object"] = o + noise_o

        return augmented

    def apply_mixup(
        self,
        batch: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        num_classes: int = 82,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Performs multimodal feature Mixup blending two sample trajectories."""
        if np.random.rand() > self.mixup_prob:
            # One-hot encode targets for uniform loss interface
            one_hot = torch.zeros(targets.size(0), num_classes, device=targets.device)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
            return batch, one_hot

        # Sample Beta distribution lambda
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        B = targets.size(0)
        perm = torch.randperm(B, device=targets.device)

        mixed_batch = {}
        for k, v in batch.items():
            mixed_batch[k] = lam * v + (1.0 - lam) * v[perm]

        # Mixed one-hot targets
        y_a = torch.zeros(B, num_classes, device=targets.device).scatter_(1, targets.unsqueeze(1), 1.0)
        y_b = torch.zeros(B, num_classes, device=targets.device).scatter_(1, targets[perm].unsqueeze(1), 1.0)
        mixed_targets = lam * y_a + (1.0 - lam) * y_b

        return mixed_batch, mixed_targets
