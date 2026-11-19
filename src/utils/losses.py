"""Advanced Loss Functions for Fine-Grained Multimodal Action Recognition.

Includes:
- FocalLossWithSmoothing: Focuses on hard/fine-grained classes while smoothing targets.
- SoftTargetCrossEntropy: For Feature Mixup training.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithSmoothing(nn.Module):
    """Focal Loss with label smoothing to tackle class imbalance and overconfidence.
    
    FL(p_t) = - alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)
        preds = log_preds.exp()

        # Create one-hot / smoothed targets
        if targets.dim() == 1:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            # Targets already soft (e.g. from Mixup)
            smooth_targets = targets * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        # Focal weighting: (1 - p_t)^gamma
        pt = (preds * smooth_targets).sum(dim=-1, keepdim=True)
        focal_weight = (1.0 - pt).clamp(min=1e-6).pow(self.gamma)

        # Cross entropy with focal modulation
        loss = -focal_weight * (smooth_targets * log_preds).sum(dim=-1, keepdim=True)

        if self.weight is not None:
            class_weights = self.weight[targets].unsqueeze(1) if targets.dim() == 1 else 1.0
            loss = loss * class_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss.squeeze(-1)


class SoftTargetCrossEntropy(nn.Module):
    """Cross Entropy for Mixup training with probabilistic targets."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()
