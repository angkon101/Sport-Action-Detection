"""Sports Action Recognition Model Architectures.

Provides 3D CNN backbones (R(2+1)D-18, R3D-18, MC3-18, S3D) with Kinetics-400
pretrained weights adapted for sport action classification.
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models.video as vmodels


class SportsActionClassifier(nn.Module):
    """Deep 3D spatio-temporal action classifier for sports recognition."""

    SUPPORTED_BACKBONES = ["r2plus1d_18", "r3d_18", "mc3_18", "s3d"]

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "r2plus1d_18",
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name.lower()

        if self.backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_name}'. Choose from: {self.SUPPORTED_BACKBONES}"
            )

        # Build backbone with optional Kinetics-400 pretrained weights
        if self.backbone_name == "r2plus1d_18":
            weights = vmodels.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
            self.model = vmodels.r2plus1d_18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        elif self.backbone_name == "r3d_18":
            weights = vmodels.R3D_18_Weights.KINETICS400_V1 if pretrained else None
            self.model = vmodels.r3d_18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        elif self.backbone_name == "mc3_18":
            weights = vmodels.MC3_18_Weights.KINETICS400_V1 if pretrained else None
            self.model = vmodels.mc3_18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        elif self.backbone_name == "s3d":
            weights = vmodels.S3D_Weights.KINETICS400_V1 if pretrained else None
            self.model = vmodels.s3d(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Conv3d(in_features, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freezes all layers except the newly initialized classification head."""
        for name, param in self.model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Video tensor of shape [Batch, Channels, Temporal_frames, Height,
            Width] e.g., [B, 3, 16, 224, 224]

        Returns:
            Class logits of shape [Batch, num_classes]
        """
        if self.backbone_name == "s3d":
            # S3D outputs [B, num_classes, 1, 1, 1]
            out = self.model(x)
            return torch.flatten(out, 1)

        return self.model(x)


def build_model(
    num_classes: int,
    backbone: str = "r2plus1d_18",
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
) -> SportsActionClassifier:
    """Factory helper to build a sports action classifier."""
    return SportsActionClassifier(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )

