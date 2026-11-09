"""Human-Object Interaction Multimodal BiLSTM Model (HOI-BiLSTM)

Architectural Framework:
- Branch 1: Visual Scene CNN Representation (c_t)
- Branch 2: Human Pose & Biomechanics Keypoints (h_t)
- Branch 3: Sports Object Trajectory (o_t)
- Branch 4: Geometric Human-Object Interaction (i_t)
- Multimodal Fusion: Concatenation + Projection
- Temporal Modeling: 2-layer Bidirectional LSTM
- Temporal Attention Pooling: Self-Attention mechanism over time steps
- Classification Head: 82 Sports Classes output
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPooling(nn.Module):
    """Self-Attention Pooling over time steps to focus on critical action moments."""

    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

            h: Tensor of shape [Batch, Time, HiddenDim] mask: Optional boolean
            mask of shape [Batch, Time]

        Returns:
            context_vector: [Batch, HiddenDim]
            attention_weights: [Batch, Time]
        """
        # Energy scores: [B, T, 1]
        energy = self.projection(h).squeeze(-1)  # [B, T]

        if mask is not None:
            energy = energy.masked_fill(~mask, -1e9)

        # Softmax over temporal dimension
        alpha = F.softmax(energy, dim=-1)  # [B, T]

        # Context vector: Weighted sum across time
        # [B, 1, T] x [B, T, D] -> [B, 1, D] -> [B, D]
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)

        return context, alpha


class HOIBiLSTMClassifier(nn.Module):
    """Human-Object Interaction Multimodal BiLSTM for Sports Action Recognition."""

    def __init__(
        self,
        num_classes: int = 82,
        # Feature input dimensions (from extractors)
        visual_in_dim: int = 2048,   # e.g., ResNet-50 / ConvNeXt pooled feature
        pose_in_dim: int = 85,       # 17 keypoints * 3 (x, y, conf) + 17 * 2 (vel)
        object_in_dim: int = 7,      # [cx, cy, w, h, vx, vy, mask]
        hoi_in_dim: int = 16,        # 4 interaction joints * (1 dist + 2 unit vec + 1 delta)
        # Latent branch projection dimensions
        visual_proj_dim: int = 256,
        pose_proj_dim: int = 128,
        object_proj_dim: int = 64,
        hoi_proj_dim: int = 64,
        # Temporal BiLSTM settings
        bilstm_hidden: int = 256,    # Each direction hidden size -> total 512
        bilstm_layers: int = 2,
        dropout: float = 0.3,
        # Ablation flags (enables testing M1 through M7)
        use_visual: bool = True,
        use_pose: bool = True,
        use_object: bool = True,
        use_hoi: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_visual = use_visual
        self.use_pose = use_pose
        self.use_object = use_object
        self.use_hoi = use_hoi
        self.use_attention = use_attention

        # ----------------------------------------------------
        # 1. Branch Projections
        # ----------------------------------------------------
        fusion_in_dim = 0

        if self.use_visual:
            self.visual_branch = nn.Sequential(
                nn.Linear(visual_in_dim, visual_proj_dim),
                nn.LayerNorm(visual_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fusion_in_dim += visual_proj_dim

        if self.use_pose:
            self.pose_branch = nn.Sequential(
                nn.Linear(pose_in_dim, pose_proj_dim),
                nn.LayerNorm(pose_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fusion_in_dim += pose_proj_dim

        if self.use_object:
            self.object_branch = nn.Sequential(
                nn.Linear(object_in_dim, object_proj_dim),
                nn.LayerNorm(object_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fusion_in_dim += object_proj_dim

        if self.use_hoi:
            self.hoi_branch = nn.Sequential(
                nn.Linear(hoi_in_dim, hoi_proj_dim),
                nn.LayerNorm(hoi_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fusion_in_dim += hoi_proj_dim

        if fusion_in_dim == 0:
            raise ValueError("At least one feature branch must be active.")

        # ----------------------------------------------------
        # 2. Multimodal Fusion Layer
        # ----------------------------------------------------
        fusion_out_dim = 512
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_out_dim),
            nn.LayerNorm(fusion_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ----------------------------------------------------
        # 3. Bidirectional LSTM (Temporal Reasoning)
        # ----------------------------------------------------
        self.bilstm = nn.LSTM(
            input_size=fusion_out_dim,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )
        total_lstm_dim = bilstm_hidden * 2  # Forward + Backward

        # ----------------------------------------------------
        # 4. Temporal Aggregation (Attention vs Last State)
        # ----------------------------------------------------
        if self.use_attention:
            self.temporal_pooling = TemporalAttentionPooling(
                hidden_dim=total_lstm_dim,
                attention_dim=128,
            )

        # ----------------------------------------------------
        # 5. Classification Head
        # ----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(total_lstm_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        visual: Optional[torch.Tensor] = None,   # [B, T, visual_in_dim]
        pose: Optional[torch.Tensor] = None,     # [B, T, pose_in_dim]
        obj: Optional[torch.Tensor] = None,      # [B, T, object_in_dim]
        hoi: Optional[torch.Tensor] = None,      # [B, T, hoi_in_dim]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:

            visual: [B, T, 2048]
            pose:   [B, T, 85]
            obj:    [B, T, 7]
            hoi:    [B, T, 16]

        Returns:
            logits: [B, num_classes]
            attention_weights: [B, T] or None
        """
        tokens = []

        if self.use_visual:
            if visual is None:
                raise ValueError("visual stream is enabled but None was passed.")
            tokens.append(self.visual_branch(visual))

        if self.use_pose:
            if pose is None:
                raise ValueError("pose stream is enabled but None was passed.")
            tokens.append(self.pose_branch(pose))

        if self.use_object:
            if obj is None:
                raise ValueError("object stream is enabled but None was passed.")
            tokens.append(self.object_branch(obj))

        if self.use_hoi:
            if hoi is None:
                raise ValueError("hoi stream is enabled but None was passed.")
            tokens.append(self.hoi_branch(hoi))

        # Concatenate across feature dimension
        # tokens: list of [B, T, D_i] -> [B, T, Σ D_i]
        fused = torch.cat(tokens, dim=-1)

        # Project into unified multimodal representation: [B, T, 512]
        z = self.fusion_layer(fused)

        # Sequence modeling through BiLSTM: [B, T, total_lstm_dim]
        lstm_out, (h_n, c_n) = self.bilstm(z)

        # Temporal aggregation
        if self.use_attention:
            context, attn_weights = self.temporal_pooling(lstm_out)
        else:
            # Fallback: concatenate last forward and backward hidden states
            # h_n shape: [num_layers * 2, B, hidden_dim]
            last_fwd = h_n[-2, :, :]
            last_bwd = h_n[-1, :, :]
            context = torch.cat([last_fwd, last_bwd], dim=-1)
            attn_weights = None

        # Final classification logits: [B, num_classes]
        logits = self.classifier(context)

        return logits, attn_weights
