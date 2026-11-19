"""Cross-Modal Attention Transformer for Human-Object Interaction Sports Action Recognition (CMAT-HOI).

Key Architectural Innovations:
1. Unified Latent Embedding Stems with 1D Temporal Convolutions (d=256 for all 4 streams).
2. Bi-Directional Cross-Attention: Pose queries Object; Object queries Pose.
3. Gated Multimodal Fusion Unit (GMFU): Dynamic modality weighting preventing Visual dominance.
4. Spatio-Temporal Transformer Encoder: 3 Layers, 8 Attention Heads with learnable [CLS] token.
5. Multi-Scale Temporal Pooling: Combines [CLS] token, Temporal Attention, Max, and Avg pooling.
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal temporal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1), :]


class ModalityStem(nn.Module):
    """Projects raw input stream to unified latent dimension with local temporal convolution."""

    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.temporal_conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_dim]
        h = self.fc(x)
        h = self.norm(h)
        # Conv1d expects [B, D, T]
        h_conv = self.temporal_conv(h.transpose(1, 2)).transpose(1, 2)
        return self.drop(self.act(h + h_conv))


class BiDirectionalCrossAttention(nn.Module):
    """Cross-attention mechanism between human kinematic pose and sports equipment."""

    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.pose_to_obj_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.obj_to_pose_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_pose = nn.LayerNorm(d_model)
        self.norm_obj = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, pose: torch.Tensor, obj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pose queries Object
        p_out, _ = self.pose_to_obj_attn(query=pose, key=obj, value=obj)
        pose_enh = self.norm_pose(pose + self.drop(p_out))

        # Object queries Pose
        o_out, _ = self.obj_to_pose_attn(query=obj, key=pose, value=pose)
        obj_enh = self.norm_obj(obj + self.drop(o_out))

        return pose_enh, obj_enh


class GatedMultimodalFusion(nn.Module):
    """Dynamic sigmoid gating across all active modalities."""

    def __init__(self, num_modalities: int, d_model: int = 256, dropout: float = 0.2):
        super().__init__()
        self.num_modalities = num_modalities
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_modalities),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, modality_tokens: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # modality_tokens: list of [B, T, D]
        concat = torch.cat(modality_tokens, dim=-1)  # [B, T, D * M]
        logits = self.gate_net(concat)               # [B, T, M]
        gates = F.softmax(logits, dim=-1)            # [B, T, M]

        # Weighted combination across modalities
        stacked = torch.stack(modality_tokens, dim=2)  # [B, T, M, D]
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=2)  # [B, T, D]

        return self.out_norm(fused), gates


class HOICrossAttentionTransformer(nn.Module):
    """Advanced State-of-the-Art Multimodal Transformer for 82 Sports Action Recognition."""

    def __init__(
        self,
        num_classes: int = 82,
        visual_in_dim: int = 2048,
        pose_in_dim: int = 85,
        object_in_dim: int = 7,
        hoi_in_dim: int = 16,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        use_visual: bool = True,
        use_pose: bool = True,
        use_object: bool = True,
        use_hoi: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_visual = use_visual
        self.use_pose = use_pose
        self.use_object = use_object
        self.use_hoi = use_hoi
        self.use_attention = use_attention

        # 1. Modality Embedding Stems
        if self.use_visual:
            self.visual_stem = ModalityStem(visual_in_dim, d_model, dropout)
        if self.use_pose:
            self.pose_stem = ModalityStem(pose_in_dim, d_model, dropout)
        if self.use_object:
            self.object_stem = ModalityStem(object_in_dim, d_model, dropout)
        if self.use_hoi:
            self.hoi_stem = ModalityStem(hoi_in_dim, d_model, dropout)

        # 2. Bi-Directional Cross-Attention (Pose <-> Object)
        self.has_cross_attention = self.use_pose and self.use_object
        if self.has_cross_attention:
            self.cross_attention = BiDirectionalCrossAttention(d_model=d_model, num_heads=4, dropout=dropout)

        # 3. Gated Multimodal Fusion Unit
        active_modalities = sum([self.use_visual, self.use_pose, self.use_object, self.use_hoi])
        if active_modalities == 0:
            raise ValueError("At least one modality stream must be enabled.")
        self.fusion = GatedMultimodalFusion(num_modalities=active_modalities, d_model=d_model, dropout=dropout)

        # 4. Spatio-Temporal Positional Encoding & [CLS] Token
        self.pos_encoder = TemporalPositionalEncoding(d_model=d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 5. Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for stable optimization
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # 6. Multi-Head Temporal Attention Pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False),
        )

        # 7. Classification Head (Multi-Scale 4-Way Pooling: CLS + Attention + Max + Avg)
        # 256 * 4 = 1024 dims
        pooled_dim = d_model * 4 if self.use_attention else d_model * 3
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        visual: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        obj: Optional[torch.Tensor] = None,
        hoi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass across the 4 multimodal streams.
        
        Returns:
            logits: [B, num_classes]
            temporal_weights: [B, T] (Attention weights across frames)
        """
        modality_list = []

        if self.use_visual:
            v_tok = self.visual_stem(visual)
            modality_list.append(v_tok)

        p_tok = self.pose_stem(pose) if self.use_pose else None
        o_tok = self.object_stem(obj) if self.use_object else None

        # Cross-Attention between Pose and Sports Gear
        if self.has_cross_attention and p_tok is not None and o_tok is not None:
            p_enh, o_enh = self.cross_attention(p_tok, o_tok)
            modality_list.append(p_enh)
            modality_list.append(o_enh)
        else:
            if p_tok is not None:
                modality_list.append(p_tok)
            if o_tok is not None:
                modality_list.append(o_tok)

        if self.use_hoi:
            h_tok = self.hoi_stem(hoi)
            modality_list.append(h_tok)

        # Gated Multimodal Fusion: [B, T, d_model]
        fused_seq, mod_gates = self.fusion(modality_list)

        # Add Temporal Positional Encoding
        seq_with_pos = self.pos_encoder(fused_seq)

        # Prepend [CLS] token
        B = seq_with_pos.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        full_seq = torch.cat([cls_tokens, seq_with_pos], dim=1)  # [B, T + 1, d_model]

        # Pass through Transformer Encoder
        trans_out = self.transformer(full_seq)  # [B, T + 1, d_model]

        cls_out = trans_out[:, 0, :]           # [B, d_model]
        frame_feats = trans_out[:, 1:, :]      # [B, T, d_model]

        # Multi-scale temporal pooling
        avg_pool = frame_feats.mean(dim=1)     # [B, d_model]
        max_pool = frame_feats.max(dim=1)[0]   # [B, d_model]

        if self.use_attention:
            attn_energy = self.temporal_attention(frame_feats).squeeze(-1)  # [B, T]
            attn_weights = F.softmax(attn_energy, dim=-1)                   # [B, T]
            attn_pool = torch.bmm(attn_weights.unsqueeze(1), frame_feats).squeeze(1)  # [B, d_model]
            pooled_features = torch.cat([cls_out, attn_pool, avg_pool, max_pool], dim=-1)  # [B, 1024]
        else:
            attn_weights = None
            pooled_features = torch.cat([cls_out, avg_pool, max_pool], dim=-1)              # [B, 768]

        # Final classification logits: [B, num_classes]
        logits = self.head(pooled_features)

        return logits, attn_weights
