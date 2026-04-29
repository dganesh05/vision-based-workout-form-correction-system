"""
model.py — BiCGRU classifier with dual-modality input for multi-class squat form analysis.

Architecture overview
─────────────────────
Input:
  joints  (B, T, J, 3)  — normalised 3-D joint positions
  angles  (B, T, 5)     — pre-computed biomechanical angles (zeros if unavailable)

1. Spatial CNN  — per-frame: Conv1d across J joints → spatial embedding (B*T, spatial_channels)
2. Angle MLP    — per-frame: MLP on 5 angles         → angle embedding   (B*T, angle_channels)
3. Fusion       — concatenate → projection           → (B, T, gru_input)
4. Bi-GRU       — stacked bidirectional GRU          → (B, T, 2*hidden)
5. Attention    — Luong dot-product on GRU outputs   → context (B, 2*hidden)
6. Classifier   — two-layer MLP + softmax            → (B, num_classes)

The attention weights (B, T) are returned alongside logits so you can
visualise which frames drove the prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ── Attention ──────────────────────────────────────────────────────────────────

class LuongDotAttention(nn.Module):
    """Scaled dot-product attention over GRU encoder outputs."""

    def forward(
        self,
        encoder_outputs: torch.Tensor,   # (B, T, H)
        query: torch.Tensor,              # (B, H)
        mask: torch.Tensor,               # (B, T) bool — True where valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H = encoder_outputs.size(-1)
        scores = torch.einsum("bth,bh->bt", encoder_outputs, query) / (H ** 0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.einsum("bt,bth->bh", weights, encoder_outputs)
        return context, weights


# ── Sub-networks ───────────────────────────────────────────────────────────────

class SpatialCNN(nn.Module):
    """
    Per-frame spatial feature extractor.
    Input : (B*T, input_dims, J)  — channels-first for Conv1d
    Output: (B*T, out_channels)
    """

    def __init__(self, input_dims: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dims, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AngleMLP(nn.Module):
    """
    Per-frame angle feature extractor.
    Input : (B*T, num_angles)
    Output: (B*T, out_channels)
    """

    def __init__(self, num_angles: int = 5, out_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_angles, out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Main classifier ────────────────────────────────────────────────────────────

class BiCGRUClassifier(nn.Module):
    """
    Dual-modality (3-D joints + biomechanical angles) squat form classifier.

    Parameters
    ----------
    input_dims      : spatial dims per joint (typically 3 for x,y,z)
    num_joints      : number of tracked joints
    num_angles      : number of pre-computed angle features (default 5)
    spatial_channels: CNN output width
    angle_channels  : angle MLP output width
    gru_hidden_size : hidden size per GRU direction
    gru_layers      : number of stacked GRU layers
    dropout         : dropout rate (applied between GRU layers and in classifier)
    num_classes     : number of form categories to predict
    """

    def __init__(
        self,
        *,
        input_dims: int = 3,
        num_joints: int = 7,
        num_angles: int = 5,
        spatial_channels: int = 64,
        angle_channels: int = 32,
        gru_hidden_size: int = 128,
        gru_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.spatial_cnn = SpatialCNN(input_dims=input_dims, out_channels=spatial_channels)
        self.angle_mlp = AngleMLP(num_angles=num_angles, out_channels=angle_channels)

        fused_dim = spatial_channels + angle_channels
        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.temporal_gru = nn.GRU(
            input_size=fused_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = LuongDotAttention()

        temporal_out = gru_hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(temporal_out, temporal_out),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(temporal_out, num_classes),
        )

    def forward(
        self,
        joints: torch.Tensor,    # (B, T, J, C)
        angles: torch.Tensor,    # (B, T, 5)
        lengths: torch.Tensor,   # (B,)
        mask: torch.Tensor,      # (B, T) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits  : (B, num_classes)
            attn_w  : (B, T)  attention weights over frames
        """
        B, T, J, C = joints.shape

        # ── Spatial features (per frame) ──────────────────────────────────────
        # Reshape to (B*T, C, J) for Conv1d
        j_flat = joints.permute(0, 1, 3, 2).reshape(B * T, C, J)
        spatial_feat = self.spatial_cnn(j_flat)          # (B*T, spatial_channels)

        # ── Angle features (per frame) ────────────────────────────────────────
        a_flat = angles.reshape(B * T, -1)               # (B*T, num_angles)
        angle_feat = self.angle_mlp(a_flat)              # (B*T, angle_channels)

        # ── Fusion ────────────────────────────────────────────────────────────
        fused = torch.cat([spatial_feat, angle_feat], dim=-1)   # (B*T, fused_dim)
        fused = self.fusion_proj(fused)
        fused = fused.view(B, T, -1)                    # (B, T, fused_dim)

        # ── Temporal GRU ──────────────────────────────────────────────────────
        packed = pack_padded_sequence(
            fused, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.temporal_gru(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        # (B, T, 2*hidden)

        # Query = concat of last-layer forward & backward hidden states
        query = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # (B, 2*hidden)

        # ── Attention + classify ──────────────────────────────────────────────
        context, attn_weights = self.attention(outputs, query, mask)
        logits = self.classifier(context)
        return logits, attn_weights
