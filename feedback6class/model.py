"""
BiCGRU classifier — optimized for SMALL datasets
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ─────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────

class LuongDotAttention(nn.Module):
    """Scaled dot-product attention over GRU outputs."""

    def forward(
        self,
        encoder_outputs: torch.Tensor,   # (B, T, H)
        query: torch.Tensor,             # (B, H)
        mask: torch.Tensor,             # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:

        H = encoder_outputs.size(-1)

        scores = torch.einsum("bth,bh->bt", encoder_outputs, query) / (H ** 0.5)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=1)
        context = torch.einsum("bt,bth->bh", weights, encoder_outputs)

        return context, weights


# ─────────────────────────────────────────────────────────────
# Spatial CNN (lighter)
# ─────────────────────────────────────────────────────────────

class SpatialCNN(nn.Module):

    def __init__(self, input_dims: int = 3, out_channels: int = 32):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# Angle MLP (simplified)
# ─────────────────────────────────────────────────────────────

class AngleMLP(nn.Module):

    def __init__(self, num_angles: int = 5 #change to 5 later
                 , out_channels: int = 16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_angles, out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────

class BiCGRUClassifier(nn.Module):

    def __init__(
        self,
        *,
        input_dims: int = 3,
        num_angles: int = 5, #change to 5 later
        num_classes: int = 6,
        num_joints: int = 7,
        gru_layers: int = 1,
        spatial_channels: int = 32,
        angle_channels: int = 16,
        gru_hidden_size: int = 12,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ── Feature extractors ─────────────────────────────
        self.spatial_cnn = SpatialCNN(input_dims, spatial_channels)
        self.angle_mlp = AngleMLP(num_angles, angle_channels)

        fused_dim = spatial_channels + angle_channels

        # simple fusion (important for small data)
        self.fusion_proj = nn.Linear(fused_dim, fused_dim)

        # ── Temporal model ────────────────────────────────
        self.gru = nn.GRU(
            input_size=fused_dim,
            hidden_size=gru_hidden_size,
            num_layers=1,        # KEEP SMALL
            batch_first=True,
            bidirectional=True,
        )

        self.norm = nn.LayerNorm(gru_hidden_size * 2)

        self.attention = LuongDotAttention()

        # ── Classifier head (small) ───────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, gru_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size, num_classes),
        )

    # ─────────────────────────────────────────────────────
    def forward(self, joints, angles, lengths, mask):

        B, T, J, C = joints.shape

        # ── Spatial features ─────────────────────────────
        j = joints.permute(0, 1, 3, 2).reshape(B * T, C, J)
        spatial = self.spatial_cnn(j)

        # ── Angle features ───────────────────────────────
        a = angles.reshape(B * T, -1)
        angle = self.angle_mlp(a)

        # ── Fusion ────────────────────────────────────────
        fused = torch.cat([spatial, angle], dim=-1)
        fused = self.fusion_proj(fused)
        fused = fused.view(B, T, -1)

        # ── GRU ───────────────────────────────────────────
        packed = pack_padded_sequence(
            fused,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=T
        )

        outputs = self.norm(outputs)

        # regularization (ONLY here)
        outputs = torch.dropout(outputs, p=0.2, train=self.training)

        # ── Query from final hidden state ────────────────
        query = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        # ── Attention ─────────────────────────────────────
        context, attn = self.attention(outputs, query, mask)

        # ── Classification ───────────────────────────────
        logits = self.classifier(context)

        return logits, attn

    # ─────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_features(self, joints, angles, lengths, mask):

        B, T, J, C = joints.shape

        j = joints.permute(0, 1, 3, 2).reshape(B * T, C, J)
        spatial = self.spatial_cnn(j)

        a = angles.reshape(B * T, -1)
        angle = self.angle_mlp(a)

        fused = torch.cat([spatial, angle], dim=-1)
        fused = self.fusion_proj(fused)
        fused = fused.view(B, T, -1)

        packed = pack_padded_sequence(
            fused,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=T
        )

        outputs = self.norm(outputs)

        query = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        context, attn = self.attention(outputs, query, mask)

        return context, outputs, attn