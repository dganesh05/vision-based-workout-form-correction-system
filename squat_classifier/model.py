from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LuongDotAttention(nn.Module):
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encoder_outputs: (B, T, H), query: (B, H), mask: (B, T)
        scores = torch.einsum("bth,bh->bt", encoder_outputs, query)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.einsum("bt,bth->bh", weights, encoder_outputs)
        return context, weights


class BiCGRUClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dims: int = 3,
        spatial_channels: int = 64,
        gru_hidden_size: int = 128,
        gru_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        # Spatial feature extraction over joints for each frame.
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(input_dims, spatial_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(spatial_channels, spatial_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.temporal_gru = nn.GRU(
            input_size=spatial_channels,
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
        sequences: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sequences: (B, T, J, C)
        bsz, timesteps, joints, channels = sequences.shape

        x = sequences.permute(0, 1, 3, 2).reshape(bsz * timesteps, channels, joints)
        x = self.spatial_conv(x).squeeze(-1)  # (B*T, spatial_channels)
        x = x.view(bsz, timesteps, -1)

        packed = pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.temporal_gru(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=timesteps)

        # Last-layer hidden states from forward and backward GRU.
        forward_last = hidden[-2]
        backward_last = hidden[-1]
        query = torch.cat([forward_last, backward_last], dim=-1)

        context, weights = self.attention(encoder_outputs=outputs, query=query, mask=mask)
        logits = self.classifier(context)
        return logits, weights

