"""
AudioTemporalEncoder — CNN audio encoder that preserves the time dimension.

Unlike AudioCNN which pools to a single global token, this encoder keeps the
time axis so we get [B, T, D] output, suitable for cross-attention with
waveform temporal sequences.

Architecture:
  conv1:   in_c → 32,  BN, ReLU, MaxPool2d(2,2)   — halve both dims
  conv2:   32 → 64,    BN, ReLU, MaxPool2d(2,2)
  conv3:   64 → 128,   BN, ReLU, MaxPool2d(2,2)
  conv4:   128 → 256,  BN, ReLU                    — no pooling
  freq_pool: AdaptiveAvgPool2d((1, None))          — pool freq to 1
  proj:    Conv2d(256, output_dim, 1), BN, ReLU, Dropout
  output:  squeeze freq dim → [B, D, T] → transpose → [B, T, D]
"""

import torch
import torch.nn as nn


class AudioTemporalEncoder(nn.Module):
    """CNN-based audio encoder that preserves the time dimension.

    Input:  [B, in_channels, n_mels, time_steps]  (mel spectrogram)
    Output: [B, T_out, output_dim]                 (temporal tokens)

    Args:
        in_channels: Number of input channels (default: 2)
        output_dim:  Output feature dimension (default: 256)
        dropout:     Dropout probability (default: 0.1)
    """

    def __init__(self, in_channels: int = 2, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = output_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Sequential(
            nn.Conv2d(256, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, in_channels, n_mels, time_steps]

        Returns:
            Tensor of shape [B, T_out, output_dim]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.freq_pool(x)      # [B, 256, 1, T]
        x = self.proj(x)           # [B, output_dim, 1, T]
        x = x.squeeze(2)           # [B, output_dim, T]
        x = x.transpose(1, 2)      # [B, T, output_dim]
        return x
