"""AudioCNN6 — small 6-layer CNN for log-mel spectrogram classification (~1.2M params).

Designed for frequency ablation experiments where model capacity should be
kept small to avoid absorbing differences in input frequency content.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import ensure_4d
from ..registry import register_backbone
from ..backbone_base import BaseBackbone


@register_backbone('audio_cnn6', description='CNN6-lite log-mel spectrogram classifier (~1.2M)', modality='audio')
class AudioCNN6(BaseBackbone):
    """Six conv layers with BatchNorm+ReLU, MaxPool every 2 layers, GlobalAvgPool, FC head.

    Channels: [64, 64, 128, 128, 256, 256] — ~1.15M params with (1, 128, 224) input.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_mels: int = 128,
        time_steps: int = 224,
        channels: tuple = (64, 64, 128, 128, 256, 256),
        dropout: float = 0.1,
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        layers: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if i % 2 == 1:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = ensure_4d(x)
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
