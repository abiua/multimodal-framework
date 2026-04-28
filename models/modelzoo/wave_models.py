"""Wave/时序 backbone 模型。

Stageable（支持中期融合）:
    tcn, resnet1d

非 stageable:
    patchtst, timesnet
"""

import math
import torch
import torch.nn as nn

from .common import ResBlock1D, PositionalEncoding, make_mlp
from ..registry import register_backbone
from ..backbone_base import StageableBackbone, BaseBackbone


# ==============================================================================
# TCN 组件
# ==============================================================================

class TemporalBlock(nn.Module):
    """TCN 膨胀卷积残差块。"""

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.padding = padding

    def forward(self, x):
        residual = x
        out = self.conv1(x)[:, :, :x.size(2)]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


# ==============================================================================
# Stageable Wave 模型
# ==============================================================================

@register_backbone('tcn', description='TCN 时序卷积特征提取器（支持 staged forward）', modality='wave')
class TCN(StageableBackbone):
    num_stages = 4

    def __init__(self, feature_dim=256, seq_len=512, in_channels=6,
                 hidden_channels=64, n_layers=6, kernel_size=3, dropout=0.2, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.stage_dims = [hidden_channels] * 4

        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels
            layers.append(TemporalBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))

        self.stages = nn.ModuleList([
            nn.Sequential(*layers[0:2]),
            nn.Sequential(*layers[2:4]),
            nn.Sequential(*layers[4:5]),
            nn.Sequential(*layers[5:6]),
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_channels, feature_dim)

    def init_state(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)  # [B, T, C] → [B, C, T]
        return x

    def forward_stage(self, state, stage_idx):
        return self.stages[stage_idx](state)

    def forward_head(self, state):
        return self.proj(self.pool(state).squeeze(-1))


@register_backbone('tcn_stageable', description='Stageable TCN（别名，同 tcn）', modality='wave')
class TCNStageable(TCN):
    """向后兼容别名，功能与 TCN 完全相同。"""


@register_backbone('resnet1d', description='1D-ResNet 时序特征提取器（支持 staged forward）', modality='wave')
class ResNet1D(StageableBackbone):
    num_stages = 4

    def __init__(self, feature_dim=256, in_channels=6, dropout=0.2, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.stage_dims = [64, 64, 128, 256]

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.stages = nn.ModuleList([
            ResBlock1D(64, 64),
            ResBlock1D(64, 128, stride=2),
            ResBlock1D(128, 256, stride=2),
            nn.Identity(),
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = make_mlp(256, feature_dim, dropout=dropout, final_activation=True)

    def init_state(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)  # [B, T, C] → [B, C, T]
        return self.stem(x)

    def forward_stage(self, state, stage_idx):
        return self.stages[stage_idx](state)

    def forward_head(self, state):
        return self.proj(self.pool(state).squeeze(-1))


# ==============================================================================
# 非 stageable Wave 模型
# ==============================================================================

@register_backbone('patchtst', description='PatchTST 时序特征提取器 (ICLR 2023)', modality='wave')
class PatchTST(BaseBackbone):
    def __init__(self, feature_dim=256, seq_len=512, patch_size=16,
                 in_channels=6, d_model=128, n_heads=8, n_layers=6,
                 dim_feedforward=512, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.d_model = d_model

        self.patch_proj = nn.Linear(patch_size * in_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, feature_dim) if feature_dim != d_model else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            x = x.transpose(1, 2)  # ensure [B, L, C]

        B, L, C = x.shape
        x = x.reshape(B, self.num_patches, self.patch_size * C)
        x = self.patch_proj(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.norm(x)
        return self.proj(x[:, 0])


@register_backbone('timesnet', description='TimesNet 时序特征提取器 (ICLR 2023)', modality='wave')
class TimesNet(BaseBackbone):
    def __init__(self, feature_dim=256, seq_len=512, in_channels=6,
                 d_model=128, n_layers=4, top_k=5, num_kernels=6,
                 dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.seq_len = seq_len

        self.enc_embedding = nn.Linear(in_channels, d_model)
        self.layers = nn.ModuleList([
            _TimesBlock(d_model, seq_len, num_kernels) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.period_conv = nn.Conv1d(d_model, top_k, kernel_size=3, padding=1)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            x = x.transpose(1, 2)  # ensure [B, L, C]

        B, L, C = x.shape
        x = self.enc_embedding(x)

        x_freq = x.transpose(1, 2)
        fft_out = torch.fft.rfft(x_freq, dim=2)
        fft_amp = torch.abs(fft_out)
        period_weights = self.period_conv(fft_amp).mean(dim=2)
        top_k_periods = torch.topk(period_weights, self.top_k, dim=1).indices + 2

        for layer in self.layers:
            x = layer(x, top_k_periods[0].tolist())

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.proj(x)


class _TimesBlock(nn.Module):
    """TimesNet 核心模块。"""

    def __init__(self, d_model, seq_len, num_kernels=6):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model), nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
        )

    def forward(self, x, period_list):
        B, L, D = x.shape
        res = torch.zeros_like(x)
        count = 1

        for period in period_list:
            if period < 2:
                continue
            num_periods = L // period
            if num_periods < 1:
                continue
            pad_len = period * num_periods - L
            x_pad = x[:, :period * num_periods, :] if pad_len <= 0 else \
                torch.cat([x, torch.zeros(B, pad_len, D, device=x.device)], dim=1)

            x_2d = x_pad.reshape(B, num_periods, period, D).permute(0, 3, 1, 2)
            x_conv = self.conv(x_2d)
            x_out = x_conv.permute(0, 2, 3, 1).reshape(B, -1, D)[:, :L, :]
            if x_out.size(1) < L:
                x_out = nn.functional.pad(x_out, (0, 0, 0, L - x_out.size(1)))
            res = res + x_out
            count += 1

        return (res + x) / count
