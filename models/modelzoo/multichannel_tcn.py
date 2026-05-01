"""Multi-Channel TCN — 三通道IMU编码器。

每个物理通道（accel/gyro/angle）独立 stem conv，
然后 concat 并通过共享多层 TCN，输出统一 IMU token 序列。
"""
import torch
import torch.nn as nn
from ..registry import register_backbone
from ..backbone_base import BaseBackbone


class ChannelStem(nn.Module):
    """单通道 stem: Conv1d + BN + ReLU"""

    def __init__(self, in_channels=3, out_channels=64, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, T, 3] → [B, 3, T] → [B, C_ch, T]
        return self.act(self.bn(self.conv(x.transpose(1, 2))))


class TemporalBlock(nn.Module):
    """膨胀卷积残差块"""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

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


@register_backbone('multichannel_tcn', description='Multi-channel TCN for IMU encoding', modality='imu')
class MultiChannelTCN(BaseBackbone):
    """三通道 IMU 编码器。

    输入: 三个通道各有 [B, T, 3] 的原始传感器数据
    输出: [B, T, output_dim] IMU token 序列
    """

    def __init__(self, channel_dim=64, tcn_channels=None, kernel_size=7,
                 dropout=0.2, output_dim=256, **kwargs):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [128, 256, 256]
        self.feature_dim = output_dim
        self.output_dim = output_dim

        self.stems = nn.ModuleDict({
            'accel': ChannelStem(3, channel_dim, kernel_size),
            'gyro': ChannelStem(3, channel_dim, kernel_size),
            'angle': ChannelStem(3, channel_dim, kernel_size),
        })

        in_ch = channel_dim * 3
        layers = []
        for i, ch in enumerate(tcn_channels):
            layers.append(TemporalBlock(in_ch, ch, kernel_size, 2 ** i, dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)

        self.output_proj = nn.Sequential(
            nn.Conv1d(tcn_channels[-1], output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, imu_channels: dict) -> torch.Tensor:
        stem_outs = [self.stems[ch](imu_channels[ch]) for ch in ['accel', 'gyro', 'angle']]
        concat = torch.cat(stem_outs, dim=1)       # [B, 3*C_ch, T]
        tcn_out = self.tcn(concat)                  # [B, tcn_out_ch, T]
        return self.output_proj(tcn_out).transpose(1, 2)  # [B, T, output_dim]

    def tokenize(self, imu_channels: dict) -> dict:
        tokens = self.forward(imu_channels)
        return {"tokens": tokens, "layout": "1d"}
