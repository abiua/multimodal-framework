"""Wave/时序模型 - 水面震动等传感器数据特征提取器"""

import torch
import torch.nn as nn
import math
from ..registry import register_backbone


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# PatchTST - Patch Time Series Transformer (2023 ICLR SOTA)
# =============================================================================
class PatchEmbedding(nn.Module):
    """PatchTST的Patch嵌入层"""
    def __init__(self, seq_len, patch_size, in_channels, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.d_model = d_model

        # 线性投影: 将每个patch映射到d_model维度
        self.proj = nn.Linear(patch_size * in_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, seq_len, channels)
        B, L, C = x.shape
        # 分割成patch: (B, num_patches, patch_size * channels)
        x = x.reshape(B, self.num_patches, self.patch_size * C)
        # 线性投影: (B, num_patches, d_model)
        x = self.proj(x)
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # 位置编码
        x = self.pos_encoding(x)
        return x


@register_backbone('patchtst', description='PatchTST时序特征提取器 (2023 ICLR SOTA)', modality='wave')
class PatchTST(nn.Module):
    """Patch Time Series Transformer - 用于时间序列特征提取

    论文: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)

    Args:
        feature_dim: 输出特征维度
        seq_len: 输入序列长度
        patch_size: patch大小
        in_channels: 输入通道数(传感器轴数)
        d_model: Transformer隐藏维度
        n_heads: 注意力头数
        n_layers: Transformer层数
        dropout: Dropout比率
    """
    def __init__(self, feature_dim=256, seq_len=512, patch_size=16,
                 in_channels=6, d_model=128, n_heads=8, n_layers=6,
                 dim_feedforward=512, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        # Patch嵌入
        self.patch_embed = PatchEmbedding(seq_len, patch_size, in_channels, d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # 特征投影
        self.proj = nn.Linear(d_model, feature_dim) if feature_dim != d_model else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, seq_len, channels) 或 (B, channels, seq_len)
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            x = x.transpose(1, 2)  # 确保 (B, L, C)

        # Patch嵌入
        x = self.patch_embed(x)  # (B, num_patches+1, d_model)

        # Transformer编码
        x = self.encoder(x)
        x = self.norm(x)

        # 取CLS token作为序列表示
        x = x[:, 0, :]  # (B, d_model)
        return self.proj(x)


# =============================================================================
# TimesNet - 2023 ICLR 另一个时序SOTA
# =============================================================================
class TimesBlock(nn.Module):
    """TimesNet的核心模块 - 2D卷积处理多周期模式"""
    def __init__(self, d_model, seq_len, num_kernels=6):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # 自适应周期检测
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
        )

    def forward(self, x, period_list):
        # x: (B, L, D)
        B, L, D = x.shape
        res = torch.zeros_like(x)
        count = 1  # 包括原始输入

        for period in period_list:
            if period < 2:
                continue
            # 重塑为2D: (B, D, period, L//period)
            num_periods = L // period
            if num_periods < 1:
                continue
            pad_len = period * num_periods - L
            if pad_len > 0:
                x_pad = torch.cat([x, torch.zeros(B, pad_len, D, device=x.device)], dim=1)
            else:
                x_pad = x[:, :period * num_periods, :]

            x_2d = x_pad.reshape(B, num_periods, period, D).permute(0, 3, 1, 2)

            # 2D卷积
            x_conv = self.conv(x_2d)

            # 恢复形状并确保长度匹配
            x_out = x_conv.permute(0, 2, 3, 1).reshape(B, -1, D)
            x_out = x_out[:, :L, :]  # 截断到原始长度
            if x_out.size(1) < L:
                # 填充到原始长度
                x_out = nn.functional.pad(x_out, (0, 0, 0, L - x_out.size(1)))

            res = res + x_out
            count += 1

        return (res + x) / count


@register_backbone('timesnet', description='TimesNet时序特征提取器 (2023 ICLR SOTA)', modality='wave')
class TimesNet(nn.Module):
    """TimesNet - 多周期时序特征提取器

    论文: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (ICLR 2023)

    Args:
        feature_dim: 输出特征维度
        seq_len: 输入序列长度
        in_channels: 输入通道数
        d_model: 模型隐藏维度
        n_layers: TimesBlock层数
        top_k: 使用的top-k周期数
        num_kernels: 卷积核数量
        dropout: Dropout比率
    """
    def __init__(self, feature_dim=256, seq_len=512, in_channels=6,
                 d_model=128, n_layers=4, top_k=5, num_kernels=6,
                 dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.seq_len = seq_len

        # 输入投影
        self.enc_embedding = nn.Linear(in_channels, d_model)

        # TimesBlock层
        self.layers = nn.ModuleList([
            TimesBlock(d_model, seq_len, num_kernels) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 自适应周期池
        self.period_conv = nn.Conv1d(d_model, top_k, kernel_size=3, padding=1)

        # 输出投影
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, seq_len, channels) 或 (B, channels, seq_len)
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            x = x.transpose(1, 2)

        B, L, C = x.shape

        # 输入嵌入
        x = self.enc_embedding(x)  # (B, L, d_model)

        # FFT检测周期
        x_freq = x.transpose(1, 2)  # (B, d_model, L)
        fft_out = torch.fft.rfft(x_freq, dim=2)
        fft_amp = torch.abs(fft_out)
        period_weights = self.period_conv(fft_amp)  # (B, top_k, L//2+1)
        period_weights = period_weights.mean(dim=2)  # (B, top_k)
        top_k_periods = torch.topk(period_weights, self.top_k, dim=1).indices + 2  # +2避免周期1

        # TimesBlock处理
        res = []
        for layer in self.layers:
            x = layer(x, top_k_periods[0].tolist())  # 使用batch0的周期估计
            res.append(x)

        x = self.norm(x)

        # 全局平均池化
        x = x.mean(dim=1)  # (B, d_model)
        return self.proj(x)


# =============================================================================
# 1D-TCN - 轻量级时序卷积网络
# =============================================================================
class TemporalBlock(nn.Module):
    """TCN残差块"""
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.padding = padding

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # 因果卷积
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)
    
@register_backbone('tcn_stageable', description='Stageable TCN', modality='wave')
class TCNStageable(nn.Module):
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

        # 6层 -> 4个stage
        self.stages = nn.ModuleList([
            nn.Sequential(*layers[0:2]),
            nn.Sequential(*layers[2:4]),
            nn.Sequential(*layers[4:5]),
            nn.Sequential(*layers[5:6]),
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_channels, feature_dim)

    def init_state(self, x):
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)   # [B, C, T]
        return x

    def forward_stage(self, state, stage_idx: int):
        return self.stages[stage_idx](state)

    def forward_head(self, state):
        x = self.pool(state).squeeze(-1)
        return self.proj(x)

    def forward(self, x):
        state = self.init_state(x)
        for stage_idx in range(self.num_stages):
            state = self.forward_stage(state, stage_idx)
        return self.forward_head(state)

@register_backbone('tcn', description='TCN时序卷积特征提取器', modality='wave')
class TCN(nn.Module):
    """Temporal Convolutional Network - 轻量级时序特征提取器

    Args:
        feature_dim: 输出特征维度
        seq_len: 输入序列长度
        in_channels: 输入通道数
        hidden_channels: 隐藏层通道数
        n_layers: TCN层数
        kernel_size: 卷积核大小
        dropout: Dropout比率
    """
    def __init__(self, feature_dim=256, seq_len=512, in_channels=6,
                 hidden_channels=64, n_layers=6, kernel_size=3,
                 dropout=0.2, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels
            layers.append(TemporalBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_channels, feature_dim)

    def forward(self, x):
        # x: (B, seq_len, channels) 或 (B, channels, seq_len)
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)  # (B, channels, seq_len)

        x = self.tcn(x)  # (B, hidden, seq_len)
        x = self.pool(x).squeeze(-1)  # (B, hidden)
        return self.proj(x)


# =============================================================================
# Wave-ResNet1D - 1D残差网络
# =============================================================================
class ResBlock1D(nn.Module):
    """1D残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


@register_backbone('resnet1d', description='1D-ResNet时序特征提取器', modality='wave')
class ResNet1D(nn.Module):
    """1D ResNet - 适用于波形/传感器数据的特征提取器

    Args:
        feature_dim: 输出特征维度
        in_channels: 输入通道数
        dropout: Dropout比率
    """
    def __init__(self, feature_dim=256, in_channels=6, dropout=0.2, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            ResBlock1D(64, 64),
            ResBlock1D(64, 128, stride=2),
            ResBlock1D(128, 256, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )

        self.proj = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, seq_len, channels) 或 (B, channels, seq_len)
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)

        x = self.features(x).squeeze(-1)
        return self.proj(x)
