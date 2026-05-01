"""ModelZoo 共享工具模块。

提供:
- 位置编码、残差块等基础组件
- TorchvisionWrapper: 通用 torchvision 模型 → BaseBackbone
- TorchvisionStageable: 通用 torchvision ResNet → StageableBackbone
- HuggingFaceWrapper: 通用 HF Transformer → BaseBackbone
- 辅助函数: ensure_4d, make_mlp, build_fallback_cnn
"""

import math
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone_base import BaseBackbone, StageableBackbone
from ..registry import register_backbone


# ==============================================================================
# 基础组件
# ==============================================================================

class PositionalEncoding(nn.Module):
    """正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ResBlock2D(nn.Module):
    """2D 残差块（用于音频频谱图等）。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResBlock1D(nn.Module):
    """1D 残差块（用于波形/时序数据）。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# ==============================================================================
# Fallback 实现
# ==============================================================================

class SimpleViT(nn.Module):
    """简化 ViT 实现，作为 torchvision ViT 不可用时的 fallback。"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class SimpleTransformerEncoder(nn.Module):
    """简化 Transformer 编码器，作为 HuggingFace 模型不可用时的 fallback。"""

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        max_len: int = 512,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        src_key_padding_mask = None if attention_mask is None else (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x, x[:, 0, :]


# ==============================================================================
# 辅助函数
# ==============================================================================

def ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """将 [B, H, W] 转为 [B, 1, H, W]，保持 [B, C, H, W] 不变。"""
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.0,
    final_activation: bool = False,
) -> nn.Sequential:
    """构建标准 MLP。

    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        hidden_dims: 隐藏层维度列表，如 [512, 256]
        dropout: dropout 比率
        final_activation: 是否在最后一层后添加 ReLU
    """
    layers: List[nn.Module] = []
    dims = [in_dim] + (hidden_dims or []) + [out_dim]

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def build_fallback_cnn(
    in_channels: int,
    out_dim: int,
    channels: Optional[List[int]] = None,
) -> nn.Sequential:
    """构建简单的 CNN fallback（用于 torchvision 模型不可用时）。

    Args:
        in_channels: 输入通道数
        out_dim: 输出特征维度
        channels: 中间通道数列表，默认 [32, 64]
    """
    if channels is None:
        channels = [32, 64]
    seq: List[nn.Module] = []
    prev = in_channels
    for ch in channels:
        seq.extend([
            nn.Conv2d(prev, ch, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ])
        prev = ch
    seq.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*seq)


# ==============================================================================
# Torchvision 包装器
# ==============================================================================

class TorchvisionWrapper(BaseBackbone):
    """通用 torchvision 模型包装器（非 stageable）。

    用于 ViT, Swin, DeiT, EfficientNet, MobileNet, ConvNeXt 等模型。
    """

    def __init__(
        self,
        feature_dim: int,
        pretrained: bool,
        model_fn: Callable[..., nn.Module],
        default_dim: int,
        head_attr: Optional[str] = None,
        in_channels: int = 3,
        fallback_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        try:
            model = model_fn(pretrained=pretrained)
            if head_attr is not None:
                setattr(model, head_attr, nn.Identity())
            self.backbone = model
            self._is_fallback = False
        except Exception:
            self.backbone = build_fallback_cnn(in_channels, default_dim, fallback_channels)
            self._is_fallback = True
            if self._is_fallback:
                default_dim = fallback_channels[-1] if fallback_channels else 64

        self.proj = nn.Linear(default_dim, feature_dim) if feature_dim != default_dim else nn.Identity()

    def forward(self, x=None, **inputs) -> torch.Tensor:
        if x is not None:
            inputs.setdefault('x', x)
        x = inputs.get("x", inputs.get("image", None))
        if x is None:
            # 取第一个 tensor 值
            x = next(v for v in inputs.values() if isinstance(v, torch.Tensor))

        x = self.backbone(x)

        # 处理不同模型的输出格式
        if isinstance(x, torch.Tensor):
            if x.dim() == 4:
                x = x.flatten(1)
            elif x.dim() > 2:
                x = x[:, 0, :] if x.size(1) > 1 else x.squeeze(1)
        elif hasattr(x, 'logits'):
            x = x.logits
        elif isinstance(x, (tuple, list)):
            # HuggingFace 风格: (last_hidden_state, pooled_output)
            x = x[1] if len(x) > 1 and x[1].dim() == 2 else x[0][:, 0, :]

        return self.proj(x)


class TorchvisionStageable(StageableBackbone):
    """通用 ResNet 系列 torchvision 模型 →
     StageableBackbone。

    专门用于有明确 layer1-4 结构的 ResNet 系列模型。
    """

    num_stages = 4

    def __init__(
        self,
        feature_dim: int,
        pretrained: bool,
        model_fn: Callable[..., nn.Module],
        default_dim: int,
        stage_dims: List[int],
        in_channels: int = 3,
        fallback_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.stage_dims = list(stage_dims)

        try:
            backbone = model_fn(pretrained=pretrained)

            self.stem = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
            )
            self.stages = nn.ModuleList([
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ])
            self.pool = backbone.avgpool
            self.proj = nn.Linear(default_dim, feature_dim) if feature_dim != default_dim else nn.Identity()
            self._is_fallback = False
        except Exception:
            # ResNet 不可用时回退到简单 CNN + 4 个 identity stage
            fallback_ch = fallback_channels or [32, 64, 128, 128]
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, fallback_ch[0], 3, padding=1),
                nn.ReLU(),
            )
            self.stages = nn.ModuleList([
                nn.Sequential(nn.Conv2d(fallback_ch[0], fallback_ch[0], 3, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(fallback_ch[0], fallback_ch[1], 3, stride=2, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(fallback_ch[1], fallback_ch[2], 3, stride=2, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(fallback_ch[2], fallback_ch[3], 3, stride=2, padding=1), nn.ReLU()),
            ])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.stage_dims = fallback_ch
            self.proj = nn.Linear(fallback_ch[-1], feature_dim)
            self._is_fallback = True

    def init_state(self, x: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        return self.stem(x)

    def forward_stage(self, state: torch.Tensor, stage_idx: int) -> torch.Tensor:
        return self.stages[stage_idx](state)

    def forward_head(self, state: torch.Tensor) -> torch.Tensor:
        x = self.pool(state)
        x = torch.flatten(x, 1)
        return self.proj(x)


# ==============================================================================
# HuggingFace 包装器
# ==============================================================================

class HuggingFaceWrapper(BaseBackbone):
    """通用 HuggingFace Transformer 模型包装器。

    用于 BERT, RoBERTa, DistilBERT, ALBERT 等文本模型。
    """

    def __init__(
        self,
        feature_dim: int,
        pretrained: bool,
        model_name: str,
        model_cls_name: str,
        config_cls_name: str,
        default_dim: int = 768,
        fallback_vocab_size: int = 30000,
        fallback_embed_dim: int = 768,
        fallback_num_heads: int = 12,
        fallback_num_layers: int = 12,
        fallback_dim_feedforward: int = 3072,
        fallback_pad_idx: int = 0,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        try:
            from transformers import AutoModel, AutoConfig
            if pretrained:
                self.transformer = AutoModel.from_pretrained(model_name)
            else:
                config = AutoConfig.from_pretrained(model_name)
                self.transformer = AutoModel.from_config(config)
        except Exception:
            self.transformer = SimpleTransformerEncoder(
                vocab_size=fallback_vocab_size,
                embed_dim=fallback_embed_dim,
                num_heads=fallback_num_heads,
                num_layers=fallback_num_layers,
                dim_feedforward=fallback_dim_feedforward,
                pad_idx=fallback_pad_idx,
            )

        self.proj = nn.Linear(default_dim, feature_dim) if feature_dim != default_dim else nn.Identity()

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if input_ids is None:
            input_ids = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))

        if isinstance(self.transformer, SimpleTransformerEncoder):
            _, pooled = self.transformer(input_ids, attention_mask)
            return self.proj(pooled)

        try:
            outputs = self.transformer(input_ids, attention_mask=attention_mask)
            pooled = getattr(outputs, 'pooler_output', None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0, :]
        except Exception:
            outputs = self.transformer(input_ids, attention_mask=attention_mask)
            if isinstance(outputs, tuple):
                pooled = outputs[1] if outputs[1].dim() == 2 else outputs[0][:, 0, :]
            else:
                pooled = outputs[:, 0, :]

        return self.proj(pooled)


# ==============================================================================
# Identity Stem
# ==============================================================================

@register_backbone('identity_stem', description='Pass-through stem for raw tensor inputs', modality='any')
class IdentityStem(BaseBackbone):
    """Identity stem -- pass-through raw tensors, used before MultiChannelTCN for IMU channels.

    IMU channel loader already outputs normalized [T, 3] data,
    no additional feature extraction is needed.
    Actual encoding is done by MultiChannelTCN.
    """

    def __init__(self, feature_dim: int = 3, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x=None, **kwargs) -> torch.Tensor:
        return x
