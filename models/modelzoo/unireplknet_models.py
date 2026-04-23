"""UniRepLKNet backbones for image / audio / wave modalities.

放置路径:
    models/modelzoo/unireplknet_models.py

前置要求:
    1) 将官方 unireplknet.py 放到 models/modelzoo/vendor/unireplknet.py
    2) 确保 models/modelzoo/vendor/__init__.py 存在
    3) 安装 timm；若需要自动下载预训练权重，还需 huggingface_hub
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_backbone
from .UniRepLKNet.unireplknet import (
    unireplknet_a,
    unireplknet_f,
    unireplknet_p,
    unireplknet_n,
    unireplknet_t,
    unireplknet_s,
    unireplknet_b,
    unireplknet_l,
    unireplknet_xl,
)


_VARIANT_FACTORIES = {
    "a": unireplknet_a,
    "f": unireplknet_f,
    "p": unireplknet_p,
    "n": unireplknet_n,
    "t": unireplknet_t,
    "s": unireplknet_s,
    "b": unireplknet_b,
    "l": unireplknet_l,
    "xl": unireplknet_xl,
}

_VARIANT_STAGE_DIMS = {
    "a": [40, 80, 160, 320],
    "f": [48, 96, 192, 384],
    "p": [64, 128, 256, 512],
    "n": [80, 160, 320, 640],
    "t": [80, 160, 320, 640],
    "s": [96, 192, 384, 768],
    "b": [128, 256, 512, 1024],
    "l": [192, 384, 768, 1536],
    "xl": [256, 512, 1024, 2048],
}


def _adapt_input_conv(in_chans: int, conv_weight: torch.Tensor) -> torch.Tensor:
    """将 RGB 输入卷积权重适配到任意输入通道数。

    参考 timm 的做法：
    - 1 通道时，对 RGB 权重求和；
    - >3 通道时，复制并按通道数缩放；
    - 2 通道时，同样走复制 + 缩放路径。
    """
    if conv_weight.ndim != 4:
        raise ValueError(f"conv_weight 应为 4D 张量，实际为 {tuple(conv_weight.shape)}")

    out_channels, old_in_chans, kh, kw = conv_weight.shape
    if in_chans == old_in_chans:
        return conv_weight

    conv_weight = conv_weight.float()

    if in_chans == 1:
        if old_in_chans > 3:
            if old_in_chans % 3 != 0:
                raise ValueError(
                    f"无法将输入通道 {old_in_chans} 适配到 1 通道，请手动处理第一层卷积"
                )
            conv_weight = conv_weight.reshape(out_channels, old_in_chans // 3, 3, kh, kw)
            conv_weight = conv_weight.sum(dim=2)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
        return conv_weight

    if old_in_chans != 3:
        raise ValueError(
            f"当前仅支持从 3 通道权重扩展到 {in_chans} 通道，实际旧通道数为 {old_in_chans}"
        )

    repeat = int(math.ceil(in_chans / 3.0))
    conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
    conv_weight *= 3.0 / float(in_chans)
    return conv_weight


def _load_checkpoint_file(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"无法解析 checkpoint: {checkpoint_path}")

    # 兼容 DDP / Lightning 等常见保存格式
    if checkpoint and all(isinstance(k, str) and k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}
    return checkpoint


def _build_variant_model(variant: str, factory, pretrained: bool, common_kwargs: Dict) -> nn.Module:
    """根据 variant 选择官方构造参数。

    说明：
    - a/f/p/n/t/s 直接走 in_1k_pretrained；
    - b/l/xl 默认走 in_22k_pretrained。
    若你之后想严格区分 22k / 22k_to1k，可再扩展一个配置字段。
    """
    if variant in {"b", "l", "xl"}:
        return factory(in_22k_pretrained=pretrained, **common_kwargs)
    return factory(in_1k_pretrained=pretrained, **common_kwargs)


def _create_unireplknet(
    variant: str,
    in_chans: int,
    pretrained: bool,
    deploy: bool,
    with_cp: bool,
    attempt_use_lk_impl: bool,
    drop_path_rate: float,
    pretrained_checkpoint: Optional[str] = None,
) -> nn.Module:
    variant = variant.lower()
    if variant not in _VARIANT_FACTORIES:
        raise ValueError(
            f"不支持的 UniRepLKNet variant: {variant}，可选值: {list(_VARIANT_FACTORIES.keys())}"
        )

    factory = _VARIANT_FACTORIES[variant]
    common_kwargs = dict(
        in_chans=in_chans,
        deploy=deploy,
        with_cp=with_cp,
        attempt_use_lk_impl=attempt_use_lk_impl,
        drop_path_rate=drop_path_rate,
    )

    # 3 通道时可直接走官方预训练加载
    if in_chans == 3 and pretrained_checkpoint is None:
        return _build_variant_model(variant, factory, pretrained, common_kwargs)

    # 先创建目标模型（不直接走官方预训练）
    model = _build_variant_model(variant, factory, False, common_kwargs)

    if not pretrained and pretrained_checkpoint is None:
        return model

    if pretrained_checkpoint is not None:
        state_dict = _load_checkpoint_file(pretrained_checkpoint)
    else:
        # 先构建一个 RGB 预训练模型，再把第一层卷积权重适配到目标通道数
        source_common_kwargs = dict(
            in_chans=3,
            deploy=deploy,
            with_cp=with_cp,
            attempt_use_lk_impl=attempt_use_lk_impl,
            drop_path_rate=drop_path_rate,
        )
        source_model = _build_variant_model(variant, factory, True, source_common_kwargs)
        state_dict = source_model.state_dict()
        del source_model

    first_conv_key = "downsample_layers.0.0.weight"
    if first_conv_key in state_dict and state_dict[first_conv_key].shape[1] != in_chans:
        state_dict = dict(state_dict)
        state_dict[first_conv_key] = _adapt_input_conv(in_chans, state_dict[first_conv_key])

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[UniRepLKNet] missing keys when loading weights: {missing}")
    if unexpected:
        print(f"[UniRepLKNet] unexpected keys when loading weights: {unexpected}")
    return model


class _UniRepLKNetBase(nn.Module):
    num_stages = 4

    def __init__(
        self,
        feature_dim: int = 512,
        pretrained: bool = False,
        variant: str = "p",
        in_chans: int = 3,
        dropout: float = 0.1,
        deploy: bool = False,
        with_cp: bool = False,
        attempt_use_lk_impl: bool = False,
        drop_path_rate: float = 0.0,
        pretrained_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in _VARIANT_STAGE_DIMS:
            raise ValueError(
                f"不支持的 UniRepLKNet variant: {variant}，可选值: {list(_VARIANT_STAGE_DIMS.keys())}"
            )

        self.feature_dim = feature_dim
        self.variant = variant
        self.in_chans = in_chans
        self.stage_dims = _VARIANT_STAGE_DIMS[variant]
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.backbone = _create_unireplknet(
            variant=variant,
            in_chans=in_chans,
            pretrained=pretrained,
            deploy=deploy,
            with_cp=with_cp,
            attempt_use_lk_impl=attempt_use_lk_impl,
            drop_path_rate=drop_path_rate,
            pretrained_checkpoint=pretrained_checkpoint,
        )

        last_dim = self.stage_dims[-1]
        self.proj = nn.Identity() if feature_dim == last_dim else nn.Linear(last_dim, feature_dim)

    def _prepare_input(self, x: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def init_state(self, x: Optional[torch.Tensor] = None, **kwargs):
        x = self._prepare_input(x=x, **kwargs)
        return x

    def forward_stage(self, state, stage_idx: int):
        x = self.backbone.downsample_layers[stage_idx](state)
        x = self.backbone.stages[stage_idx](x)
        return x

    def forward_head(self, state):
        x = state.mean(dim=(-2, -1))
        x = self.backbone.norm(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs):
        state = self.init_state(x=x, **kwargs)
        for stage_idx in range(self.num_stages):
            state = self.forward_stage(state, stage_idx)
        return self.forward_head(state)

    def reparameterize_unireplknet(self):
        self.backbone.reparameterize_unireplknet()


@register_backbone(
    "unireplknet_image",
    description="UniRepLKNet image backbone for multimodal fish feeding",
    modality="image",
)
class UniRepLKNetImage(_UniRepLKNetBase):
    def __init__(self, feature_dim=512, pretrained=True, variant="p", **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            variant=variant,
            in_chans=3,
            **kwargs,
        )

    def _prepare_input(self, x: Optional[torch.Tensor] = None, image: Optional[torch.Tensor] = None, **kwargs):
        x = x if x is not None else image
        if x is None:
            raise ValueError("UniRepLKNetImage 未接收到输入，请检查 image loader 输出")
        if x.dim() != 4:
            raise ValueError(f"图像输入应为 [B, 3, H, W]，实际为 {tuple(x.shape)}")
        if x.size(1) != 3:
            raise ValueError(f"图像输入通道应为 3，实际为 {x.size(1)}")
        return x


@register_backbone(
    "unireplknet_audio",
    description="UniRepLKNet audio backbone for stereo mel-spectrogram",
    modality="audio",
)
class UniRepLKNetAudio(_UniRepLKNetBase):
    def __init__(self, feature_dim=512, pretrained=False, variant="p", audio_channels=2, **kwargs):
        self.audio_channels = audio_channels
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            variant=variant,
            in_chans=audio_channels,
            **kwargs,
        )

    def _prepare_input(
        self,
        x: Optional[torch.Tensor] = None,
        mel_spectrogram: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        x = x if x is not None else mel_spectrogram
        x = x if x is not None else audio
        if x is None:
            raise ValueError("UniRepLKNetAudio 未接收到输入，请检查 audio loader 输出")

        if x.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"音频输入应为 [B, C, H, W] 或 [B, H, W]，实际为 {tuple(x.shape)}")

        if x.size(1) == 1 and self.audio_channels == 2:
            x = x.repeat(1, 2, 1, 1)
        elif x.size(1) != self.audio_channels:
            raise ValueError(
                f"音频输入通道数应为 {self.audio_channels}，实际为 {x.size(1)}；"
                f"请检查 stereo loader 输出"
            )

        if x.size(-1) != 224 or x.size(-2) != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x


@register_backbone(
    "unireplknet_wave",
    description="UniRepLKNet wave backbone with learnable 1D-to-2D projector",
    modality="wave",
)
class UniRepLKNetWave(_UniRepLKNetBase):
    def __init__(
        self,
        feature_dim=512,
        pretrained=False,
        variant="p",
        seq_len=512,
        in_channels=6,
        image_size=224,
        adapter_channels=64,
        **kwargs,
    ):
        self.seq_len = seq_len
        self.wave_in_channels = in_channels
        self.image_size = image_size
        self.adapter_channels = adapter_channels

        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            variant=variant,
            in_chans=1,
            **kwargs,
        )

        self.wave_adapter = nn.Sequential(
            nn.Conv1d(in_channels, adapter_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(adapter_channels),
            nn.GELU(),
            nn.Conv1d(adapter_channels, adapter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(adapter_channels),
            nn.GELU(),
        )
        self.wave_to_map = nn.Conv1d(adapter_channels, image_size, kernel_size=1)

    def _prepare_input(
        self,
        x: Optional[torch.Tensor] = None,
        wave: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        x = x if x is not None else wave
        if x is None:
            raise ValueError("UniRepLKNetWave 未接收到输入，请检查 wave loader 输出")
        if x.dim() != 3:
            raise ValueError(f"Wave 输入应为 [B, T, C] 或 [B, C, T]，实际为 {tuple(x.shape)}")

        # 与项目内 TCN 保持一致：默认 loader 输出 [B, T, C]
        if x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)  # [B, C, T]

        if x.size(1) != self.wave_in_channels:
            raise ValueError(
                f"Wave 输入通道数应为 {self.wave_in_channels}，实际为 {x.size(1)}"
            )

        x = self.wave_adapter(x)
        if x.size(-1) != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode="linear", align_corners=False)

        x = self.wave_to_map(x)      # [B, 224, 224]
        x = x.unsqueeze(1)           # [B, 1, 224, 224]
        x = (x - x.mean(dim=(-2, -1), keepdim=True)) / (x.std(dim=(-2, -1), keepdim=True) + 1e-6)
        return x