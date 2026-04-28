"""图像 backbone 模型。

Stageable（支持中期融合）:
    resnet18, resnet50, resnet101

非 stageable:
    efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,
    vit_base_patch16_224, vit_base_patch32_224,
    swin_tiny, swin_small, swin_base,
    convnext_tiny, convnext_small, convnext_base,
    deit_small, deit_base,
    efficientnet_v2_s, efficientnet_v2_m
"""

import torchvision.models as tv_models

from .common import TorchvisionStageable, TorchvisionWrapper
from ..registry import register_backbone


# ==============================================================================
# Stageable 模型 — 支持多 stage 中期融合
# ==============================================================================

@register_backbone('resnet18', description='ResNet18 图像特征提取器（支持 staged forward）', modality='image')
class ResNet18(TorchvisionStageable):
    def __init__(self, feature_dim=512, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.resnet18,
            default_dim=512,
            stage_dims=[64, 128, 256, 512],
            fallback_channels=[32, 64, 128, 128],
        )


@register_backbone('resnet18_stageable', description='Stageable ResNet18（别名，同 resnet18）', modality='image')
class ResNet18Stageable(ResNet18):
    """向后兼容别名，功能与 ResNet18 完全相同。"""


@register_backbone('resnet50', description='ResNet50 图像特征提取器（支持 staged forward）', modality='image')
class ResNet50(TorchvisionStageable):
    def __init__(self, feature_dim=2048, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.resnet50,
            default_dim=2048,
            stage_dims=[256, 512, 1024, 2048],
            fallback_channels=[64, 128, 256, 512],
        )


@register_backbone('resnet101', description='ResNet101 图像特征提取器（支持 staged forward）', modality='image')
class ResNet101(TorchvisionStageable):
    def __init__(self, feature_dim=2048, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.resnet101,
            default_dim=2048,
            stage_dims=[256, 512, 1024, 2048],
            fallback_channels=[64, 128, 256, 512],
        )


# ==============================================================================
# 非 stageable 模型
# ==============================================================================

@register_backbone('efficientnet_b0', description='EfficientNet-B0 图像特征提取器', modality='image')
class EfficientNetB0(TorchvisionWrapper):
    def __init__(self, feature_dim=1280, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.efficientnet_b0,
            default_dim=1280,
            fallback_channels=[32, 64],
        )


@register_backbone('mobilenet_v2', description='MobileNet-V2 轻量级图像特征提取器', modality='image')
class MobileNetV2(TorchvisionWrapper):
    def __init__(self, feature_dim=1280, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.mobilenet_v2,
            default_dim=1280,
            fallback_channels=[32, 64],
        )

    def forward(self, x=None, **inputs):
        if x is not None:
            inputs.setdefault('x', x)
        x = inputs.get('x', next(iter(inputs.values())))
        try:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x) if hasattr(self.backbone, 'avgpool') else x.mean(dim=(-2, -1))
            x = torch.flatten(x, 1)
            x = self.backbone.classifier(x)
            x = x if x.dim() == 2 else x[:, 0]
        except Exception:
            x = self.backbone(x)
            if x.dim() == 4:
                x = x.flatten(1)
        return self.proj(x)


import torch


@register_backbone('mobilenet_v3_small', description='MobileNet-V3-Small 轻量级图像特征提取器', modality='image')
class MobileNetV3Small(TorchvisionWrapper):
    def __init__(self, feature_dim=960, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.mobilenet_v3_small,
            default_dim=960,
            fallback_channels=[32, 64],
        )

    def forward(self, x=None, **inputs):
        if x is not None:
            inputs.setdefault('x', x)
        x = inputs.get('x', next(iter(inputs.values())))
        try:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        except Exception:
            x = self.backbone(x)
            if x.dim() == 4:
                x = x.flatten(1)
        return self.proj(x)


@register_backbone('mobilenet_v3_large', description='MobileNet-V3-Large 轻量级图像特征提取器', modality='image')
class MobileNetV3Large(TorchvisionWrapper):
    def __init__(self, feature_dim=1280, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.mobilenet_v3_large,
            default_dim=1280,
            fallback_channels=[32, 64],
        )

    def forward(self, x=None, **inputs):
        if x is not None:
            inputs.setdefault('x', x)
        x = inputs.get('x', next(iter(inputs.values())))
        try:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        except Exception:
            x = self.backbone(x)
            if x.dim() == 4:
                x = x.flatten(1)
        return self.proj(x)


@register_backbone('vit_base_patch16_224', description='ViT-B/16 图像特征提取器', modality='image')
class ViTBasePatch16(TorchvisionWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.vit_b_16,
            default_dim=768,
            head_attr='heads',
            fallback_channels=[64, 128],
        )


@register_backbone('vit_base_patch32_224', description='ViT-B/32 图像特征提取器', modality='image')
class ViTBasePatch32(TorchvisionWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.vit_b_32,
            default_dim=768,
            head_attr='heads',
            fallback_channels=[64, 128],
        )


@register_backbone('swin_tiny', description='Swin-T 图像特征提取器', modality='image')
class SwinTransformerTiny(TorchvisionWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.swin_t,
            default_dim=768,
            head_attr='head',
            fallback_channels=[48, 96],
        )


@register_backbone('swin_small', description='Swin-S 图像特征提取器', modality='image')
class SwinTransformerSmall(TorchvisionWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.swin_s,
            default_dim=768,
            head_attr='head',
            fallback_channels=[48, 96],
        )


@register_backbone('swin_base', description='Swin-B 图像特征提取器', modality='image')
class SwinTransformerBase(TorchvisionWrapper):
    def __init__(self, feature_dim=1024, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.swin_b,
            default_dim=1024,
            head_attr='head',
            fallback_channels=[64, 128],
        )


@register_backbone('convnext_tiny', description='ConvNeXt-T 图像特征提取器', modality='image')
class ConvNeXtTiny(TorchvisionWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.convnext_tiny,
            default_dim=768,
            head_attr='classifier',
            fallback_channels=[48, 96],
        )

    def forward(self, x=None, **inputs):
        if x is not None:
            inputs.setdefault('x', x)
        x = inputs.get('x', next(iter(inputs.values())))
        try:
            x = self.backbone.features(x)
            x = x.mean(dim=(-2, -1))
        except Exception:
            x = self.backbone(x)
            if x.dim() == 4:
                x = x.flatten(1)
        return self.proj(x)


@register_backbone('convnext_small', description='ConvNeXt-S 图像特征提取器', modality='image')
class ConvNeXtSmall(TorchvisionWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.convnext_small,
            default_dim=768,
            head_attr='classifier',
            fallback_channels=[48, 96],
        )

    def forward(self, x=None, **inputs):
        if x is not None:
            inputs.setdefault('x', x)
        x = inputs.get('x', next(iter(inputs.values())))
        try:
            x = self.backbone.features(x)
            x = x.mean(dim=(-2, -1))
        except Exception:
            x = self.backbone(x)
            if x.dim() == 4:
                x = x.flatten(1)
        return self.proj(x)


@register_backbone('convnext_base', description='ConvNeXt-B 图像特征提取器', modality='image')
class ConvNeXtBase(TorchvisionWrapper):
    def __init__(self, feature_dim=1024, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.convnext_base,
            default_dim=1024,
            head_attr='classifier',
            fallback_channels=[64, 128],
        )

    def forward(self, x=None, **inputs):
        if x is not None:
            inputs.setdefault('x', x)
        x = inputs.get('x', next(iter(inputs.values())))
        try:
            x = self.backbone.features(x)
            x = x.mean(dim=(-2, -1))
        except Exception:
            x = self.backbone(x)
            if x.dim() == 4:
                x = x.flatten(1)
        return self.proj(x)


@register_backbone('deit_small', description='DeiT-S 图像特征提取器', modality='image')
class DeiTSmall(TorchvisionWrapper):
    def __init__(self, feature_dim=384, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.vit_small_patch16_224,
            default_dim=384,
            head_attr='heads',
            fallback_channels=[48, 96],
        )


@register_backbone('deit_base', description='DeiT-B 图像特征提取器', modality='image')
class DeiTBase(TorchvisionWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.vit_base_patch16_224,
            default_dim=768,
            head_attr='heads',
            fallback_channels=[64, 128],
        )


@register_backbone('efficientnet_v2_s', description='EfficientNet-V2-S 图像特征提取器', modality='image')
class EfficientNetV2Small(TorchvisionWrapper):
    def __init__(self, feature_dim=1280, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.efficientnet_v2_s,
            default_dim=1280,
            fallback_channels=[24, 48],
        )


@register_backbone('efficientnet_v2_m', description='EfficientNet-V2-M 图像特征提取器', modality='image')
class EfficientNetV2Medium(TorchvisionWrapper):
    def __init__(self, feature_dim=1280, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_fn=tv_models.efficientnet_v2_m,
            default_dim=1280,
            fallback_channels=[48, 96],
        )
