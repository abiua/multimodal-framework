"""图像模型"""

import torch
import torch.nn as nn
import torchvision.models as models
from ..registry import register_backbone

@register_backbone('resnet18_stageable', description='Stageable ResNet18', modality='image')
class ResNet18Stageable(nn.Module):
    num_stages = 4

    def __init__(self, feature_dim=512, pretrained=True, **kwargs):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.stages = nn.ModuleList([
            backbone.layer1,  # stage 0
            backbone.layer2,  # stage 1
            backbone.layer3,  # stage 2
            backbone.layer4,  # stage 3
        ])
        self.pool = backbone.avgpool
        self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
        self.feature_dim = feature_dim

    def init_state(self, x):
        x = self.stem(x)
        return x

    def forward_stage(self, state, stage_idx: int):
        return self.stages[stage_idx](state)

    def forward_head(self, state):
        x = self.pool(state)
        x = torch.flatten(x, 1)
        return self.proj(x)

    def forward(self, x):
        state = self.init_state(x)
        for stage_idx in range(self.num_stages):
            state = self.forward_stage(state, stage_idx)
        return self.forward_head(state)

@register_backbone('resnet18', description='ResNet18图像特征提取器', modality='image')
class ResNet18(nn.Module):
    def __init__(self, feature_dim=512, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        backbone = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('resnet50', description='ResNet50图像特征提取器', modality='image')
class ResNet50(nn.Module):
    def __init__(self, feature_dim=2048, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        backbone = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(2048, feature_dim) if feature_dim != 2048 else nn.Identity()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('resnet101', description='ResNet101图像特征提取器', modality='image')
class ResNet101(nn.Module):
    def __init__(self, feature_dim=2048, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        backbone = models.resnet101(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(2048, feature_dim) if feature_dim != 2048 else nn.Identity()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('efficientnet_b0', description='EfficientNet-B0图像特征提取器', modality='image')
class EfficientNetB0(nn.Module):
    def __init__(self, feature_dim=1280, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import efficientnet_b0
            backbone = efficientnet_b0(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            self.proj = nn.Linear(64, feature_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('mobilenet_v2', description='MobileNet-V2轻量级图像特征提取器', modality='image')
class MobileNetV2(nn.Module):
    def __init__(self, feature_dim=1280, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        backbone = models.mobilenet_v2(pretrained=pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)
    
@register_backbone('mobilenet_v3_small', description='MobileNet-V3-Small轻量级图像特征提取器', modality='image')
class MobileNetV3Small(nn.Module):
    def __init__(self, feature_dim=960, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        backbone = models.mobilenet_v3_small(pretrained=pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(960, feature_dim) if feature_dim != 960 else nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)
    

@register_backbone('mobilenet_v3_large', description='MobileNet-V3-Large轻量级图像特征提取器', modality='image')

class MobileNetV3Large(nn.Module):
    def __init__(self, feature_dim=1280, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        backbone = models.mobilenet_v3_large(pretrained=pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        return self.proj(x)
    
@register_backbone('mobilevitv3_small', description='MobileViT-V3-Small轻量级图像特征提取器', modality='image')
class MobileViT_V3_Small(nn.Module):
    def __init__(self, feature_dim=512, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        backbone = models.mobilenet_v3_small(pretrained=pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)
    
@register_backbone('mobilevitv3_large', description='MobileViT-V3-Large轻量级图像特征提取器', modality='image')
class MobileViT_V3_Large(nn.Module):
    def __init__(self, feature_dim=512, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        backbone = models.mobilenet_v3_large(pretrained=pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('vit_base_patch16_224', description='Vision Transformer Base (ViT-B/16)图像特征提取器', modality='image')
class ViTBasePatch16(nn.Module):
    """Vision Transformer Base模型，使用16x16的patch大小"""
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import vit_b_16
            backbone = vit_b_16(pretrained=pretrained)
            self.features = backbone
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except:
            # 回退到简单实现
            self.features = self._build_simple_vit()
            self.proj = nn.Linear(768, feature_dim)
    
    def _build_simple_vit(self):
        """简化的ViT实现"""
        import torch.nn.functional as F
        from functools import partial
        
        class SimpleViT(nn.Module):
            def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
                super().__init__()
                self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2 + 1, embed_dim))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                             dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
                    for _ in range(depth)
                ])
                self.norm = nn.LayerNorm(embed_dim)
            
            def forward(self, x):
                x = self.patch_embed(x)  # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
                x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
                cls_token = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = x + self.pos_embed
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return x[:, 0]  # 返回CLS token
        
        return SimpleViT()
    
    def forward(self, x):
        x = self.features(x)
        return self.proj(x)


@register_backbone('vit_base_patch32_224', description='Vision Transformer Base (ViT-B/32)图像特征提取器', modality='image')
class ViTBasePatch32(nn.Module):
    """Vision Transformer Base模型，使用32x32的patch大小"""
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import vit_b_32
            backbone = vit_b_32(pretrained=pretrained)
            self.features = backbone
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except:
            # 回退到简单实现
            self.features = self._build_simple_vit()
            self.proj = nn.Linear(768, feature_dim)
    
    def _build_simple_vit(self):
        """简化的ViT实现"""
        class SimpleViT(nn.Module):
            def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
                super().__init__()
                self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2 + 1, embed_dim))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                             dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
                    for _ in range(depth)
                ])
                self.norm = nn.LayerNorm(embed_dim)
            
            def forward(self, x):
                x = self.patch_embed(x)  # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
                x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
                cls_token = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = x + self.pos_embed
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return x[:, 0]  # 返回CLS token
        
        return SimpleViT()
    
    def forward(self, x):
        x = self.features(x)
        return self.proj(x)


@register_backbone('swin_tiny', description='Swin Transformer Tiny图像特征提取器', modality='image')
class SwinTransformerTiny(nn.Module):
    """Swin Transformer Tiny模型"""
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import swin_t
            backbone = swin_t(pretrained=pretrained)
            backbone.head = nn.Identity()
            self.features = backbone
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            self.proj = nn.Linear(192, feature_dim)
    
    def forward(self, x):
        x = self.features(x)
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('swin_small', description='Swin Transformer Small图像特征提取器', modality='image')
class SwinTransformerSmall(nn.Module):
    """Swin Transformer Small模型"""
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import swin_s
            backbone = swin_s(pretrained=pretrained)
            backbone.head = nn.Identity()
            self.features = backbone
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            self.proj = nn.Linear(192, feature_dim)
    
    def forward(self, x):
        x = self.features(x)
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('swin_base', description='Swin Transformer Base图像特征提取器', modality='image')
class SwinTransformerBase(nn.Module):
    """Swin Transformer Base模型"""
    def __init__(self, feature_dim=1024, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import swin_b
            backbone = swin_b(pretrained=pretrained)
            # 移除分类头，只保留特征提取部分
            backbone.head = nn.Identity()
            self.features = backbone
            self.proj = nn.Linear(1024, feature_dim) if feature_dim != 1024 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            self.proj = nn.Linear(256, feature_dim)
    
    def forward(self, x):
        x = self.features(x)
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('convnext_tiny', description='ConvNeXt Tiny图像特征提取器', modality='image')
class ConvNeXtTiny(nn.Module):
    """ConvNeXt Tiny模型"""
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import convnext_tiny
            backbone = convnext_tiny(pretrained=pretrained)
            backbone.classifier = nn.Identity()
            self.features = backbone
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            self.pool = nn.Identity()
            self.proj = nn.Linear(192, feature_dim)
    
    def forward(self, x):
        x = self.features.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('convnext_small', description='ConvNeXt Small图像特征提取器', modality='image')
class ConvNeXtSmall(nn.Module):
    """ConvNeXt Small模型"""
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import convnext_small
            backbone = convnext_small(pretrained=pretrained)
            backbone.classifier = nn.Identity()
            self.features = backbone
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            self.pool = nn.Identity()
            self.proj = nn.Linear(192, feature_dim)
    
    def forward(self, x):
        x = self.features.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('convnext_base', description='ConvNeXt Base图像特征提取器', modality='image')
class ConvNeXtBase(nn.Module):
    """ConvNeXt Base模型"""
    def __init__(self, feature_dim=1024, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import convnext_base
            backbone = convnext_base(pretrained=pretrained)
            backbone.classifier = nn.Identity()
            self.features = backbone
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            self.pool = nn.Identity()
            self.proj = nn.Linear(256, feature_dim)
    
    def forward(self, x):
        x = self.features.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('deit_small', description='DeiT Small图像特征提取器', modality='image')
class DeiTSmall(nn.Module):
    """DeiT (Data-efficient image Transformer) Small模型"""
    def __init__(self, feature_dim=384, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import vit_small_patch16_224
            backbone = vit_small_patch16_224(pretrained=pretrained)
            self.features = backbone
            self.proj = nn.Linear(384, feature_dim) if feature_dim != 384 else nn.Identity()
        except:
            # 回退到简单实现
            self.features = self._build_simple_vit()
            self.proj = nn.Linear(384, feature_dim)
    
    def _build_simple_vit(self):
        """简化的ViT实现"""
        class SimpleViT(nn.Module):
            def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.):
                super().__init__()
                self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2 + 1, embed_dim))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                             dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
                    for _ in range(depth)
                ])
                self.norm = nn.LayerNorm(embed_dim)
            
            def forward(self, x):
                x = self.patch_embed(x)  # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
                x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
                cls_token = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = x + self.pos_embed
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return x[:, 0]  # 返回CLS token
        
        return SimpleViT()
    
    def forward(self, x):
        x = self.features(x)
        return self.proj(x)


@register_backbone('deit_base', description='DeiT Base图像特征提取器', modality='image')
class DeiTBase(nn.Module):
    """DeiT (Data-efficient image Transformer) Base模型"""
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import vit_base_patch16_224
            backbone = vit_base_patch16_224(pretrained=pretrained)
            self.features = backbone
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except:
            # 回退到简单实现
            self.features = self._build_simple_vit()
            self.proj = nn.Linear(768, feature_dim)
    
    def _build_simple_vit(self):
        """简化的ViT实现"""
        class SimpleViT(nn.Module):
            def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
                super().__init__()
                self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2 + 1, embed_dim))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                             dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
                    for _ in range(depth)
                ])
                self.norm = nn.LayerNorm(embed_dim)
            
            def forward(self, x):
                x = self.patch_embed(x)  # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
                x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
                cls_token = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = x + self.pos_embed
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return x[:, 0]  # 返回CLS token
        
        return SimpleViT()
    
    def forward(self, x):
        x = self.features(x)
        return self.proj(x)


@register_backbone('efficientnet_v2_s', description='EfficientNet-V2 Small图像特征提取器', modality='image')
class EfficientNetV2Small(nn.Module):
    """EfficientNet-V2 Small模型"""
    def __init__(self, feature_dim=1280, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import efficientnet_v2_s
            backbone = efficientnet_v2_s(pretrained=pretrained)
            self.features = backbone
            self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            self.proj = nn.Linear(48, feature_dim)
    
    def forward(self, x):
        x = self.features(x)
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('efficientnet_v2_m', description='EfficientNet-V2 Medium图像特征提取器', modality='image')
class EfficientNetV2Medium(nn.Module):
    """EfficientNet-V2 Medium模型"""
    def __init__(self, feature_dim=1280, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            from torchvision.models import efficientnet_v2_m
            backbone = efficientnet_v2_m(pretrained=pretrained)
            self.features = backbone
            self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            self.proj = nn.Linear(96, feature_dim)
    
    def forward(self, x):
        x = self.features(x)
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.proj(x)
    
