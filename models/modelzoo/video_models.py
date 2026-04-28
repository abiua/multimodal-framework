"""视频 backbone 模型。

全部为非 stageable（视频模型 stage 分解复杂，暂不支持中期融合）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..registry import register_backbone
from ..backbone_base import BaseBackbone


# ==============================================================================
# 视频 Transformer 组件
# ==============================================================================

class PatchEmbed3D(nn.Module):
    """3D Patch Embedding。"""

    def __init__(self, img_size=224, patch_size=16, tube_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tube_size = tube_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tube_size, patch_size, patch_size),
            stride=(tube_size, patch_size, patch_size),
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-head attention。"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP 模块。"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm。"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# 视频模型
# ==============================================================================

def _normalize_video_input(x: torch.Tensor) -> torch.Tensor:
    """统一视频输入为 (B, C, T, H, W) 格式。"""
    if x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
        x = x.permute(0, 2, 1, 3, 4)
    return x


@register_backbone('timesformer', description='TimeSformer 视频特征提取器', modality='video')
class TimeSformer(BaseBackbone):
    def __init__(self, img_size=224, patch_size=16, num_frames=8,
                 in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 feature_dim=768, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.num_tokens = self.num_patches * num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, feature_dim) if feature_dim != embed_dim else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.time_embed, std=.02)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = _normalize_video_input(x)

        B, T = x.shape[0], x.shape[2]
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) → (B, C, T, H, W)

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_no_cls = x[:, 1:, :]
        x_no_cls = rearrange(x_no_cls, 'b (t n) d -> b t n d', t=T)
        x_no_cls = x_no_cls + self.time_embed[:, :T, :].unsqueeze(2)
        x_no_cls = rearrange(x_no_cls, 'b t n d -> b (t n) d')
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.proj(x[:, 0])


@register_backbone('vivit', description='ViViT 视频特征提取器', modality='video')
class ViViT(BaseBackbone):
    def __init__(self, img_size=224, patch_size=16, num_frames=16,
                 in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 feature_dim=768, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        self.spatial_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth // 2)
        ])
        self.temporal_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth // 2)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, feature_dim) if feature_dim != embed_dim else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.time_embed, std=.02)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = _normalize_video_input(x)

        B, T = x.shape[0], x.shape[2]

        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)

        cls_token = self.cls_token.expand(B, 1, -1).unsqueeze(1).expand(-1, T, -1, -1)
        x = torch.cat([cls_token, x], dim=2)
        x = x + self.pos_embed

        x = rearrange(x, 'b t n d -> (b t) n d')
        for blk in self.spatial_blocks:
            x = blk(x)
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)

        cls_tokens = x[:, :, 0, :]
        cls_tokens = cls_tokens + self.time_embed

        for blk in self.temporal_blocks:
            cls_tokens = blk(cls_tokens)

        x = cls_tokens.mean(dim=1)
        x = self.norm(x)
        return self.proj(x)


@register_backbone('video_swin_tiny', description='Video Swin Transformer Tiny', modality='video')
class VideoSwinTransformerTiny(BaseBackbone):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 feature_dim=768, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(2, patch_size, patch_size),
            stride=(2, patch_size, patch_size),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 分离 transformer stages 和 patch merging
        self.transformer_stages = nn.ModuleList()
        self.patch_merging = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i_stage in range(len(depths)):
            stage_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim * (2 ** i_stage),
                    num_heads=num_heads[i_stage],
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                )
                for _ in range(depths[i_stage])
            ])
            self.transformer_stages.append(stage_blocks)
            self.norms.append(nn.LayerNorm(embed_dim * (2 ** i_stage)))

            if i_stage < len(depths) - 1:
                self.patch_merging.append(nn.Conv3d(
                    embed_dim * (2 ** i_stage),
                    embed_dim * (2 ** (i_stage + 1)),
                    kernel_size=2, stride=2,
                ))
            else:
                self.patch_merging.append(nn.Identity())

        self.final_dim = embed_dim * (2 ** (len(depths) - 1))
        self.proj = nn.Linear(self.final_dim, feature_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = _normalize_video_input(x)

        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        for i, (stage_blocks, merge) in enumerate(zip(self.transformer_stages, self.patch_merging)):
            for blk in stage_blocks:
                x = blk(x)
            x = self.norms[i](x)

            if not isinstance(merge, nn.Identity):
                h = w = int(x.shape[1] ** 0.5)
                x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
                x = merge(x.unsqueeze(2)).squeeze(2)
                x = rearrange(x, 'b c h w -> b (h w) c')

        x = x.mean(dim=1)
        return self.proj(x)


@register_backbone('slowfast_r50', description='SlowFast R50 视频特征提取器', modality='video')
class SlowFastR50(BaseBackbone):
    def __init__(self, img_size=224, in_chans=3,
                 slow_feature_dim=2048, fast_feature_dim=256,
                 alpha=4, tau=16, feature_dim=2304, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.alpha = alpha

        # Slow pathway
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.slow_res2 = self._make_layer(64, 64, 3)
        self.slow_res3 = self._make_layer(256, 128, 4, stride=(1, 2, 2))
        self.slow_res4 = self._make_layer(512, 256, 6, stride=(1, 2, 2))
        self.slow_res5 = self._make_layer(1024, 512, 3, stride=(1, 2, 2))

        # Fast pathway
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fast_res2 = self._make_layer(8, 8, 3, fast=True)
        self.fast_res3 = self._make_layer(32, 16, 4, stride=(1, 2, 2), fast=True)
        self.fast_res4 = self._make_layer(64, 32, 6, stride=(1, 2, 2), fast=True)
        self.fast_res5 = self._make_layer(128, 64, 3, stride=(1, 2, 2), fast=True)

        # Lateral connections
        self.lateral_p1 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32, 32 * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64, 64 * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128, 128 * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(slow_feature_dim + fast_feature_dim, feature_dim)

    @staticmethod
    def _make_layer(inplanes, planes, blocks, stride=(1, 1, 1), fast=False):
        layers = []
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        mult = 8 if fast else 4
        layers.append(nn.Conv3d(inplanes, planes * mult, kernel_size=1, stride=stride))
        layers.append(nn.BatchNorm3d(planes * mult))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv3d(planes * mult, planes * mult, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(planes * mult))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = _normalize_video_input(x)
        B = x.shape[0]

        # Slow pathway
        slow_x = x[:, :, ::self.alpha, :, :]
        slow_x = self.slow_maxpool(self.slow_relu(self.slow_bn1(self.slow_conv1(slow_x))))

        # Fast pathway
        fast_x = self.fast_maxpool(self.fast_relu(self.fast_bn1(self.fast_conv1(x))))

        # Lateral + stage 2
        slow_x = slow_x + self.lateral_p1(fast_x)
        slow_x = self.slow_res2(slow_x)
        fast_x = self.fast_res2(fast_x)
        slow_x = slow_x + self.lateral_res2(fast_x)

        # Stage 3
        slow_x = self.slow_res3(slow_x)
        fast_x = self.fast_res3(fast_x)
        slow_x = slow_x + self.lateral_res3(fast_x)

        # Stage 4
        slow_x = self.slow_res4(slow_x)
        fast_x = self.fast_res4(fast_x)
        slow_x = slow_x + self.lateral_res4(fast_x)

        # Stage 5
        slow_x = self.slow_res5(slow_x)
        fast_x = self.fast_res5(fast_x)

        slow_x = self.avgpool(slow_x).flatten(1)
        fast_x = self.avgpool(fast_x).flatten(1)
        return self.proj(torch.cat([slow_x, fast_x], dim=1))


# ==============================================================================
# Torchvision 视频模型包装
# ==============================================================================

def _build_video_fallback(in_chans, embed_dim=64):
    return nn.Sequential(
        nn.Conv3d(in_chans, embed_dim, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
        nn.BatchNorm3d(embed_dim),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        nn.AdaptiveAvgPool3d(1),
    )


@register_backbone('r3d_18', description='R3D-18 视频特征提取器', modality='video')
class R3D18(BaseBackbone):
    def __init__(self, img_size=224, in_chans=3, feature_dim=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        try:
            from torchvision.models.video import r3d_18
            backbone = r3d_18(pretrained=True)
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
        except Exception:
            self.backbone = _build_video_fallback(in_chans)
            self.proj = nn.Linear(64, feature_dim)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = _normalize_video_input(x)
        x = self.backbone(x)
        return self.proj(x.flatten(1))


@register_backbone('mc3_18', description='MC3-18 视频特征提取器', modality='video')
class MC318(BaseBackbone):
    def __init__(self, img_size=224, in_chans=3, feature_dim=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        try:
            from torchvision.models.video import mc3_18
            backbone = mc3_18(pretrained=True)
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
        except Exception:
            self.backbone = _build_video_fallback(in_chans)
            self.proj = nn.Linear(64, feature_dim)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = _normalize_video_input(x)
        x = self.backbone(x)
        return self.proj(x.flatten(1))


@register_backbone('r2plus1d_18', description='R(2+1)D-18 视频特征提取器', modality='video')
class R2Plus1D18(BaseBackbone):
    def __init__(self, img_size=224, in_chans=3, feature_dim=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        try:
            from torchvision.models.video import r2plus1d_18
            backbone = r2plus1d_18(pretrained=True)
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
        except Exception:
            self.backbone = nn.Sequential(
                nn.Conv3d(3, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
                nn.BatchNorm3d(45), nn.ReLU(inplace=True),
                nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.AdaptiveAvgPool3d(1),
            )
            self.proj = nn.Linear(64, feature_dim)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = _normalize_video_input(x)
        x = self.backbone(x)
        return self.proj(x.flatten(1))


@register_backbone('videomae', description='VideoMAE 视频特征提取器', modality='video')
class VideoMAE(BaseBackbone):
    """VideoMAE-style 视频特征提取器。

    简化实现：3D Patch Embedding + Transformer Encoder（无 decoder）。
    预训练 checkpoint 可通过 timm 加载。
    """

    def __init__(self, img_size=224, patch_size=16, num_frames=16,
                 tubelet_size=2, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 feature_dim=768, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

        num_patches = (img_size // patch_size) ** 2 * (num_frames // tubelet_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, feature_dim) if feature_dim != embed_dim else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = _normalize_video_input(x)
        x = x.permute(0, 2, 1, 3, 4)     # (B, T, C, H, W) -> (B, C, T, H, W)

        B = x.shape[0]
        x = self.patch_embed(x)          # [B, embed_dim, T', H', W']
        x = x.flatten(2).transpose(1, 2) # [B, N, embed_dim]

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.proj(x[:, 0])
