"""视频模型"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from ..registry import register_backbone


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for video"""
    def __init__(self, img_size=224, patch_size=16, tube_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tube_size = tube_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(tube_size, patch_size, patch_size),
            stride=(tube_size, patch_size, patch_size)
        )
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head attention"""
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
    """MLP module"""
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
    """Transformer block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@register_backbone('timesformer', description='TimeSformer视频特征提取器', modality='video')
class TimeSformer(nn.Module):
    """TimeSformer: Is Space-Time Attention All You Need for Video Understanding?
    
    使用Divided Space-Time Attention机制
    """
    def __init__(self, img_size=224, patch_size=16, num_frames=8, 
                 in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 feature_dim=768, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.num_tokens = self.num_patches * num_frames
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Time embeddings
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, feature_dim) if feature_dim != embed_dim else nn.Identity()
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.time_embed, std=.02)
    
    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            # 如果输入是 (B, C, T, H, W)，转换为 (B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, T*H/P*W/P, embed_dim)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position and time embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Add time embedding to temporal positions
        # Reshape to add time embedding
        x_no_cls = x[:, 1:, :]
        x_no_cls = rearrange(x_no_cls, 'b (t n) d -> b t n d', t=T)
        time_embed = self.time_embed[:, :T, :].unsqueeze(2)
        x_no_cls = x_no_cls + time_embed
        x_no_cls = rearrange(x_no_cls, 'b t n d -> b (t n) d')
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        return self.proj(x)


@register_backbone('vivit', description='Video Vision Transformer (ViViT)视频特征提取器', modality='video')
class ViViT(nn.Module):
    """ViViT: A Video Vision Transformer
    
    使用Factorized Self-Attention机制
    """
    def __init__(self, img_size=224, patch_size=16, num_frames=16,
                 in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 feature_dim=768, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # Spatial patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.num_patches = (img_size // patch_size) ** 2
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Spatial position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Temporal position embeddings
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # Spatial transformer blocks
        self.spatial_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth // 2)
        ])
        
        # Temporal transformer blocks
        self.temporal_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth // 2)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, feature_dim) if feature_dim != embed_dim else nn.Identity()
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.time_embed, std=.02)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        if x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        
        B, T, C, H, W = x.shape
        
        # Spatial patch embedding for each frame
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)  # (B*T, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, embed_dim)
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)
        
        # Add CLS token to each frame
        cls_token = self.cls_token.expand(B, 1, -1).unsqueeze(1).expand(-1, T, -1, -1)
        x = torch.cat([cls_token, x], dim=2)  # (B, T, N+1, embed_dim)
        
        # Add spatial position embeddings
        x = x + self.pos_embed
        
        # Spatial attention
        x = rearrange(x, 'b t n d -> (b t) n d')
        for blk in self.spatial_blocks:
            x = blk(x)
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)
        
        # Extract CLS tokens from each frame
        cls_tokens = x[:, :, 0, :]  # (B, T, embed_dim)
        
        # Add temporal position embeddings
        cls_tokens = cls_tokens + self.time_embed
        
        # Temporal attention
        for blk in self.temporal_blocks:
            cls_tokens = blk(cls_tokens)
        
        # Average pooling over time
        x = cls_tokens.mean(dim=1)  # (B, embed_dim)
        
        x = self.norm(x)
        return self.proj(x)


@register_backbone('video_swin_tiny', description='Video Swin Transformer Tiny视频特征提取器', modality='video')
class VideoSwinTransformerTiny(nn.Module):
    """Video Swin Transformer Tiny模型
    
    将2D Swin Transformer扩展到3D用于视频理解
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 feature_dim=768, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(2, patch_size, patch_size),
            stride=(2, patch_size, patch_size)
        )
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim))
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i_stage in range(len(depths)):
            stage = nn.ModuleList([
                Block(dim=embed_dim * (2 ** i_stage), num_heads=num_heads[i_stage],
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                      drop=drop_rate, attn_drop=attn_drop_rate,
                      drop_path=dpr[sum(depths[:i_stage]) + i])
                for i in range(depths[i_stage])
            ])
            self.stages.append(stage)
            self.norms.append(nn.LayerNorm(embed_dim * (2 ** i_stage)))
            
            # Add patch merging between stages (except last)
            if i_stage < len(depths) - 1:
                self.stages.append(nn.Conv3d(
                    embed_dim * (2 ** i_stage),
                    embed_dim * (2 ** (i_stage + 1)),
                    kernel_size=2, stride=2
                ))
        
        # Final projection
        self.proj = nn.Linear(embed_dim * (2 ** (len(depths) - 1)), feature_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self, x):
        # x: (B, T, C, H, W) or (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Process through stages
        for i, stage in enumerate(self.stages):
            if isinstance(stage, nn.ModuleList):
                # Transformer blocks
                for blk in stage:
                    x = blk(x)
                x = self.norms[i](x)
            else:
                # Patch merging
                x = rearrange(x, 'b (t h w) c -> b c t h w', 
                             h=int((x.shape[1] ** 0.5)), w=int((x.shape[1] ** 0.5)))
                x = stage(x)
                x = x.flatten(2).transpose(1, 2)
        
        # Global average pooling
        x = x.mean(dim=1)
        return self.proj(x)


@register_backbone('slowfast_r50', description='SlowFast R50视频特征提取器', modality='video')
class SlowFastR50(nn.Module):
    """SlowFast Network R50
    
    使用双路径网络处理视频：Slow路径处理低帧率，Fast路径处理高帧率
    """
    def __init__(self, img_size=224, in_chans=3,
                 slow_feature_dim=2048, fast_feature_dim=256,
                 alpha=4, beta=8, tau=16,
                 feature_dim=2304, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.alpha = alpha  # 帧率比率
        self.beta = 1 // beta  # 通道比率
        self.tau = temporal_stride = tau
        
        # Slow pathway
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Slow pathway blocks
        self.slow_res2 = self._make_layer(64, 64, 3)
        self.slow_res3 = self._make_layer(256, 128, 4, stride=(1, 2, 2))
        self.slow_res4 = self._make_layer(512, 256, 6, stride=(1, 2, 2))
        self.slow_res5 = self._make_layer(1024, 512, 3, stride=(1, 2, 2))
        
        # Fast pathway
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Fast pathway blocks
        self.fast_res2 = self._make_layer(8, 8, 3, fast=True)
        self.fast_res3 = self._make_layer(32, 16, 4, stride=(1, 2, 2), fast=True)
        self.fast_res4 = self._make_layer(64, 32, 6, stride=(1, 2, 2), fast=True)
        self.fast_res5 = self._make_layer(128, 64, 3, stride=(1, 2, 2), fast=True)
        
        # Lateral connections
        self.lateral_p1 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32, 32 * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64, 64 * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128, 128 * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        # Final projection
        self.proj = nn.Linear(slow_feature_dim + fast_feature_dim, feature_dim)
    
    def _make_layer(self, inplanes, planes, blocks, stride=(1, 1, 1), fast=False):
        """Create a residual layer"""
        layers = []
        # 处理stride参数，可以是int或tuple
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        
        layers.append(nn.Conv3d(inplanes, planes * 4 if not fast else planes * 8, 
                               kernel_size=1, stride=stride))
        layers.append(nn.BatchNorm3d(planes * 4 if not fast else planes * 8))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv3d(planes * 4 if not fast else planes * 8,
                                   planes * 4 if not fast else planes * 8,
                                   kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(planes * 4 if not fast else planes * 8))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, T, C, H, W) or (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        
        B, C, T, H, W = x.shape
        
        # Slow pathway
        slow_x = x[:, :, ::self.alpha, :, :]  # Subsample frames
        slow_x = self.slow_conv1(slow_x)
        slow_x = self.slow_bn1(slow_x)
        slow_x = self.slow_relu(slow_x)
        slow_x = self.slow_maxpool(slow_x)
        
        # Fast pathway
        fast_x = self.fast_conv1(x)
        fast_x = self.fast_bn1(fast_x)
        fast_x = self.fast_relu(fast_x)
        fast_x = self.fast_maxpool(fast_x)
        
        # Lateral connection from fast to slow
        fast_x_lateral = self.lateral_p1(fast_x)
        slow_x = slow_x + fast_x_lateral
        
        # Res2
        slow_x = self.slow_res2(slow_x)
        fast_x = self.fast_res2(fast_x)
        fast_x_lateral = self.lateral_res2(fast_x)
        slow_x = slow_x + fast_x_lateral
        
        # Res3
        slow_x = self.slow_res3(slow_x)
        fast_x = self.fast_res3(fast_x)
        fast_x_lateral = self.lateral_res3(fast_x)
        slow_x = slow_x + fast_x_lateral
        
        # Res4
        slow_x = self.slow_res4(slow_x)
        fast_x = self.fast_res4(fast_x)
        fast_x_lateral = self.lateral_res4(fast_x)
        slow_x = slow_x + fast_x_lateral
        
        # Res5
        slow_x = self.slow_res5(slow_x)
        fast_x = self.fast_res5(fast_x)
        
        # Global average pooling
        slow_x = self.avgpool(slow_x)
        fast_x = self.avgpool(fast_x)
        
        slow_x = slow_x.view(B, -1)
        fast_x = fast_x.view(B, -1)
        
        # Concatenate slow and fast features
        x = torch.cat([slow_x, fast_x], dim=1)
        return self.proj(x)


@register_backbone('r3d_18', description='R3D-18 3D ResNet视频特征提取器', modality='video')
class R3D18(nn.Module):
    """R3D-18: 3D ResNet-18 for video classification
    
    使用3D卷积的ResNet-18变体
    """
    def __init__(self, img_size=224, in_chans=3, feature_dim=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        try:
            from torchvision.models.video import r3d_18
            backbone = r3d_18(pretrained=True)
            self.features = backbone
            self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
        except:
            # 简单实现
            self.features = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.AdaptiveAvgPool3d(1)
            )
            self.proj = nn.Linear(64, feature_dim)
    
    def forward(self, x):
        # x: (B, T, C, H, W) or (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('mc3_18', description='MC3-18 Mixed Convolution视频特征提取器', modality='video')
class MC318(nn.Module):
    """MC3-18: Mixed Convolution Network for video classification
    
    混合使用2D和3D卷积
    """
    def __init__(self, img_size=224, in_chans=3, feature_dim=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        try:
            from torchvision.models.video import mc3_18
            backbone = mc3_18(pretrained=True)
            self.features = backbone
            self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
        except:
            # 简单实现
            self.features = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.AdaptiveAvgPool3d(1)
            )
            self.proj = nn.Linear(64, feature_dim)
    
    def forward(self, x):
        # x: (B, T, C, H, W) or (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('r2plus1d_18', description='R(2+1)D-18视频特征提取器', modality='video')
class R2Plus1D18(nn.Module):
    """R(2+1)D-18: Spatiotemporal Decomposed ResNet for video classification
    
    将3D卷积分解为空间和时间两个部分
    """
    def __init__(self, img_size=224, in_chans=3, feature_dim=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        try:
            from torchvision.models.video import r2plus1d_18
            backbone = r2plus1d_18(pretrained=True)
            self.features = backbone
            self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
        except:
            # 简单实现
            self.features = nn.Sequential(
                nn.Conv3d(3, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
                nn.BatchNorm3d(45),
                nn.ReLU(inplace=True),
                nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.AdaptiveAvgPool3d(1)
            )
            self.proj = nn.Linear(64, feature_dim)
    
    def forward(self, x):
        # x: (B, T, C, H, W) or (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)