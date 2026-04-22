"""音频模型"""

import torch
import torch.nn as nn
from ..registry import register_backbone

@register_backbone('audiocnn_stageable', description='Stageable AudioCNN', modality='audio')
class AudioCNNStageable(nn.Module):
    num_stages = 4

    def __init__(self, feature_dim=512, n_mels=128, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.stage_dims = [32, 64, 128, 128]
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def init_state(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)   # [B, 1, H, W]
        return x

    def forward_stage(self, state, stage_idx: int):
        return self.stages[stage_idx](state)

    def forward_head(self, state):
        x = self.pool(state).flatten(1)
        return self.proj(x)

@register_backbone('audiocnn', description='CNN音频特征提取器（基于梅尔频谱图）', modality='audio')
class AudioCNN(nn.Module):
    def __init__(self, feature_dim=512, n_mels=128, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_mels = n_mels
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.proj = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('audiocnn_deep', description='深层CNN音频特征提取器', modality='audio')
class AudioCNNDeep(nn.Module):
    def __init__(self, feature_dim=512, n_mels=128, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.proj = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('audioresnet', description='ResNet风格音频特征提取器', modality='audio')
class AudioResNet(nn.Module):
    """基于ResNet的音频特征提取器"""
    
    def __init__(self, feature_dim=512, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 初始卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 残差块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.proj = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
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
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class AudioPositionalEncoding(nn.Module):
    """音频位置编码模块"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


@register_backbone('ast', description='Audio Spectrogram Transformer (AST)音频特征提取器', modality='audio')
class AudioSpectrogramTransformer(nn.Module):
    """Audio Spectrogram Transformer (AST)
    
    基于Vision Transformer的音频分类模型，将音频频谱图视为图像
    """
    def __init__(self, feature_dim=768, n_mels=128, patch_size=16, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 dropout=0.1, max_len=1024, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        # Patch embedding (将频谱图分割成patch)
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 计算patch数量
        self.num_patches = (n_mels // patch_size) * (max_len // patch_size)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                batch_first=True
            )
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Projection layer
        self.proj = nn.Linear(embed_dim, feature_dim) if feature_dim != embed_dim else nn.Identity()
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # x: (batch, n_mels, time) 或 (batch, 1, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, embed_dim, h, w)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        # Add CLS token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Use CLS token representation
        x = x[:, 0, :]  # (batch, embed_dim)
        return self.proj(x)


@register_backbone('audio_transformer_small', description='小型Transformer音频特征提取器', modality='audio')
class AudioTransformerSmall(nn.Module):
    """小型Transformer音频特征提取器
    
    适用于资源受限场景的轻量级Transformer音频模型
    """
    def __init__(self, feature_dim=256, n_mels=128, embed_dim=128,
                 num_heads=4, num_layers=4, dim_feedforward=512, dropout=0.1,
                 max_len=1024, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 频谱图特征提取
        self.spectrogram_embed = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )
        
        # Positional encoding
        self.pos_encoder = AudioPositionalEncoding(embed_dim, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection layer
        self.proj = nn.Linear(embed_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # x: (batch, n_mels, time) 或 (batch, 1, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        # 特征提取
        x = self.spectrogram_embed(x)  # (batch, 64, 1, time_reduced)
        x = x.squeeze(2).transpose(1, 2)  # (batch, time_reduced, 64)
        
        # 投影到embed_dim
        if x.size(-1) != 128:  # 假设embed_dim=128
            x = nn.functional.pad(x, (0, 128 - x.size(-1)))
        
        # Add CLS token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Use CLS token representation
        x = x[:, 0, :]  # (batch, embed_dim)
        return self.proj(x)


@register_backbone('audio_resnet50', description='ResNet50音频特征提取器', modality='audio')
class AudioResNet50(nn.Module):
    """ResNet50音频特征提取器
    
    使用ResNet50架构处理音频梅尔频谱图
    """
    def __init__(self, feature_dim=2048, n_mels=128, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        try:
            import torchvision.models as models
            # 使用预训练的ResNet50
            resnet = models.resnet50(pretrained=True)
            # 修改第一层以接受单通道输入
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                *list(resnet.children())[1:-1]  # 去掉原始的第一层和最后的fc层
            )
            self.proj = nn.Linear(2048, feature_dim) if feature_dim != 2048 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.proj = nn.Linear(64, feature_dim)
    
    def forward(self, x):
        # x: (batch, n_mels, time) 或 (batch, 1, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('audio_vggish', description='VGGish风格音频特征提取器', modality='audio')
class AudioVGGish(nn.Module):
    """VGGish风格音频特征提取器
    
    基于VGGish架构的音频特征提取器
    """
    def __init__(self, feature_dim=128, n_mels=128, dropout=0.5, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        # VGGish风格的卷积层
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, feature_dim)
        )
    
    def forward(self, x):
        # x: (batch, n_mels, time) 或 (batch, 1, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@register_backbone('audio_mobilenet_v2', description='MobileNet-V2音频特征提取器', modality='audio')
class AudioMobileNetV2(nn.Module):
    """MobileNet-V2音频特征提取器
    
    使用MobileNet-V2架构处理音频梅尔频谱图
    """
    def __init__(self, feature_dim=1280, n_mels=128, dropout=0.2, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        try:
            import torchvision.models as models
            # 使用预训练的MobileNet-V2
            mobilenet = models.mobilenet_v2(pretrained=True)
            # 修改第一层以接受单通道输入
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
                *list(mobilenet.children())[1:-1]  # 去掉原始的第一层和最后的分类器
            )
            self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.proj = nn.Linear(32, feature_dim)
    
    def forward(self, x):
        # x: (batch, n_mels, time) 或 (batch, 1, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


@register_backbone('audio_efficientnet_b0', description='EfficientNet-B0音频特征提取器', modality='audio')
class AudioEfficientNetB0(nn.Module):
    """EfficientNet-B0音频特征提取器
    
    使用EfficientNet-B0架构处理音频梅尔频谱图
    """
    def __init__(self, feature_dim=1280, n_mels=128, dropout=0.2, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        try:
            import torchvision.models as models
            # 使用预训练的EfficientNet-B0
            efficientnet = models.efficientnet_b0(pretrained=True)
            # 修改第一层以接受单通道输入
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                *list(efficientnet.children())[1:-1]  # 去掉原始的第一层和最后的分类器
            )
            self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()
        except:
            # 回退到简单CNN
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.proj = nn.Linear(32, feature_dim)
    
    def forward(self, x):
        # x: (batch, n_mels, time) 或 (batch, 1, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)

# =========================================================================================================================
@register_backbone('audio_ast', description='音频Transformer特征提取器（Mel Spectrogram）', modality='audio')
class AudioAST(nn.Module):
    def __init__(self, feature_dim=512, n_mels=128, patch_size=(16,16), 
                 embed_dim=256, depth=4, num_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.patch_embed = nn.Conv2d(1, embed_dim, 
                                    kernel_size=patch_size, 
                                    stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None  # 动态生成
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, feature_dim)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.patch_embed(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        x = self.transformer(x)
        x = x[:, 0]  # CLS token
        
        return self.proj(x)

@register_backbone('audio_tcn', description='CNN+TCN音频特征提取器', modality='audio')
class AudioTCN(nn.Module):
    def __init__(self, feature_dim=512, n_mels=128, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.tcn = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=8, dilation=8),
            nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.proj = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.cnn(x)  # [B, C, H, W]
        x = x.mean(dim=2)  # 压缩频率维 → [B, C, T]
        
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        
        return self.proj(x)

@register_backbone('audio_rawnet', description='原始波形音频特征提取器', modality='audio')
class AudioRawNet(nn.Module):
    def __init__(self, feature_dim=512, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.proj = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        
        return self.proj(x)