"""音频 backbone 模型（基于梅尔频谱图）。

Stageable（支持中期融合）:
    audiocnn, audiocnn_deep, audioresnet, audio_vggish

非 stageable:
    ast, audio_transformer_small, audio_resnet50, audio_mobilenet_v2,
    audio_efficientnet_b0, audio_ast, audio_tcn, audio_rawnet
"""

import torch
import torch.nn as nn

from .common import ensure_4d, make_mlp, ResBlock2D, PositionalEncoding
from ..registry import register_backbone
from ..backbone_base import StageableBackbone, BaseBackbone


# ==============================================================================
# Stageable 音频模型
# ==============================================================================

@register_backbone('audiocnn', description='CNN 音频特征提取器（支持 staged forward）', modality='audio')
class AudioCNN(StageableBackbone):
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
        self.proj = make_mlp(128, feature_dim, dropout=dropout, final_activation=True)

    def init_state(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        return ensure_4d(x)

    def forward_stage(self, state, stage_idx):
        return self.stages[stage_idx](state)

    def forward_head(self, state):
        return self.proj(self.pool(state).flatten(1))


@register_backbone('audiocnn_stageable', description='Stageable AudioCNN（别名，同 audiocnn）', modality='audio')
class AudioCNNStageable(AudioCNN):
    """向后兼容别名，功能与 AudioCNN 完全相同。"""


@register_backbone('audiocnn_deep', description='深层 CNN 音频特征提取器（支持 staged forward）', modality='audio')
class AudioCNNDeep(StageableBackbone):
    num_stages = 4

    def __init__(self, feature_dim=512, n_mels=128, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.stage_dims = [32, 64, 128, 256]

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = make_mlp(256, feature_dim, dropout=dropout, final_activation=True)

    def init_state(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        return ensure_4d(x)

    def forward_stage(self, state, stage_idx):
        return self.stages[stage_idx](state)

    def forward_head(self, state):
        return self.proj(self.pool(state).flatten(1))


@register_backbone('audioresnet', description='ResNet 风格音频特征提取器（支持 staged forward）', modality='audio')
class AudioResNet(StageableBackbone):
    num_stages = 4

    def __init__(self, feature_dim=512, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stages = nn.ModuleList([
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            nn.Identity(),  # stage 3: placeholder, dims handled by stage 2
        ])
        self.stage_dims = [64, 128, 256, 256]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = make_mlp(256, feature_dim, dropout=dropout, final_activation=True)

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks, stride=1):
        layers = [ResBlock2D(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock2D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def init_state(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        return self.stem(ensure_4d(x))

    def forward_stage(self, state, stage_idx):
        return self.stages[stage_idx](state)

    def forward_head(self, state):
        return self.proj(self.pool(state).flatten(1))


@register_backbone('audio_vggish', description='VGGish 风格音频特征提取器（支持 staged forward）', modality='audio')
class AudioVGGish(StageableBackbone):
    num_stages = 4

    def __init__(self, feature_dim=128, n_mels=128, dropout=0.5, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.stage_dims = [64, 128, 256, 512]

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, feature_dim),
        )

    def init_state(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        return ensure_4d(x)

    def forward_stage(self, state, stage_idx):
        return self.stages[stage_idx](state)

    def forward_head(self, state):
        return self.proj(self.pool(state).flatten(1))


# ==============================================================================
# 非 stageable 音频模型
# ==============================================================================

@register_backbone('ast', description='Audio Spectrogram Transformer (AST)', modality='audio')
class AudioSpectrogramTransformer(BaseBackbone):
    def __init__(self, feature_dim=768, n_mels=128, patch_size=16,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 dropout=0.1, max_len=1024, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (n_mels // patch_size) * (max_len // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout, batch_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, feature_dim) if feature_dim != embed_dim else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = ensure_4d(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj(x[:, 0])


@register_backbone('audio_transformer_small', description='小型 Transformer 音频特征提取器', modality='audio')
class AudioTransformerSmall(BaseBackbone):
    def __init__(self, feature_dim=256, n_mels=128, embed_dim=128,
                 num_heads=4, num_layers=4, dim_feedforward=512, dropout=0.1,
                 max_len=1024, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.spectrogram_embed = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.pos_encoder = PositionalEncoding(embed_dim, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.embed_dim = embed_dim

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = ensure_4d(x)
        x = self.spectrogram_embed(x)
        x = x.squeeze(2).transpose(1, 2)

        if x.size(-1) != self.embed_dim:
            x = nn.functional.pad(x, (0, self.embed_dim - x.size(-1)))

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.proj(x[:, 0])


@register_backbone('audio_resnet50', description='ResNet50 音频特征提取器', modality='audio')
class AudioResNet50(BaseBackbone):
    def __init__(self, feature_dim=2048, n_mels=128, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        try:
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                *list(resnet.children())[1:-1],
            )
            self.proj = nn.Linear(2048, feature_dim) if feature_dim != 2048 else nn.Identity()
        except Exception:
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(64, feature_dim)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = ensure_4d(x)
        x = self.features(x)
        return self.proj(x.flatten(1))


@register_backbone('audio_mobilenet_v2', description='MobileNet-V2 音频特征提取器', modality='audio')
class AudioMobileNetV2(BaseBackbone):
    def __init__(self, feature_dim=1280, n_mels=128, dropout=0.2, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        try:
            import torchvision.models as models
            mobilenet = models.mobilenet_v2(pretrained=True)
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
                *list(mobilenet.children())[1:-1],
            )
            self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()
        except Exception:
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(32, feature_dim)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = ensure_4d(x)
        x = self.features(x)
        return self.proj(x.flatten(1))


@register_backbone('audio_efficientnet_b0', description='EfficientNet-B0 音频特征提取器', modality='audio')
class AudioEfficientNetB0(BaseBackbone):
    def __init__(self, feature_dim=1280, n_mels=128, dropout=0.2, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        try:
            import torchvision.models as models
            efficientnet = models.efficientnet_b0(pretrained=True)
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.SiLU(inplace=True),
                *list(efficientnet.children())[1:-1],
            )
            self.proj = nn.Linear(1280, feature_dim) if feature_dim != 1280 else nn.Identity()
        except Exception:
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(32, feature_dim)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = ensure_4d(x)
        x = self.features(x)
        return self.proj(x.flatten(1))


@register_backbone('audio_ast', description='音频 Transformer (Mel Spectrogram)', modality='audio')
class AudioAST(BaseBackbone):
    def __init__(self, feature_dim=512, n_mels=128, patch_size=(16, 16),
                 embed_dim=256, depth=4, num_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, feature_dim),
        )

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = ensure_4d(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.transformer(x)
        return self.proj(x[:, 0])


@register_backbone('audio_tcn', description='CNN+TCN 音频特征提取器', modality='audio')
class AudioTCN(BaseBackbone):
    def __init__(self, feature_dim=512, n_mels=128, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.tcn = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=8, dilation=8), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = make_mlp(128, feature_dim, dropout=dropout, final_activation=True)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        x = ensure_4d(x)
        x = self.cnn(x)
        x = x.mean(dim=2)  # 压缩频率维 → [B, C, T]
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


@register_backbone('audio_rawnet', description='原始波形音频特征提取器', modality='audio')
class AudioRawNet(BaseBackbone):
    def __init__(self, feature_dim=512, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 5, stride=2, padding=2), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = make_mlp(256, feature_dim, dropout=dropout, final_activation=True)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)
