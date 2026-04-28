"""文本 backbone 模型。

Stageable（支持中期融合）:
    text_transformer_small

非 stageable:
    textlstm, textgru, textcnn, transformer_encoder,
    bert_base, roberta_base, distilbert_base, albert_base
"""

import copy
import torch
import torch.nn as nn

from .common import PositionalEncoding, HuggingFaceWrapper
from ..registry import register_backbone
from ..backbone_base import StageableBackbone, BaseBackbone


# ==============================================================================
# Stageable 文本模型
# ==============================================================================

@register_backbone('text_transformer_small', description='小型 Transformer 文本特征提取器（支持 staged forward）', modality='text')
class TextTransformerSmall(StageableBackbone):
    num_stages = 4

    def __init__(self, feature_dim=256, vocab_size=30000, embed_dim=128,
                 num_heads=4, num_layers=4, dim_feedforward=512, dropout=0.1,
                 max_len=512, **kwargs):
        super().__init__()
        if num_layers != 4:
            raise ValueError("text_transformer_small 要求 num_layers=4（4 stages）")

        self.feature_dim = feature_dim
        self.stage_dims = [embed_dim] * 4

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(4)])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.proj = nn.Linear(embed_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def init_state(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))

        x = self.embedding(input_ids)
        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        x = self.pos_encoder(x)
        return {"x": x, "attention_mask": attention_mask}

    def forward_stage(self, state, stage_idx):
        x = state["x"]
        mask = state["attention_mask"]
        src_key_padding_mask = None if mask is None else (mask == 0)
        x = self.layers[stage_idx](x, src_key_padding_mask=src_key_padding_mask)
        out = dict(state)
        out["x"] = x
        return out

    def forward_head(self, state):
        cls_feat = state["x"][:, 0, :]
        return self.proj(self.dropout(cls_feat))


@register_backbone('text_transformer_small_stageable',
                   description='Stageable 小型 Transformer（别名，同 text_transformer_small）',
                   modality='text')
class TextTransformerSmallStageable(TextTransformerSmall):
    """向后兼容别名，功能与 TextTransformerSmall 完全相同。"""


# ==============================================================================
# 非 stageable 文本模型
# ==============================================================================

@register_backbone('textlstm', description='LSTM 文本特征提取器', modality='text')
class TextLSTM(BaseBackbone):
    def __init__(self, feature_dim=512, vocab_size=30000, embed_dim=256,
                 hidden_dim=512, num_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))

        x = self.embedding(input_ids)
        x = self.dropout(x)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(x)

        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.proj(hidden)


@register_backbone('textgru', description='GRU 文本特征提取器', modality='text')
class TextGRU(BaseBackbone):
    def __init__(self, feature_dim=512, vocab_size=30000, embed_dim=256,
                 hidden_dim=512, num_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))

        x = self.embedding(input_ids)
        x = self.dropout(x)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(x)

        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.proj(hidden)


@register_backbone('textcnn', description='CNN 文本特征提取器', modality='text')
class TextCNN(BaseBackbone):
    def __init__(self, feature_dim=512, vocab_size=30000, embed_dim=256,
                 num_filters=128, filter_sizes=(2, 3, 4, 5), dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        total_filters = num_filters * len(filter_sizes)
        self.proj = nn.Sequential(
            nn.Linear(total_filters, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))

        x = self.embedding(input_ids)
        x = self.dropout(x)
        x = x.transpose(1, 2)

        conv_outs = [torch.max(torch.relu(conv(x)), dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        return self.proj(x)


@register_backbone('transformer_encoder', description='Transformer 编码器文本特征提取器', modality='text')
class TransformerEncoder(BaseBackbone):
    def __init__(self, feature_dim=512, vocab_size=30000, embed_dim=256,
                 num_heads=8, num_layers=6, dim_feedforward=2048, dropout=0.1,
                 max_len=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
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

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = next(v for v in kwargs.values() if isinstance(v, torch.Tensor))

        x = self.embedding(input_ids)
        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        x = self.pos_encoder(x)
        src_key_padding_mask = None if attention_mask is None else (attention_mask == 0)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.proj(x[:, 0])


# ==============================================================================
# HuggingFace 包装模型
# ==============================================================================

@register_backbone('bert_base', description='BERT Base 文本特征提取器', modality='text')
class BERTBase(HuggingFaceWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_name='bert-base-uncased',
            model_cls_name='BertModel',
            config_cls_name='BertConfig',
            default_dim=768,
            fallback_vocab_size=30000,
            fallback_embed_dim=768,
            fallback_num_heads=12,
            fallback_num_layers=12,
            fallback_dim_feedforward=3072,
            fallback_pad_idx=0,
        )


@register_backbone('roberta_base', description='RoBERTa Base 文本特征提取器', modality='text')
class RoBERTaBase(HuggingFaceWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_name='roberta-base',
            model_cls_name='RobertaModel',
            config_cls_name='RobertaConfig',
            default_dim=768,
            fallback_vocab_size=50265,
            fallback_embed_dim=768,
            fallback_num_heads=12,
            fallback_num_layers=12,
            fallback_dim_feedforward=3072,
            fallback_pad_idx=1,
        )


@register_backbone('distilbert_base', description='DistilBERT Base 文本特征提取器', modality='text')
class DistilBERTBase(HuggingFaceWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_name='distilbert-base-uncased',
            model_cls_name='DistilBertModel',
            config_cls_name='DistilBertConfig',
            default_dim=768,
            fallback_vocab_size=30522,
            fallback_embed_dim=768,
            fallback_num_heads=12,
            fallback_num_layers=6,
            fallback_dim_feedforward=3072,
            fallback_pad_idx=0,
        )


@register_backbone('albert_base', description='ALBERT Base 文本特征提取器', modality='text')
class ALBERTBase(HuggingFaceWrapper):
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_name='albert-base-v2',
            model_cls_name='AlbertModel',
            config_cls_name='AlbertConfig',
            default_dim=768,
            fallback_vocab_size=30000,
            fallback_embed_dim=128,
            fallback_num_heads=12,
            fallback_num_layers=12,
            fallback_dim_feedforward=3072,
            fallback_pad_idx=0,
        )
