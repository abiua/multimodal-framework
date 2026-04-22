"""文本模型"""
import copy

import torch
import torch.nn as nn
from ..registry import register_backbone

@register_backbone('text_transformer_small_stageable',
                   description='Stageable 小型Transformer文本特征提取器',
                   modality='text')
class TextTransformerSmallStageable(nn.Module):
    num_stages = 4

    def __init__(self, feature_dim=256, vocab_size=30000, embed_dim=128,
                 num_heads=4, num_layers=4, dim_feedforward=512, dropout=0.1,
                 max_len=512, **kwargs):
        super().__init__()
        if num_layers != 4:
            raise ValueError("当前 stageable 版本固定要求 num_layers=4")

        self.feature_dim = feature_dim
        self.stage_dims = [embed_dim] * 4

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(4)])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.proj = nn.Linear(embed_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def init_state(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)   # [B, L, C]

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        x = self.pos_encoder(x)
        return {"x": x, "attention_mask": attention_mask}

    def forward_stage(self, state, stage_idx: int):
        x = state["x"]
        attention_mask = state["attention_mask"]

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
            x = self.layers[stage_idx](x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.layers[stage_idx](x)

        out = dict(state)
        out["x"] = x
        return out

    def forward_head(self, state):
        cls_feat = state["x"][:, 0, :]
        cls_feat = self.dropout(cls_feat)
        return self.proj(cls_feat)

    def forward(self, input_ids, attention_mask=None):
        state = self.init_state(input_ids=input_ids, attention_mask=attention_mask)
        for stage_idx in range(self.num_stages):
            state = self.forward_stage(state, stage_idx)
        return self.forward_head(state)


@register_backbone('textlstm', description='LSTM文本特征提取器', modality='text')
class TextLSTM(nn.Module):
    def __init__(self, feature_dim=512, vocab_size=30000, embed_dim=256, 
                 hidden_dim=512, num_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, feature_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(x)
        
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.proj(hidden)


@register_backbone('textgru', description='GRU文本特征提取器', modality='text')
class TextGRU(nn.Module):
    def __init__(self, feature_dim=512, vocab_size=30000, embed_dim=256,
                 hidden_dim=512, num_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, feature_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(x)
        
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.proj(hidden)


@register_backbone('textcnn', description='CNN文本特征提取器', modality='text')
class TextCNN(nn.Module):
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
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        conv_outs = []
        for conv in self.convs:
            out = torch.relu(conv(x))
            out = torch.max(out, dim=2)[0]  # max pooling
            conv_outs.append(out)
        
        x = torch.cat(conv_outs, dim=1)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    """位置编码模块"""
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


@register_backbone('transformer_encoder', description='Transformer编码器文本特征提取器', modality='text')
class TransformerEncoder(nn.Module):
    """Transformer编码器文本特征提取器
    
    使用标准的Transformer编码器架构
    """
    def __init__(self, feature_dim=512, vocab_size=30000, embed_dim=256,
                 num_heads=8, num_layers=6, dim_feedforward=2048, dropout=0.1,
                 max_len=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len, dropout)
        
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
    
    def forward(self, input_ids, attention_mask=None):
        # Token embedding
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # Add CLS token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Adjust attention mask for CLS token
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        if attention_mask is not None:
            # Convert attention mask to transformer format (0 for attend, -inf for ignore)
            src_key_padding_mask = (attention_mask == 0)
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer_encoder(x)
        
        # Use CLS token representation
        x = x[:, 0, :]  # (batch, embed_dim)
        return self.proj(x)


@register_backbone('bert_base', description='BERT Base文本特征提取器', modality='text')
class BERTBase(nn.Module):
    """BERT Base文本特征提取器
    
    使用HuggingFace的BERT模型作为特征提取器
    """
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        try:
            from transformers import BertModel, BertConfig
            
            if pretrained:
                self.bert = BertModel.from_pretrained('bert-base-uncased')
            else:
                config = BertConfig()
                self.bert = BertModel(config)
            
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except ImportError:
            # 回退到简单实现
            self.bert = self._build_simple_bert()
            self.proj = nn.Linear(768, feature_dim)
    
    def _build_simple_bert(self):
        """简化的BERT实现"""
        class SimpleBERT(nn.Module):
            def __init__(self, vocab_size=30000, embed_dim=768, num_heads=12, num_layers=12, dim_feedforward=3072):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.pos_encoder = PositionalEncoding(embed_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads,
                    dim_feedforward=dim_feedforward, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.pooler = nn.Linear(embed_dim, embed_dim)
                self.activation = nn.Tanh()
            
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.pos_encoder(x)
                if attention_mask is not None:
                    src_key_padding_mask = (attention_mask == 0)
                    x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
                else:
                    x = self.encoder(x)
                pooled_output = self.activation(self.pooler(x[:, 0, :]))
                return x, pooled_output
        
        return SimpleBERT()
    
    def forward(self, input_ids, attention_mask=None):
        if hasattr(self.bert, 'forward') and callable(getattr(self.bert, 'forward')):
            try:
                outputs = self.bert(input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
            except:
                outputs = self.bert(input_ids, attention_mask=attention_mask)
                pooled_output = outputs[1] if isinstance(outputs, tuple) else outputs[:, 0, :]
        else:
            _, pooled_output = self.bert(input_ids, attention_mask=attention_mask)
        
        return self.proj(pooled_output)


@register_backbone('roberta_base', description='RoBERTa Base文本特征提取器', modality='text')
class RoBERTaBase(nn.Module):
    """RoBERTa Base文本特征提取器
    
    使用HuggingFace的RoBERTa模型作为特征提取器
    """
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        try:
            from transformers import RobertaModel, RobertaConfig
            
            if pretrained:
                self.roberta = RobertaModel.from_pretrained('roberta-base')
            else:
                config = RobertaConfig()
                self.roberta = RobertaModel(config)
            
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except ImportError:
            # 回退到简单实现
            self.roberta = self._build_simple_roberta()
            self.proj = nn.Linear(768, feature_dim)
    
    def _build_simple_roberta(self):
        """简化的RoBERTa实现"""
        class SimpleRoBERTa(nn.Module):
            def __init__(self, vocab_size=50265, embed_dim=768, num_heads=12, num_layers=12, dim_feedforward=3072):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
                self.pos_encoder = PositionalEncoding(embed_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads,
                    dim_feedforward=dim_feedforward, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.pos_encoder(x)
                if attention_mask is not None:
                    src_key_padding_mask = (attention_mask == 0)
                    x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
                else:
                    x = self.encoder(x)
                return x, x[:, 0, :]
        
        return SimpleRoBERTa()
    
    def forward(self, input_ids, attention_mask=None):
        if hasattr(self.roberta, 'forward') and callable(getattr(self.roberta, 'forward')):
            try:
                outputs = self.roberta(input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
            except:
                outputs = self.roberta(input_ids, attention_mask=attention_mask)
                pooled_output = outputs[1] if isinstance(outputs, tuple) else outputs[:, 0, :]
        else:
            _, pooled_output = self.roberta(input_ids, attention_mask=attention_mask)
        
        return self.proj(pooled_output)


@register_backbone('distilbert_base', description='DistilBERT Base文本特征提取器', modality='text')
class DistilBERTBase(nn.Module):
    """DistilBERT Base文本特征提取器
    
    蒸馏版的BERT模型，更轻量级
    """
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        try:
            from transformers import DistilBertModel, DistilBertConfig
            
            if pretrained:
                self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            else:
                config = DistilBertConfig()
                self.distilbert = DistilBertModel(config)
            
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except ImportError:
            # 回退到简单实现
            self.distilbert = self._build_simple_distilbert()
            self.proj = nn.Linear(768, feature_dim)
    
    def _build_simple_distilbert(self):
        """简化的DistilBERT实现"""
        class SimpleDistilBERT(nn.Module):
            def __init__(self, vocab_size=30522, embed_dim=768, num_heads=12, num_layers=6, dim_feedforward=3072):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.pos_encoder = PositionalEncoding(embed_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads,
                    dim_feedforward=dim_feedforward, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.pos_encoder(x)
                if attention_mask is not None:
                    src_key_padding_mask = (attention_mask == 0)
                    x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
                else:
                    x = self.encoder(x)
                return x
        
        return SimpleDistilBERT()
    
    def forward(self, input_ids, attention_mask=None):
        if hasattr(self.distilbert, 'forward') and callable(getattr(self.distilbert, 'forward')):
            outputs = self.distilbert(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = self.distilbert(input_ids, attention_mask=attention_mask)
        
        # 使用第一个token（CLS）的表示
        pooled_output = hidden_states[:, 0, :]
        return self.proj(pooled_output)


@register_backbone('text_transformer_small', description='小型Transformer文本特征提取器', modality='text')
class TextTransformerSmall(nn.Module):
    """小型Transformer文本特征提取器
    
    适用于资源受限场景的轻量级Transformer
    """
    def __init__(self, feature_dim=256, vocab_size=30000, embed_dim=128,
                 num_heads=4, num_layers=4, dim_feedforward=512, dropout=0.1,
                 max_len=512, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len, dropout)
        
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
    
    def forward(self, input_ids, attention_mask=None):
        # Token embedding
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # Add CLS token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Adjust attention mask for CLS token
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer_encoder(x)
        
        # Use CLS token representation
        x = x[:, 0, :]  # (batch, embed_dim)
        return self.proj(x)


@register_backbone('albert_base', description='ALBERT Base文本特征提取器', modality='text')
class ALBERTBase(nn.Module):
    """ALBERT Base文本特征提取器
    
    轻量级的BERT变体，使用参数共享
    """
    def __init__(self, feature_dim=768, pretrained=True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        
        try:
            from transformers import AlbertModel, AlbertConfig
            
            if pretrained:
                self.albert = AlbertModel.from_pretrained('albert-base-v2')
            else:
                config = AlbertConfig()
                self.albert = AlbertModel(config)
            
            self.proj = nn.Linear(768, feature_dim) if feature_dim != 768 else nn.Identity()
        except ImportError:
            # 回退到简单实现
            self.albert = self._build_simple_albert()
            self.proj = nn.Linear(768, feature_dim)
    
    def _build_simple_albert(self):
        """简化的ALBERT实现"""
        class SimpleALBERT(nn.Module):
            def __init__(self, vocab_size=30000, embed_dim=128, hidden_dim=768, num_heads=12, num_layers=12, num_groups=1):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.embedding_projection = nn.Linear(embed_dim, hidden_dim)
                self.pos_encoder = PositionalEncoding(hidden_dim)
                # 使用参数共享的Transformer层
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=num_heads,
                    dim_feedforward=hidden_dim*4, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.embedding_projection(x)
                x = self.pos_encoder(x)
                if attention_mask is not None:
                    src_key_padding_mask = (attention_mask == 0)
                    x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
                else:
                    x = self.encoder(x)
                return x, x[:, 0, :]
        
        return SimpleALBERT()
    
    def forward(self, input_ids, attention_mask=None):
        if hasattr(self.albert, 'forward') and callable(getattr(self.albert, 'forward')):
            try:
                outputs = self.albert(input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
            except:
                outputs = self.albert(input_ids, attention_mask=attention_mask)
                pooled_output = outputs[1] if isinstance(outputs, tuple) else outputs[:, 0, :]
        else:
            _, pooled_output = self.albert(input_ids, attention_mask=attention_mask)
        
        return self.proj(pooled_output)