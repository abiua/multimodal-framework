"""文本加载器"""

import torch
from typing import Dict, Any, Optional
from ..registry import BaseLoader, register_loader


@register_loader('text_loader', description='标准文本加载器', modality='text')
class TextLoader(BaseLoader):
    """标准文本加载器"""
    
    def __init__(self, max_length: int = 128, vocab_size: int = 30000, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.vocab = self._build_vocab()
    
    def _build_vocab(self):
        return {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
    
    def load(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        tokens = ['[CLS]'] + list(text.lower())[:self.max_length - 2] + ['[SEP]']
        token_ids = [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]
        
        # 填充或截断
        if len(token_ids) < self.max_length:
            token_ids += [self.vocab['[PAD]']] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        attention_mask = [1 if tid != self.vocab['[PAD]'] else 0 for tid in token_ids]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def get_transform(self, is_training: bool = True):
        return self.tokenize


@register_loader('text_loader_char', description='字符级文本加载器', modality='text')
class TextLoaderChar(BaseLoader):
    """字符级文本加载器"""
    
    def __init__(self, max_length: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
    
    def load(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        # 简单字符编码
        chars = list(text[:self.max_length])
        char_ids = [ord(c) % 256 for c in chars]
        
        if len(char_ids) < self.max_length:
            char_ids += [0] * (self.max_length - len(char_ids))
        
        attention_mask = [1 if cid != 0 else 0 for cid in char_ids]
        
        return {
            'input_ids': torch.tensor(char_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def get_transform(self, is_training: bool = True):
        return self.tokenize