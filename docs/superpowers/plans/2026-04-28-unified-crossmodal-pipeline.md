# Unified Cross-Modal Pipeline 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建新的多模态流水线 — Stem(现有modelzoo) → Tokenization(统一D空间) → Cross-Modal Interaction(可配置堆叠Block) → Mid Fusion → Decision(预留接口) → Classifier，保留旧路径兼容。

**Architecture:** 方案A — 保留模态边界 `Dict[str, Tensor[B,N,D]]` 贯穿 Interaction 层。每个 InteractionBlock = SharedTransform + FusionStrategy，Fusion 通过注册表可插拔。最终 Pooling → Mid Fusion → Decision → Classifier。

**Tech Stack:** PyTorch, einops, omegaconf, 现有 ModelZoo/LoaderRegistry 不变

---

## 文件结构

```
models/
  backbone_base.py          # [修改] BaseBackbone 增加 tokenize() 可选方法
  tokenizer.py              # [新建] Tokenization 层
  fusion/
    __init__.py             # [新建] 导出 FusionRegistry
    registry.py             # [新建] FusionRegistry + BaseFusion
    strategies.py           # [新建] 内置4种融合策略
  interaction.py            # [新建] InteractionBlock
  decision.py               # [新建] DecisionModule 预留接口
  pipeline_v2.py            # [新建] MultimodalPipelineV2
  builder.py                # [修改] 增加 build_pipeline_v2() 方法
  __init__.py               # [修改] 导出新模块
  modelzoo/
    video_models.py         # [修改] 新增 VideoMAE backbone

utils/
  config.py                 # [修改] 新增 unified_pipeline 配置段

configs/
  default_multimodal_v2.yaml # [新建] 示例配置

tests/
  test_pipeline_v2.py       # [新建] 测试新流水线
```

---

### Task 1: BaseBackbone 增加 tokenize() 接口

**Files:**
- Modify: `models/backbone_base.py`

backbone_base.py 只加一个可选方法，零破坏性：

- [ ] **Step 1: 添加 `tokenize()` 方法到 `BaseBackbone`**

```python
# models/backbone_base.py — 在 BaseBackbone 类中增加以下方法

def tokenize(self, *args, **inputs) -> Dict[str, torch.Tensor]:
    """可选的 token 化输出。
    
    默认实现：调用 forward() 得到 [B, D]，包装为单 token [B, 1, D]。
    子类可重写以输出更丰富的 token 序列（如 feature map flatten）。
    
    Returns:
        Dict with keys:
            "tokens": Tensor [B, N, D] — token 序列
            "layout": str — "1d" (temporal) | "2d" (spatial) | "scalar" (single)
    """
    x = inputs.pop('x', None)
    if x is not None:
        inputs.setdefault('x', x)
    feat = self.forward(**inputs)  # [B, D]
    return {"tokens": feat.unsqueeze(1), "layout": "scalar"}  # [B, 1, D]
```

- [ ] **Step 2: 验证无破坏**

```bash
python -c "from models.backbone_base import BaseBackbone, StageableBackbone; print('OK')"
```

Expected: `OK`，现有导入不受影响。

- [ ] **Step 3: Commit**

```bash
git add models/backbone_base.py
git commit -m "feat: add tokenize() optional method to BaseBackbone"
```

---

### Task 2: Fusion Registry + BaseFusion + 内置策略

**Files:**
- Create: `models/fusion/__init__.py`
- Create: `models/fusion/registry.py`
- Create: `models/fusion/strategies.py`

- [ ] **Step 1: 创建 `__init__.py`**

```python
# models/fusion/__init__.py
from .registry import FusionRegistry, BaseFusion
from .strategies import (
    IdentityFusion,
    GateInjectionFusion,
    CrossAttentionFusion,
    TokenMixerFusion,
)
```

- [ ] **Step 2: 创建 `registry.py`**

```python
# models/fusion/registry.py
from abc import ABC, abstractmethod
from typing import Dict, Type
import torch
import torch.nn as nn


class BaseFusion(nn.Module, ABC):
    """跨模态融合基类。
    
    输入: tokens Dict[str, Tensor[B,N,D]]
    输出: tokens Dict[str, Tensor[B,N,D]]
    
    所有融合策略操作统一 token 空间，不改变 shape。
    """

    def __init__(self, modalities, dim, **kwargs):
        super().__init__()
        self.modalities = list(modalities)
        self.dim = dim

    @abstractmethod
    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...


class FusionRegistry:
    _fusions: Dict[str, Type[BaseFusion]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fusion_cls):
            cls._fusions[name] = fusion_cls
            return fusion_cls
        return decorator

    @classmethod
    def create(cls, name: str, modalities, dim: int, **kwargs) -> BaseFusion:
        if name == "none" or name == "identity":
            return IdentityFusion(modalities, dim)
        if name not in cls._fusions:
            raise KeyError(
                f"未知融合策略: '{name}'，可用: {list(cls._fusions.keys())}"
            )
        return cls._fusions[name](modalities=modalities, dim=dim, **kwargs)

    @classmethod
    def list_all(cls) -> list:
        return list(cls._fusions.keys()) + ["none", "identity"]
```

- [ ] **Step 3: 创建 `strategies.py`** — IdentityFusion

```python
# models/fusion/strategies.py (第一部分)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaseFusion, FusionRegistry


class IdentityFusion(BaseFusion):
    """无融合 — 各模态独立通过。"""

    def forward(self, tokens):
        return tokens


@FusionRegistry.register("gate")
class GateInjectionFusion(BaseFusion):
    """门控注入融合。
    
    各模态 token 池化 → 投影到公共空间 → 平均 → 门控注入回各模态。
    
    Extra kwargs:
        gate_hidden_dim: int = None  # 默认 dim // 2
        dropout: float = 0.0
    """

    def __init__(self, modalities, dim, gate_hidden_dim=None, dropout=0.0, **kwargs):
        super().__init__(modalities, dim)
        h = gate_hidden_dim or max(dim // 2, 32)

        self.to_common = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU())
            for m in modalities
        })
        self.to_injection = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(h, dim), nn.Tanh())
            for m in modalities
        })
        self.gate = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(h, dim), nn.Sigmoid())
            for m in modalities
        })
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, tokens):
        # 各模态池化到 [B, dim]
        pooled = {}
        for m in self.modalities:
            x = tokens[m]
            if x.dim() == 3:
                pooled[m] = x.mean(dim=1)
            elif x.dim() == 2:
                pooled[m] = x
            else:
                pooled[m] = x.flatten(1)

        # 投影到公共空间
        projected = [self.to_common[m](pooled[m]) for m in self.modalities]
        fused = torch.stack(projected, dim=0).mean(dim=0)  # [B, h]
        fused = self.dropout(fused)

        # 门控注入回各模态
        out = {}
        for m in self.modalities:
            inj = self.to_injection[m](fused)  # [B, dim]
            g = self.gate[m](fused)            # [B, dim]
            delta = g * inj

            x = tokens[m]
            if x.dim() == 3:
                delta = delta.unsqueeze(1)
            out[m] = x + delta
        return out
```

- [ ] **Step 4: 创建 `strategies.py`** — CrossAttentionFusion

```python
# models/fusion/strategies.py (第二部分)

@FusionRegistry.register("cross_attn")
class CrossAttentionFusion(BaseFusion):
    """跨模态注意力融合。
    
    每个模态用 cross-attention 从其他所有模态拉信息。
    Q = 当前模态, K/V = 所有其他模态 token concat。
    
    Extra kwargs:
        num_heads: int = 8
        dropout: float = 0.0
    """

    def __init__(self, modalities, dim, num_heads=8, dropout=0.0, **kwargs):
        super().__init__(modalities, dim)
        self.num_heads = num_heads

        self.cross_attns = nn.ModuleDict({
            m: nn.MultiheadAttention(
                dim, num_heads, dropout=dropout,
                batch_first=True,
            )
            for m in modalities
        })
        self.norms = nn.ModuleDict({
            m: nn.LayerNorm(dim) for m in modalities
        })

    def forward(self, tokens):
        out = {}
        for m in self.modalities:
            # Q: 当前模态 token
            q = tokens[m]  # [B, N_m, D]

            # K/V: 所有其他模态 token 拼接
            kv_list = [tokens[o] for o in self.modalities if o != m]
            if not kv_list:
                out[m] = tokens[m]
                continue
            kv = torch.cat(kv_list, dim=1)  # [B, N_other, D]

            attn_out, _ = self.cross_attns(
                query=q, key=kv, value=kv,
                need_weights=False,
            )
            out[m] = self.norms[m](tokens[m] + attn_out)
        return out


@FusionRegistry.register("token_mix")
class TokenMixerFusion(BaseFusion):
    """全 Token 混合融合。
    
    把所有模态 token concat → Self-Attention → split 回各模态。
    
    Extra kwargs:
        num_heads: int = 8
        dropout: float = 0.0
    """

    def __init__(self, modalities, dim, num_heads=8, dropout=0.0, **kwargs):
        super().__init__(modalities, dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tokens):
        # Concat 所有模态 token
        token_list = [tokens[m] for m in self.modalities]
        sizes = [t.shape[1] for t in token_list]
        x = torch.cat(token_list, dim=1)  # [B, N_total, D]

        # Self-Attention
        x = x + self.attn(self.norm(x), self.norm(x), self.norm(x))[0]
        x = x + self.mlp(self.norm2(x))

        # Split 回各模态
        out = {}
        start = 0
        for m, n in zip(self.modalities, sizes):
            out[m] = x[:, start:start + n, :]
            start += n
        return out
```

- [ ] **Step 5: 运行导入验证**

```bash
python -c "
from models.fusion.registry import FusionRegistry, BaseFusion
from models.fusion.strategies import (
    IdentityFusion, GateInjectionFusion, CrossAttentionFusion, TokenMixerFusion,
)
print('Available:', FusionRegistry.list_all())
# 创建测试
f = FusionRegistry.create('gate', modalities=['image','audio'], dim=256)
print(f'GateFusion created: {type(f).__name__}')
f = FusionRegistry.create('cross_attn', modalities=['image','audio'], dim=256, num_heads=4)
print(f'CrossAttnFusion created: {type(f).__name__}')
f = FusionRegistry.create('token_mix', modalities=['image','audio'], dim=256, num_heads=4)
print(f'TokenMixerFusion created: {type(f).__name__}')
f = FusionRegistry.create('none', modalities=['image','audio'], dim=256)
print(f'IdentityFusion created: {type(f).__name__}')
print('All fusion tests passed')
"
```

Expected: 4 个策略全部创建成功，无报错。

- [ ] **Step 6: Commit**

```bash
git add models/fusion/
git commit -m "feat: add FusionRegistry with 4 pluggable cross-modal fusion strategies"
```

---

### Task 3: Tokenization 层

**Files:**
- Create: `models/tokenizer.py`

- [ ] **Step 1: 创建 `models/tokenizer.py`**

```python
"""Tokenization 层 — 将各模态 stem 特征投影到统一 token 空间 [B, N, D]。

组件:
    ModalityProjection: C_modality → D（统一维度）
    ModalityEmbedding:   可学习模态身份嵌入
    PositionEncoding:    可学习位置编码（时间/空间）
    MultiModalTokenizer: 组合以上组件的完整 Tokenization 层
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ModalityProjection(nn.Module):
    """将各模态异构特征投影到统一维度 D。"""

    def __init__(self, feature_dims: Dict[str, int], unified_dim: int):
        super().__init__()
        self.projections = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(feature_dims[m]),
                nn.Linear(feature_dims[m], unified_dim),
                nn.GELU(),
            )
            for m in feature_dims
        })

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """features: {modality: Tensor[B, D_m]} → tokens: {modality: Tensor[B, 1, D]}"""
        return {
            m: self.projections[m](feat).unsqueeze(1)  # [B, 1, D]
            for m, feat in features.items()
        }


class ModalityEmbedding(nn.Module):
    """可学习的模态身份嵌入 — 加到每个模态的 token 上。

    用法:
        tokens[m] = tokens[m] + modal_emb(m)  # broadcast 到所有 token
    """

    def __init__(self, modalities: List[str], dim: int):
        super().__init__()
        self.embeddings = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(1, 1, dim))
            for m in modalities
        })
        for p in self.embeddings.values():
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: t + self.embeddings[m] for m, t in tokens.items()}


class PositionEncoding(nn.Module):
    """可学习位置编码。

    按模态配置是否启用和最大长度。

    Config 示例:
        pe_configs = {
            "image": {"enabled": True, "max_len": 197},   # 14x14 patches + 1
            "audio": {"enabled": True, "max_len": 197},
            "wave":  {"enabled": True, "max_len": 512},
        }
    """

    def __init__(
        self,
        modalities: List[str],
        dim: int,
        pe_configs: Optional[Dict[str, dict]] = None,
    ):
        super().__init__()
        self.pe = nn.ParameterDict()
        pe_configs = pe_configs or {}
        for m in modalities:
            cfg = pe_configs.get(m, {})
            if cfg.get("enabled", True):
                max_len = cfg.get("max_len", 256)
                self.pe[m] = nn.Parameter(torch.zeros(1, max_len, dim))
                nn.init.trunc_normal_(self.pe[m], std=0.02)
            else:
                self.pe[m] = None

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for m, t in tokens.items():
            if self.pe.get(m) is not None:
                n = t.shape[1]
                out[m] = t + self.pe[m][:, :n, :]
            else:
                out[m] = t
        return out


class MultiModalTokenizer(nn.Module):
    """完整的 Tokenization 层。

    输入: 各模态 stem 特征 {modality: Tensor[B, D_m]}
    输出: 统一 token 序列   {modality: Tensor[B, N_m, D]}

    流程:
        features → ModalityProjection → ModalityEmbedding → PositionEncoding → tokens
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        unified_dim: int,
        modalities: List[str],
        pe_configs: Optional[Dict[str, dict]] = None,
    ):
        super().__init__()
        self.projection = ModalityProjection(feature_dims, unified_dim)
        self.modal_emb = ModalityEmbedding(modalities, unified_dim)
        self.pos_enc = PositionEncoding(modalities, unified_dim, pe_configs)
        self.unified_dim = unified_dim

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = self.projection(features)       # [B, 1, D] per modality
        tokens = self.modal_emb(tokens)          # + modality embedding
        tokens = self.pos_enc(tokens)            # + position encoding
        return tokens
```

- [ ] **Step 2: 运行导入验证**

```bash
python -c "
import torch
from models.tokenizer import (
    ModalityProjection, ModalityEmbedding, PositionEncoding, MultiModalTokenizer
)
# 测试完整流程
batch = torch.randn(4, 3, 224, 224)
feats = {'image': torch.randn(4, 512), 'audio': torch.randn(4, 512)}
tok = MultiModalTokenizer(
    feature_dims={'image': 512, 'audio': 512},
    unified_dim=256,
    modalities=['image', 'audio'],
)
out = tok(feats)
print(f'image tokens: {out[\"image\"].shape}')  # [4, 1, 256]
print(f'audio tokens: {out[\"audio\"].shape}')  # [4, 1, 256]
print('Tokenizer test passed')
"
```

Expected: `[4, 1, 256]` for both modalities.

- [ ] **Step 3: Commit**

```bash
git add models/tokenizer.py
git commit -m "feat: add MultiModalTokenizer with projection, modality embedding, and position encoding"
```

---

### Task 4: InteractionBlock — 可堆叠跨模态交互

**Files:**
- Create: `models/interaction.py`

- [ ] **Step 1: 创建 `models/interaction.py`**

```python
"""Cross-Modal Interaction — 可堆叠的 InteractionBlock。

每个 Block = SharedTransform（共享参数，各模态独立应用） + Fusion（跨模态交换）。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .fusion.registry import FusionRegistry, BaseFusion


class TransformerBlock(nn.Module):
    """单个 Transformer Encoder Block，batch_first=True。"""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SharedTransform(nn.Module):
    """共享变换 — 所有模态 token 经过同一个网络层。

    约定：对每个模态独立调用同一组参数。这意味着所有模态在同一个特征空间中
    以相同规则被处理，但不直接交换信息（交换信息由 Fusion 完成）。
    """

    def __init__(self, block_type: str, dim: int, **block_kwargs):
        super().__init__()
        if block_type == "transformer":
            self.block = TransformerBlock(dim=dim, **block_kwargs)
        else:
            raise ValueError(f"未知 block_type: {block_type}")

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: self.block(t) for m, t in tokens.items()}


class InteractionBlock(nn.Module):
    """一个跨模态交互块。

    执行顺序:
        1. SharedTransform: 各模态 token 经过共享网络独立变换
        2. Fusion:         跨模态信息交换

    Config 粒度：每个 block 可独立选择 transform_type 和 fusion_type。
    """

    def __init__(
        self,
        modalities,
        dim: int,
        transform_type: str = "transformer",
        transform_kwargs: Optional[dict] = None,
        fusion_type: str = "none",
        fusion_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.transform = SharedTransform(
            block_type=transform_type,
            dim=dim,
            **(transform_kwargs or {}),
        )
        self.fusion = FusionRegistry.create(
            fusion_type,
            modalities=modalities,
            dim=dim,
            **(fusion_kwargs or {}),
        )

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = self.transform(tokens)
        tokens = self.fusion(tokens)
        return tokens
```

- [ ] **Step 2: 验证 InteractionBlock 可堆叠**

```bash
python -c "
import torch
from models.interaction import InteractionBlock

tokens = {
    'image': torch.randn(2, 1, 256),
    'audio': torch.randn(2, 1, 256),
}

# 堆叠 4 个 block，不同融合策略
blocks = torch.nn.ModuleList([
    InteractionBlock(['image','audio'], dim=256, fusion_type='none'),
    InteractionBlock(['image','audio'], dim=256, fusion_type='gate', fusion_kwargs={'gate_hidden_dim': 128}),
    InteractionBlock(['image','audio'], dim=256, fusion_type='cross_attn', fusion_kwargs={'num_heads': 4}),
    InteractionBlock(['image','audio'], dim=256, fusion_type='token_mix', fusion_kwargs={'num_heads': 4}),
])

for i, blk in enumerate(blocks):
    tokens = blk(tokens)
    print(f'Block {i}: image={tokens[\"image\"].shape}, audio={tokens[\"audio\"].shape}')

print('InteractionBlock stack test passed')
"
```

Expected: 4 个 block 全部通过，shape 保持 `[2, 1, 256]`。

- [ ] **Step 3: Commit**

```bash
git add models/interaction.py
git commit -m "feat: add InteractionBlock with shared transform and pluggable fusion"
```

---

### Task 5: Decision 模块 — 预留接口

**Files:**
- Create: `models/decision.py`

- [ ] **Step 1: 创建 `models/decision.py`**

```python
"""Decision 模块 — 预留接口，支持未来证据/置信度融合。

当前提供:
    - BaseDecision: 抽象基类，定义标准接口
    - IdentityDecision: 默认实现，不做任何处理

未来可继承 BaseDecision 实现:
    - EvidenceDecision: Dirichlet evidence fusion
    - UncertaintyDecision: MC Dropout / Deep Ensemble
    - GraphReasoningDecision: 图推理决策
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class BaseDecision(nn.Module, ABC):
    """决策模块抽象基类。

    forward 返回:
        feature: Tensor [B, D] — 决策后的特征
        aux: Optional[Dict] — 辅助信息（如 uncertainty, evidence, attention weights）
    """

    @abstractmethod
    def forward(
        self, fused_feature: torch.Tensor, tokens: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            fused_feature: Mid Fusion 后的特征 [B, D_fused]
            tokens:        各模态 token 序列（可选，供复杂决策使用）
        Returns:
            (decision_feature [B, D_reason], aux dict or None)
        """
        ...


class IdentityDecision(BaseDecision):
    """默认决策模块 — 透传不做任何处理。"""

    def __init__(self, in_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, fused_feature, tokens=None):
        return self.proj(fused_feature), None


# Decision Registry — 为未来扩展准备
class DecisionRegistry:
    _decisions: Dict[str, type] = {"identity": IdentityDecision}

    @classmethod
    def register(cls, name: str):
        def decorator(decision_cls):
            cls._decisions[name] = decision_cls
            return decision_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDecision:
        if name not in cls._decisions:
            raise KeyError(f"未知决策模块: '{name}'，可用: {list(cls._decisions.keys())}")
        return cls._decisions[name](**kwargs)
```

- [ ] **Step 2: 验证接口**

```bash
python -c "
import torch
from models.decision import IdentityDecision, BaseDecision, DecisionRegistry

d = DecisionRegistry.create('identity', in_dim=512)
out, aux = d(torch.randn(4, 512))
print(f'Output: {out.shape}, Aux: {aux}')  # [4, 512], None
print('Decision module test passed')
"
```

Expected: `[4, 512], None`.

- [ ] **Step 3: Commit**

```bash
git add models/decision.py
git commit -m "feat: add Decision module with reserved interface for evidence/uncertainty fusion"
```

---

### Task 6: MultimodalPipelineV2 — 新流水线模型

**Files:**
- Create: `models/pipeline_v2.py`

- [ ] **Step 1: 创建 `models/pipeline_v2.py`**

```python
"""MultimodalPipelineV2 — 新多模态流水线。

Flow:
    Input → Stem(各模态独立) → Tokenization → InteractionBlocks
        → Per-Modal Pooling → Mid Fusion → Decision → Classifier → logits

Config 驱动，所有组件可插拔。
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from .tokenizer import MultiModalTokenizer
from .interaction import InteractionBlock
from .heads import MultimodalFusion
from .decision import DecisionRegistry, BaseDecision


class MultimodalPipelineV2(nn.Module):
    """新多模态流水线模型。

    Config 示例见 configs/default_multimodal_v2.yaml。
    """

    def __init__(
        self,
        # Stem
        stems: Dict[str, nn.Module],
        # Tokenizer
        tokenizer: MultiModalTokenizer,
        # Interaction
        interaction_blocks: nn.ModuleList,
        # Mid Fusion
        mid_fusion: Optional[MultimodalFusion],
        mid_fusion_output_dim: int,
        # Decision
        decision: BaseDecision,
        decision_output_dim: int,
        # Classifier
        num_classes: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.stems = nn.ModuleDict(stems)
        self.tokenizer = tokenizer
        self.interaction_blocks = interaction_blocks
        self.mid_fusion = mid_fusion
        self.decision = decision

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(decision_output_dim, num_classes),
        )

        self.mid_fusion_output_dim = mid_fusion_output_dim

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
                "logits": Tensor [B, num_classes]
                "aux": Optional[Dict] — 来自 decision 的辅助信息
        """
        # 1. Stem: 各模态独立特征提取
        stem_features = {}
        for m, stem in self.stems.items():
            inputs = self._resolve_inputs(batch, m)
            if inputs is not None:
                if isinstance(inputs, torch.Tensor):
                    stem_features[m] = stem(inputs)
                else:
                    stem_features[m] = stem(**inputs)

        if not stem_features:
            raise ValueError("输入批次中没有找到与已注册 stem 匹配的模态数据")

        # 2. Tokenization: 投影到统一空间
        tokens = self.tokenizer(stem_features)

        # 3. Cross-Modal Interaction
        for block in self.interaction_blocks:
            tokens = block(tokens)

        # 4. Per-Modal Pooling → [B, D] per modality
        pooled = self._pool_tokens(tokens)

        # 5. Mid Fusion
        if len(pooled) == 1:
            fused = list(pooled.values())[0]
        elif self.mid_fusion is not None:
            fused = self.mid_fusion(list(pooled.values()))
        else:
            fused = torch.cat(list(pooled.values()), dim=1)

        # 6. Decision
        decision_out, aux = self.decision(fused, tokens)

        # 7. Classifier
        logits = self.classifier(decision_out)

        return {"logits": logits, "aux": aux}

    def _pool_tokens(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """每个模态 token 序列池化为一个特征向量 [B, D]"""
        return {m: t.mean(dim=1) for m, t in tokens.items()}

    @staticmethod
    def _resolve_inputs(batch: Dict[str, Any], modality: str):
        """兼容新旧 batch 格式"""
        if modality in batch:
            return batch[modality]
        prefix = f"{modality}_"
        prefixed = {
            key[len(prefix):]: value
            for key, value in batch.items()
            if key.startswith(prefix)
        }
        return prefixed if prefixed else None
```

- [ ] **Step 2: 运行端到端测试**

```bash
python -c "
import torch
from models.tokenizer import MultiModalTokenizer
from models.interaction import InteractionBlock
from models.decision import DecisionRegistry
from models.pipeline_v2 import MultimodalPipelineV2

# 模拟 stem
class DummyStem(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feature_dim = dim
    def forward(self, x):
        return torch.randn(x.shape[0], self.feature_dim)

# 构建流水线
stems = {
    'image': DummyStem(512),
    'audio': DummyStem(512),
    'wave': DummyStem(256),
}

tokenizer = MultiModalTokenizer(
    feature_dims={'image': 512, 'audio': 512, 'wave': 256},
    unified_dim=256,
    modalities=['image', 'audio', 'wave'],
)

blocks = torch.nn.ModuleList([
    InteractionBlock(['image','audio','wave'], dim=256, fusion_type='none'),
    InteractionBlock(['image','audio','wave'], dim=256, fusion_type='gate', fusion_kwargs={'gate_hidden_dim': 128}),
    InteractionBlock(['image','audio','wave'], dim=256, fusion_type='cross_attn', fusion_kwargs={'num_heads': 4}),
    InteractionBlock(['image','audio','wave'], dim=256, fusion_type='token_mix', fusion_kwargs={'num_heads': 4}),
])

from models.heads import MultimodalFusion
mid_fusion = MultimodalFusion(
    feature_dims=[256, 256, 256],
    output_dim=256,
    fusion_type='attention',
)

decision = DecisionRegistry.create('identity', in_dim=256, out_dim=256)

model = MultimodalPipelineV2(
    stems=stems,
    tokenizer=tokenizer,
    interaction_blocks=blocks,
    mid_fusion=mid_fusion,
    mid_fusion_output_dim=256,
    decision=decision,
    decision_output_dim=256,
    num_classes=3,
)

batch = {
    'image': torch.randn(4, 3, 224, 224),
    'audio': torch.randn(4, 2, 224, 224),
    'wave': torch.randn(4, 512, 6),
}

out = model(batch)
print(f'logits: {out[\"logits\"].shape}, aux: {out[\"aux\"]}')
# [4, 3], None
print('MultimodalPipelineV2 end-to-end test passed')
"
```

Expected: `[4, 3], None`.

- [ ] **Step 3: Commit**

```bash
git add models/pipeline_v2.py
git commit -m "feat: add MultimodalPipelineV2 with stem→tokenization→interaction→fusion→decision→classifier flow"
```

---

### Task 7: ModelBuilder.build_pipeline_v2() — 构建器方法

**Files:**
- Modify: `models/builder.py`

- [ ] **Step 1: 在 `ModelBuilder` 末尾增加 `build_pipeline_v2` 类方法**

```python
# models/builder.py — 在 class ModelBuilder 末尾增加以下方法

@classmethod
def build_pipeline_v2(cls, config) -> "MultimodalPipelineV2":
    from .pipeline_v2 import MultimodalPipelineV2
    from .tokenizer import MultiModalTokenizer
    from .interaction import InteractionBlock
    from .decision import DecisionRegistry
    from .heads import MultimodalFusion

    pipe_cfg = config.model.unified_pipeline
    modalities = list(config.data.modalities)

    # 1. Stems: 复用现有 build_backbone
    stems = {}
    feature_dims = {}
    for m in modalities:
        if m not in config.model.backbones:
            raise ValueError(f"模态 '{m}' 在 model.backbones 中未定义")
        b_cfg = config.model.backbones[m]
        backbone = cls.build_backbone(
            backbone_type=b_cfg.type,
            feature_dim=b_cfg.feature_dim,
            pretrained=getattr(b_cfg, "pretrained", True),
            dropout_rate=config.model.dropout_rate,
            extra_params=getattr(b_cfg, "extra_params", {}),
        )
        if getattr(b_cfg, "freeze", False):
            for p in backbone.parameters():
                p.requires_grad = False
        stems[m] = backbone
        feature_dims[m] = b_cfg.feature_dim

    # 2. Tokenizer
    token_dim = pipe_cfg.token_dim
    pe_cfgs = getattr(pipe_cfg, "position_encodings", None)
    pe_cfgs = dict(pe_cfgs) if pe_cfgs else None
    tokenizer = MultiModalTokenizer(
        feature_dims=feature_dims,
        unified_dim=token_dim,
        modalities=modalities,
        pe_configs=pe_cfgs,
    )

    # 3. Interaction Blocks
    interaction_blocks = nn.ModuleList()
    for i, blk_cfg in enumerate(pipe_cfg.interaction_blocks):
        block = InteractionBlock(
            modalities=modalities,
            dim=token_dim,
            transform_type=getattr(blk_cfg, "transform_type", "transformer"),
            transform_kwargs=dict(getattr(blk_cfg, "transform_kwargs", {})),
            fusion_type=blk_cfg.fusion_type,
            fusion_kwargs=dict(getattr(blk_cfg, "fusion_kwargs", {})),
        )
        interaction_blocks.append(block)

    # 4. Mid Fusion
    mid_fusion_enabled = getattr(pipe_cfg, "mid_fusion_enabled", True)
    mid_fusion_out_dim = pipe_cfg.mid_fusion_output_dim

    if len(modalities) > 1 and mid_fusion_enabled:
        mid_fusion = cls.build_fusion(
            feature_dims=[token_dim] * len(modalities),
            output_dim=mid_fusion_out_dim,
            fusion_type=pipe_cfg.mid_fusion_type,
            dropout_rate=config.model.dropout_rate,
        )
    else:
        mid_fusion = None
        mid_fusion_out_dim = token_dim * len(modalities) if len(modalities) > 1 else token_dim

    # 5. Decision
    decision_cfg = getattr(pipe_cfg, "decision", None)
    if decision_cfg:
        decision = DecisionRegistry.create(
            name=decision_cfg.type,
            in_dim=mid_fusion_out_dim,
            **dict(getattr(decision_cfg, "extra_params", {})),
        )
        # 获取 decision 输出维度
        try:
            test_in = torch.randn(1, mid_fusion_out_dim)
            test_out, _ = decision(test_in)
            decision_out_dim = test_out.shape[-1]
        except Exception:
            decision_out_dim = mid_fusion_out_dim
    else:
        decision = DecisionRegistry.create("identity", in_dim=mid_fusion_out_dim)
        decision_out_dim = mid_fusion_out_dim

    return MultimodalPipelineV2(
        stems=stems,
        tokenizer=tokenizer,
        interaction_blocks=interaction_blocks,
        mid_fusion=mid_fusion,
        mid_fusion_output_dim=mid_fusion_out_dim,
        decision=decision,
        decision_output_dim=decision_out_dim,
        num_classes=config.classes.num_classes,
        dropout_rate=config.model.dropout_rate,
    )
```

- [ ] **Step 2: 在 `ModelBuilder.build_model` 入口加路由**

在 `build_model()` 方法开头加入路由逻辑：

```python
# models/builder.py — 在 build_model() 方法第一行加入

@classmethod
def build_model(cls, config) -> nn.Module:
    # 新流水线路由
    if hasattr(config.model, "unified_pipeline") and config.model.unified_pipeline is not None:
        return cls.build_pipeline_v2(config)
    
    # ... 原有逻辑保持不变 ...
```

- [ ] **Step 3: 验证导入和路由**

```bash
python -c "
from utils.config import load_config

# 用旧 config 验证旧路径不受影响
cfg = load_config('configs/fish_feeding_unireplknet.yaml')
from models.builder import ModelBuilder
print('Old path still works - import OK')

# 验证 build_pipeline_v2 方法存在
assert hasattr(ModelBuilder, 'build_pipeline_v2'), 'build_pipeline_v2 not found'
print('build_pipeline_v2 method exists')
"
```

Expected: 旧路径导入无报错，新方法存在。

- [ ] **Step 4: Commit**

```bash
git add models/builder.py
git commit -m "feat: add ModelBuilder.build_pipeline_v2() with config-driven pipeline construction"
```

---

### Task 8: Config 数据类扩展

**Files:**
- Modify: `utils/config.py`

- [ ] **Step 1: 新增 dataclass**

```python
# utils/config.py — 在其他 dataclass 之后、Config 之前添加

@dataclass
class InteractionBlockConfig:
    """单个 InteractionBlock 配置"""
    transform_type: str = "transformer"
    transform_kwargs: Dict[str, Any] = field(default_factory=dict)
    fusion_type: str = "none"
    fusion_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionConfig:
    """Decision 模块配置"""
    type: str = "identity"
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PEResolutionConfig:
    """Position Encoding 按模态配置"""
    enabled: bool = True
    max_len: int = 256


@dataclass
class UnifiedPipelineConfig:
    """统一多模态流水线配置（新）"""
    token_dim: int = 256
    interaction_blocks: List[InteractionBlockConfig] = field(default_factory=list)
    position_encodings: Dict[str, PEResolutionConfig] = field(default_factory=dict)
    mid_fusion_type: str = "attention"
    mid_fusion_output_dim: int = 256
    mid_fusion_enabled: bool = True
    decision: DecisionConfig = field(default_factory=DecisionConfig)
```

- [ ] **Step 2: 在 `ModelConfig` 中加入 `unified_pipeline` 字段**

```python
# utils/config.py — 在 ModelConfig dataclass 末尾增加

    # 新统一流水线（为 None 时走旧路径）
    unified_pipeline: Optional[UnifiedPipelineConfig] = None
```

- [ ] **Step 3: 在 `validate_config` 中增加校验**

```python
# utils/config.py — 在 validate_config 函数末尾 (system 校验之前) 增加

    # unified pipeline 校验
    if cfg.model.unified_pipeline is not None:
        pipe = cfg.model.unified_pipeline
        _require_positive_int("unified_pipeline.token_dim", int(pipe.token_dim))
        _require_positive_int("unified_pipeline.mid_fusion_output_dim", int(pipe.mid_fusion_output_dim))

        allowed_fusions_v2 = {"concat", "add", "attention", "none"}
        if pipe.mid_fusion_type not in allowed_fusions_v2:
            raise ValueError(
                f"unified_pipeline.mid_fusion_type 必须是 {sorted(allowed_fusions_v2)} 之一"
            )

        for i, blk in enumerate(pipe.interaction_blocks):
            if blk.fusion_type not in FusionRegistry.list_all():
                raise ValueError(
                    f"interaction_blocks[{i}].fusion_type '{blk.fusion_type}' 无效，"
                    f"可用: {FusionRegistry.list_all()}"
                )
```

注意：`FusionRegistry` 在此处需要延迟导入以避免循环引用。把 import 放在校验函数内部：

```python
    # unified pipeline 校验
    if cfg.model.unified_pipeline is not None:
        from models.fusion.registry import FusionRegistry as _FR
        ...
        if blk.fusion_type not in _FR.list_all():
```

- [ ] **Step 4: 验证新旧 config 加载**

```bash
python -c "
from utils.config import load_config

# 旧 config 不设 unified_pipeline，走旧路径
cfg = load_config('configs/fish_feeding_unireplknet.yaml')
assert cfg.model.unified_pipeline is None, 'old config should have unified_pipeline=None'
print('Old config loads OK')

# 测试新 config 字段存在
from utils.config import UnifiedPipelineConfig, InteractionBlockConfig, DecisionConfig
print('New config dataclasses import OK')
"
```

Expected: 旧配置正常加载，新 dataclass 可导入。

- [ ] **Step 5: Commit**

```bash
git add utils/config.py
git commit -m "feat: add UnifiedPipelineConfig dataclass and validation for new pipeline"
```

---

### Task 9: 示例配置 + models/__init__.py 更新

**Files:**
- Create: `configs/default_multimodal_v2.yaml`
- Modify: `models/__init__.py`

- [ ] **Step 1: 创建示例配置**

```yaml
# configs/default_multimodal_v2.yaml
# 统一多模态流水线示例配置

data:
  batch_size: 16
  num_workers: 4
  pin_memory: true

  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"

  modalities:
    - image
    - audio
    - wave

  loaders:
    image:
      type: image_loader
      extra_params: {}
    audio:
      type: audio_loader_stereo
      extra_params:
        sample_rate: 16000
        max_length: 160000
        n_mels: 224
        time_steps: 224
    wave:
      type: wave_loader
      extra_params:
        max_length: 512
        num_features: 6
        normalize: true

  image_size: 224

classes:
  num_classes: 3
  class_names: [None, Strong, Weak]

model:
  # Stem backbone 配置（与旧路径一致，复用 modelzoo）
  backbones:
    image:
      type: unireplknet_image
      feature_dim: 512
      pretrained: false
      freeze: false
      extra_params:
        variant: p
        drop_path_rate: 0.1
    audio:
      type: unireplknet_audio
      feature_dim: 512
      pretrained: false
      freeze: false
      extra_params:
        variant: p
        audio_channels: 2
        drop_path_rate: 0.1
    wave:
      type: unireplknet_wave
      feature_dim: 512
      pretrained: false
      freeze: false
      extra_params:
        variant: p
        seq_len: 512
        in_channels: 6
        image_size: 224
        drop_path_rate: 0.1

  # === 新流水线特有配置 ===
  unified_pipeline:
    token_dim: 256

    # Token 的位置编码（按模态）
    position_encodings:
      image: { enabled: true, max_len: 256 }
      audio: { enabled: true, max_len: 256 }
      wave:  { enabled: true, max_len: 512 }

    # 跨模态交互块（可堆叠，每块独立选融合策略）
    interaction_blocks:
      - transform_type: transformer
        transform_kwargs: { num_heads: 8, mlp_ratio: 4.0, dropout: 0.1 }
        fusion_type: none

      - transform_type: transformer
        transform_kwargs: { num_heads: 8, mlp_ratio: 4.0, dropout: 0.1 }
        fusion_type: gate
        fusion_kwargs: { gate_hidden_dim: 128, dropout: 0.1 }

      - transform_type: transformer
        transform_kwargs: { num_heads: 8, mlp_ratio: 4.0, dropout: 0.1 }
        fusion_type: cross_attn
        fusion_kwargs: { num_heads: 4, dropout: 0.1 }

      - transform_type: transformer
        transform_kwargs: { num_heads: 8, mlp_ratio: 4.0, dropout: 0.1 }
        fusion_type: token_mix
        fusion_kwargs: { num_heads: 8, dropout: 0.1 }

    # 中期融合
    mid_fusion_type: attention
    mid_fusion_output_dim: 256

    # 决策（预留，"identity" 是透传）
    decision:
      type: identity
      extra_params: {}

  dropout_rate: 0.35

train:
  epochs: 100
  learning_rate: 0.00005
  weight_decay: 0.001
  lr_scheduler: cosine
  warmup_epochs: 5
  optimizer: adamw
  label_smoothing: 0.1
  early_stop: { enabled: true, patience: 8, min_delta: 0.001, monitor: val_loss, mode: min }
  val_interval: 1

eval:
  metrics: [accuracy, precision, recall, f1]
  save_predictions: true
  confusion_matrix: true

system:
  seed: 42
  gpu_ids: [0]
  distributed: false
  fp16: false
  log_interval: 20
  save_interval: 10
  output_dir: output/multimodal_v2
  tensorboard_enabled: true
  experiment_name: multimodal_v2
```

- [ ] **Step 2: 更新 `models/__init__.py`**

```python
# models/__init__.py — 增加新模块导出
from .tokenizer import MultiModalTokenizer
from .fusion import FusionRegistry, BaseFusion
from .interaction import InteractionBlock
from .decision import DecisionRegistry, BaseDecision, IdentityDecision
from .pipeline_v2 import MultimodalPipelineV2
```

- [ ] **Step 3: 加载新配置并构建模型**

```bash
python -c "
from utils.config import load_config
from models.builder import ModelBuilder

cfg = load_config('configs/default_multimodal_v2.yaml')
print(f'Config loaded: {cfg.model.unified_pipeline.token_dim} token_dim')
print(f'Interaction blocks: {len(cfg.model.unified_pipeline.interaction_blocks)}')

model = ModelBuilder.build_model(cfg)
print(f'Model type: {type(model).__name__}')
print(f'Stems: {list(model.stems.keys())}')
print(f'Interaction blocks: {len(model.interaction_blocks)}')

# 前向测试
import torch
batch = {
    'image': torch.randn(2, 3, 224, 224),
    'audio': torch.randn(2, 2, 224, 224),
    'wave': torch.randn(2, 512, 6),
}
out = model(batch)
print(f'Output: logits={out[\"logits\"].shape}, aux={out[\"aux\"]}')
print('Full pipeline build + forward test passed')
"
```

Expected: 模型构建成功，forward 输出 `[2, 3]`，`aux=None`。

- [ ] **Step 4: Commit**

```bash
git add configs/default_multimodal_v2.yaml models/__init__.py
git commit -m "feat: add example config for unified pipeline v2 and update model exports"
```

---

### Task 10: VideoMAE 视频 Backbone

**Files:**
- Modify: `models/modelzoo/video_models.py`

video_models.py 已有 7 个视频 backbone，新增 VideoMAE。

- [ ] **Step 1: 在 video_models.py 末尾添加 VideoMAE**

```python
# models/modelzoo/video_models.py — 在文件末尾添加


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
```

- [ ] **Step 2: 验证注册**

```bash
python -c "
from models.registry import ModelZoo
ModelZoo._auto_load_models()
v = ModelZoo.create('videomae', feature_dim=512, img_size=224, num_frames=8, depth=4, embed_dim=256, num_heads=4)
import torch
out = v(torch.randn(2, 3, 8, 224, 224))
print(f'VideoMAE output: {out.shape}')  # [2, 512]
print('VideoMAE test passed')
"
```

Expected: `[2, 512]`。

- [ ] **Step 3: Commit**

```bash
git add models/modelzoo/video_models.py
git commit -m "feat: add VideoMAE video backbone"
```

---

### Task 11: 端到端训练验证

**Files:**
- Create: `tests/test_pipeline_v2.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_pipeline_v2.py
"""MultimodalPipelineV2 端到端测试"""
import torch
import pytest

from utils.config import load_config
from models.builder import ModelBuilder
from models.pipeline_v2 import MultimodalPipelineV2
from models.tokenizer import MultiModalTokenizer
from models.interaction import InteractionBlock
from models.decision import DecisionRegistry
from models.heads import MultimodalFusion


class TestMultimodalPipelineV2:
    def test_minimal_pipeline_build(self):
        """最小流水线：1模态，无fusion，无interaction"""
        stems = {"image": _DummyStem(512)}
        tokenizer = MultiModalTokenizer(
            feature_dims={"image": 512}, unified_dim=256,
            modalities=["image"],
        )
        model = MultimodalPipelineV2(
            stems=stems, tokenizer=tokenizer,
            interaction_blocks=torch.nn.ModuleList([]),
            mid_fusion=None, mid_fusion_output_dim=256,
            decision=DecisionRegistry.create("identity", in_dim=256),
            decision_output_dim=256, num_classes=3,
        )
        batch = {"image": torch.randn(2, 3, 224, 224)}
        out = model(batch)
        assert out["logits"].shape == (2, 3)
        assert out["aux"] is None

    def test_three_modality_pipeline(self):
        """3模态 + 4个interaction blocks + attention fusion"""
        stems = {
            "image": _DummyStem(512),
            "audio": _DummyStem(512),
            "wave": _DummyStem(256),
        }
        tokenizer = MultiModalTokenizer(
            feature_dims={"image": 512, "audio": 512, "wave": 256},
            unified_dim=256, modalities=["image", "audio", "wave"],
        )
        blocks = torch.nn.ModuleList([
            InteractionBlock(["image","audio","wave"], dim=256, fusion_type="none"),
            InteractionBlock(["image","audio","wave"], dim=256, fusion_type="gate",
                             fusion_kwargs={"gate_hidden_dim": 128}),
            InteractionBlock(["image","audio","wave"], dim=256, fusion_type="cross_attn",
                             fusion_kwargs={"num_heads": 4}),
            InteractionBlock(["image","audio","wave"], dim=256, fusion_type="token_mix",
                             fusion_kwargs={"num_heads": 4}),
        ])
        mid_fusion = MultimodalFusion(
            feature_dims=[256, 256, 256], output_dim=256,
            fusion_type="attention",
        )
        model = MultimodalPipelineV2(
            stems=stems, tokenizer=tokenizer, interaction_blocks=blocks,
            mid_fusion=mid_fusion, mid_fusion_output_dim=256,
            decision=DecisionRegistry.create("identity", in_dim=256),
            decision_output_dim=256, num_classes=5,
        )
        batch = {
            "image": torch.randn(4, 3, 224, 224),
            "audio": torch.randn(4, 2, 224, 224),
            "wave": torch.randn(4, 512, 6),
        }
        out = model(batch)
        assert out["logits"].shape == (4, 5)

    def test_forward_backward(self):
        """验证梯度可回传"""
        stems = {
            "image": torch.nn.Linear(512, 512),
            "audio": torch.nn.Linear(512, 512),
        }
        tokenizer = MultiModalTokenizer(
            feature_dims={"image": 512, "audio": 512},
            unified_dim=256, modalities=["image", "audio"],
        )
        blocks = torch.nn.ModuleList([
            InteractionBlock(["image","audio"], dim=256, fusion_type="cross_attn",
                             fusion_kwargs={"num_heads": 4}),
        ])
        model = MultimodalPipelineV2(
            stems=stems, tokenizer=tokenizer, interaction_blocks=blocks,
            mid_fusion=None, mid_fusion_output_dim=512,
            decision=DecisionRegistry.create("identity", in_dim=512),
            decision_output_dim=512, num_classes=3,
        )
        batch = {
            "image": torch.randn(2, 512),
            "audio": torch.randn(2, 512),
        }
        out = model(batch)
        loss = out["logits"].sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} has no grad"

    def test_old_config_still_works(self):
        """旧配置仍走旧路径"""
        cfg = load_config("configs/fish_feeding_unireplknet.yaml")
        assert cfg.model.unified_pipeline is None
        model = ModelBuilder.build_model(cfg)
        from models.builder import MultimodalClassifier
        assert isinstance(model, MultimodalClassifier)


class _DummyStem(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feature_dim = dim
    def forward(self, x):
        return torch.randn(x.shape[0], self.feature_dim)
```

- [ ] **Step 2: 运行测试**

```bash
pytest tests/test_pipeline_v2.py -v
```

Expected: 4 tests PASS。

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_v2.py
git commit -m "test: add end-to-end tests for MultimodalPipelineV2"
```

---

## Self-Review

### 1. Spec Coverage

| 需求 | 覆盖 | 任务 |
|---|---|---|
| 各模态 Stem（复用 modelzoo） | Yes | Task 7 — `build_pipeline_v2` 调用 `build_backbone` 构建 stem |
| Tokenization 统一空间 | Yes | Task 3 & 6 — MultiModalTokenizer 投影 + 模态嵌入 + 位置编码 |
| 可配置堆叠 InteractionBlock | Yes | Task 4 & 7 — InteractionBlock 列表，每个独立配 transform/fusion |
| 可插拔 Fusion 策略 | Yes | Task 2 — FusionRegistry + 4 种内置策略（none/gate/cross_attn/token_mix） |
| 保留模态边界 | Yes | 全程 `Dict[str, Tensor[B,N,D]]` |
| Decision 预留接口 | Yes | Task 5 — BaseDecision 抽象类 + IdentityDecision 默认实现 |
| 视频 Backbone | Yes | Task 10 — VideoMAE 新增（其余 7 个已有） |
| 旧路径兼容 | Yes | Task 7 — `use_unified_pipeline` 为 None 时走旧路径 |

### 2. Placeholder Scan

无 TBD/TODO，无 "add appropriate error handling"。所有代码步骤包含完整实现。

### 3. Type Consistency

- `modalities: List[str]` — 全链路一致
- `dim: int` — 统一 token 维度 D
- `tokens: Dict[str, Tensor[B,N,D]]` — InteractionBlock 输入输出一致
- `feature_dims: Dict[str, int]` — Tokenization 入口
- `BaseDecision.forward → Tuple[Tensor, Optional[Dict]]` — Decision 接口一致
