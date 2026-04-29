# Modality Balance P0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 实现模态Dropout + 辅助分类头，解决image模态梯度压制问题，使弱模态得到充分训练

**Architecture:** 修改pipeline_v2.py添加模态dropout和辅助分类头，修改训练脚本支持联合loss。4GPU并行训练3频率

---

### Task 1: 修改 MultimodalPipelineV2 添加模态Dropout和辅助分类头

**Files:**
- Modify: `models/pipeline_v2.py`

实现：
1. 构造函数添加 `modality_dropout: float = 0.0` 参数
2. 添加辅助分类头 `aux_classifiers: nn.ModuleDict`（每模态独立）
3. forward时：以概率p随机drop模态token；计算各模态单模态logits

### Task 2: 创建模态平衡训练脚本

**Files:**
- Create: `scripts/train_balanced.py`

联合loss训练：
- Loss = fusion_loss + λ * mean(aux_losses)
- 支持`--modality-dropout`和`--aux-lambda`参数
- 4GPU并行（每GPU一个频率）

### Task 3: 运行训练 + 评测

3个频率并行训练，然后评测对比
