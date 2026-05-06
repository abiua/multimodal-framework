# SACF + Adaptive Evidence Reasoning (AER) — 设计文档

## 背景

PMIN (arxiv 2506.14170v3) 提出 Adaptive Evidence Reasoning (AER) 模块，用于多模态决策层融合。它把每个分支的分类预测视为"证据"，用可学习的权重 w_m 和可靠性 r_m 做 Dempster-Shafer 证据推理，显式建模模态间的冲突和不确定性。

SACF 已有 3 个 classifier（phys / image / final），目前用 KL 散度共识损失对齐。将 AER 替换 KL 共识，做决策层融合。

## 改动方案

**方案 A：AER 替换 KL 共识**

```
当前:                              改为:
phys_logits  → CE (alpha)          phys_logits   → CE (alpha)
image_logits → CE (beta)           image_logits  → CE (beta)
final_logits → CE (main) + KL      phys_logits   ┐
                                    image_logits  ├→ AER → CE (main)
                                    final_logits  ┘
                                    KL 共识 → 删除
```

损失函数:
```
loss = CE(aer_output, target)
     + alpha * CE(phys_logits, target)
     + beta * CE(image_logits, target)
```

去掉 `gamma`（原 KL 共识权重）和 `loss_cons`。

## AER 模块详细设计

### 文件: `models/fusion/aer.py`

```python
class AdaptiveEvidenceReasoning(nn.Module):
    """
    M 条证据 (M 个分支 logits) → 联合置信度 → log-probabilities

    可学习参数:
      - evidence_weight (M,): w_m, 初始化为 1.0
      - evidence_reliability (M,): r_m, 初始化为 0.5

    前向:
      1. softmax → 置信度分布
      2. c_{rw,m} = 1 / (1 + w_m - r_m)  归一化因子
      3. 加权置信度 + 全局无知项
      4. 公式(14) 融合 → 联合置信度
      5. log-softmax → CE loss 用
    """
```

### 关键公式

```
c_{rw,m} = 1 / (1 + w_m - r_m)

p̃_{θ,m} = c_{rw,m} * w_m * p_{θ,m}  (加权置信度)
p̃_{P(Θ),m} = c_{rw,m} * (1 - r_m)   (全局无知项)

融合 (公式14):
P_{θ_n} = L * [∏(c_{rw,m}*(1 - r_m) + p̃_{θ_n,m}) - ∏(c_{rw,m}*(1 - r_m))]
          / [1 - L * ∏(c_{rw,m}*(1 - r_m))]

L = [∑_n ∏(c_{rw,m}*(1 - r_m) + p̃_{θ_n,m}) - (N-1) * ∏(c_{rw,m}*(1 - r_m))]^(-1)

输出: log_softmax(P_θ)
```

### 约束

- `w_m` 保持 > 0 (用 softplus)
- `r_m` 保持 ∈ (0, 1) (用 sigmoid)
- 数值稳定: log-space 运算或 eps 保护

## 涉及文件和改动

| 文件 | 改动 |
|------|------|
| `models/fusion/aer.py` | 新建 AER 模块 |
| `models/pipeline_sacf.py` | forward 返回 3 个 logits (不通过 AER) |
| `scripts/train_sacf.py` | 删除 KL loss，加入 AER 融合 + CE |
| `configs/sacf_fish_feeding.yaml` | 删除 gamma，添加 aer 子配置 (可选) |

## 超参数

- `alpha: 0.5` — phys_classifier 辅助 CE 权重 (保持)
- `beta: 0.5` — image_classifier 辅助 CE 权重 (保持)
- `gamma` — 删除 (原 KL 共识权重)
- AER 内部: w_m 初始 1.0, r_m 初始 0.5 (可学习)

## 测试和验证

- [ ] AER 模块单元测试: 输入 M×N logits，输出概率分布，概率和 ≈ 1
- [ ] 训练收敛: 对比原 SACF 的 loss 曲线
- [ ] 评估对比: accuracy / F1 与原 SACF 对比
- [ ] forward 兼容: pipeline 不改接口，返回字典仍包含 3 个 logits
