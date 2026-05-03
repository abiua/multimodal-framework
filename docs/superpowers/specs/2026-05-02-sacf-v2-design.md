# SACF v2 — Stage-Aware Consensus Fusion

**Status:** Draft  
**Date:** 2026-05-02  
**Author:** Abiu + Claude Code

## 1. 背景与动机

### 1.1 问题

当前 V3 Pipeline（Physics-First Asymmetric Fusion）存在三个核心缺陷：

1. **模态单边依赖**: Image 贡献 86% 梯度，Wave/Audio 各贡献 5-9%。移除 Image 后准确率从 0.87 暴跌至 0.40。物理模态成为搭便车者。
2. **Audio 时序信息丢失**: Audio 声谱图被压成单个全局 token [B,1,D]，Wave↔Audio cross-attention 退化为 Wave 看一个全局音频摘要，无法实现真正的时序对齐。
3. **Evidence Gate 结构失效**: 在两个数据集（5Hz、Fish Feeding）、两种视觉模态（Video、Image）下梯度均≈0，确认为架构缺陷而非数据问题。
4. **Image 定位伪对齐风险**: 单帧 Image 试图生成 [B,T] 时间注意力权重，语义上不可靠——Image 快照无法推断细粒度时间重要性。

### 1.2 约束

- 当前模型是**多模态教师模型**（Image+Wave+Audio）
- 后续将训练**学生模型**（Wave+Audio only），去掉 Image 模态
- 教师模型的目标：让物理模态（Wave+Audio）从 Image 中学习共识特征，而非依赖 Image
- 数据集：Fish Feeding Local，4961 train / 1416 val / 712 test，3 类（None/Strong/Weak）
- 训练时使用4gpu，最快速度、最稳设置训练

### 1.3 单模态能力基线

| 配置 | 准确率 | 含义 |
|------|--------|------|
| Full（三模态） | 0.869 | 教师参考上限 |
| Image+Audio（no_wave） | 0.864 | Image≈0.86，Audio 边际增益 |
| Image+Wave（no_audio） | 0.867 | Image≈0.86，Wave 边际增益 |
| Wave+Audio 融合路径 | 0.403 | 融合层依赖 Image，去掉即崩塌 |
| Wave+Audio 物理分类器 | 0.598 | 物理编码器有独立能力 |

**核心矛盾**：物理编码器独立可达 0.60，但 V3 的融合层（AsymmetricInteraction+Gate）学的是 Image→Physical 单向依赖。去掉 Image 后，融合层输出比物理编码器直接分类还差。

## 2. 架构设计

### 2.1 总体流程

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Temporal Consensus (Wave + Audio)                  │
│                                                             │
│ Wave [B,512,6]  → WaveEncoder  → [B,T_w,D]                 │
│ Audio [B,2,F,T] → AudioTemporalCNN → [B,T_a,D]             │
│                         ↓                                   │
│   Bidirectional Cross-Attention (Wave↔Audio)               │
│                         ↓                                   │
│   Physical Tokens [B,T,D] → Pooling → phys_pooled [B,D]    │
│                         ↓                                   │
│   phys_classifier → phys_logits  (独立损失 L_phys)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Image FiLM Modulation + Residual                   │
│                                                             │
│ Image [B,3,224,224] → ResNet50 → z_img [B,D_img]           │
│                         ↓                                   │
│   ┌─ FiLM Gate: scale/shift = MLP(z_img)                   │
│   │      ↓                                                  │
│   │  phys_pooled' = scale ⊙ phys_pooled + shift            │
│   │                                                         │
│   └─ SideNet: r_img = SmallMLP(z_img) [B,d], d < D         │
│          ↓                                                  │
│      gate = Sigmoid(MLP_g(z_img))                          │
│          ↓                                                  │
│   f_final = phys_pooled' + gate ⊙ r_img                    │
│          ↓                                                  │
│   final_classifier → logits (主损失 L_main)                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Consensus Regularization                           │
│                                                             │
│   L_main: CrossEntropy(final_logits, labels)               │
│   L_phys: CrossEntropy(phys_logits, labels)                 │
│   L_image: CrossEntropy(image_logits, labels)               │
│   L_cons: KL(stopgrad(p_final) || p_phys)                  │
│          + KL(stopgrad(p_final) || p_img)                   │
│                                                             │
│   L_total = L_main + α·L_phys + β·L_image + γ·L_cons       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 组件说明

#### 2.2.1 WaveEncoder

复用现有 `WaveEncoder`（scripts/train_v3_fish.py）：
- Conv1D stem (6→64) → 多层膨胀 TCN → 1×1 conv 投影
- 输出 [B, T_w, D]，T_w 取决于输入长度和 dilation

#### 2.2.2 AudioTemporalEncoder（新）

替代现有 `audiocnn` 的全局池化输出：

```
Mel Spectrogram [B,2,F,T] → 卷积层组 → [B,C,H',T']
    → AdaptiveAvgPool(H维度) → [B,C,T']
    → 1×1 Conv 投影 → [B,C',T_a]
    → Transpose → [B,T_a,D]
```

- 在频率维度池化，保留时间维度
- T_a 由 spectrogram 时间步数决定（如 224 → 经过 stride 后 → 14~28 个 token）
- 每个 token 代表一个时间段内的频谱特征

#### 2.2.3 Bidirectional Cross-Attention（新）

```python
def forward(wave_tokens, audio_tokens):
    # Wave queries Audio
    w2a = CrossAttn(q=wave_tokens, kv=audio_tokens)  # [B,T_w,D]
    # Audio queries Wave
    a2w = CrossAttn(q=audio_tokens, kv=wave_tokens)  # [B,T_a,D]
    # Concatenate along sequence dim
    physical = Concat([w2a, a2w], dim=1)  # [B,T_w+T_a,D]
    # Shared self-attention fusion
    physical = TransformerBlocks(physical)  # [B,T_w+T_a,D]
    return physical
```

- 双向 cross-attn 确保两个模态**互相感知对方的时序结构**
- 共享 transformer 层将两个视角融合为统一表示
- 替代 V3 的 `PhysicalDynamicsEncoder`（原 IMU→cross-attn→Audio→transformer 单向流）

#### 2.2.4 FiLM Modulation

```python
z_img = image_backbone(image)  # [B, D_img]
scale = Linear(D_img, D)(z_img)
shift = Linear(D_img, D)(z_img)
phys_modulated = scale * phys_pooled + shift
```

- Image 通过 channel-wise scale/shift 调节物理特征
- 语义："强调/抑制物理特征的哪些维度"
- 不涉及时间定位——Image 只做全局条件调制

#### 2.2.5 SideNet + Gated Residual

```python
r_img = SmallMLP(z_img)    # [B, d]，d = D // 4
gate = Sigmoid(MLP_g(z_img))  # [B, d]
f_final = phys_modulated + gate * r_img
```

- d < D 限制残差容量，防止 Image shortcut
- Sigmoid gate 让模型自主决定何时依赖 Image 残差
- 训练时对 r_img 做 30% stochastic dropout

#### 2.2.6 共识损失

```python
L_cons = KL(stopgrad(p_final) || p_phys) + KL(stopgrad(p_final) || p_img)
```

- Final 多模态输出作为 teacher，单模态分支向它学习
- stopgrad 阻断反向梯度，本质是"final 教 phys/image 达成共识"
- 不会出现噪声模态拖偏 final 的情况

#### 2.2.7 Modal Dropout + Null Token

训练时 30% 概率随机 mask 1 个或 2 个模态。

| 丢弃模态 | 退化路径 |
|---------|---------|
| Image | FiLM 恒等（scale=1, shift=0），r_img=0，f=phys_pooled |
| Audio | Audio tokens 替换为 null_token，cross-attn 退化为 Wave 自注意 |
| Wave | Wave tokens 替换为 null_token，cross-attn 退化为 Audio 自注意 |
| Image+Audio | 只剩 Wave token，跳过 cross-attn，直接 pooling |
| Wave+Audio | Image 残差关闭，f=phys_pooled（注意此时 phys_pooled 来自 null tokens） |

每个模态持有可学习参数 `null_token`，mask 时 broadcast 到对应形状。同时传递 `modality_mask` 给 cross-attn 层以正确计算注意力。

### 2.3 与 V3 的逐项对比

| | V3 | SACF v2 |
|------|-----|---------|
| Audio 表示 | 单 token [B,1,D] | 多 token [B,T_a,D] |
| Wave/Audio 交互 | 单向 (IMU→Physical) | 双向 Cross-Attn |
| Image 作用 | 硬交互 + EvidenceGate | FiLM 调制 + 门控残差 |
| 融合公式 | `concat[phys, vis*gate]` | `phys' + gate·residual` |
| 模态独立性 | 无 | 三路独立分类器 |
| 共识机制 | 无 | 单向 KL(final→single) |
| 模态丢弃 | 无 | 30% 训练时随机 mask |
| Evidence Gate | 已确认死亡 | 移除 |

### 2.4 参数量估算

| 组件 | 参数量 |
|------|--------|
| WaveEncoder | ~0.5M |
| AudioTemporalEncoder | ~2M |
| ResNet50（Image） | ~25M |
| Cross-Attention Blocks | ~2M |
| FiLM + SideNet + Classifiers | ~0.5M |
| **总计** | **~30M**（与 V3 持平） |

## 3. 训练计划

### 3.1 阶段 1: 基础训练（教师模型）

| 配置 | 值 |
|------|-----|
| 优化器 | AdamW, lr=1e-4 |
| 调度器 | CosineAnnealing, T_max=80 |
| 损失权重 | α=0.5, β=0.5, γ=0.3 |
| Batch Size | 16/GPU × 4 GPU = 64 effective |
| Epochs | 80（early stop patience=15） |
| FP16 | 混合精度 |
| 模态丢弃率 | 0.3 |

验证指标：
- Full 三模态 test acc（target ≥ 0.87）
- Wave+Audio only test acc（target ≥ 0.70，当前 0.40）
- 共识损失收敛趋势
- 各模态独立分类器 acc

### 3.2 阶段 2: 消融验证

训练完成后执行：
1. 单模态/组合模态测试（Image-only, Wave-only, Audio-only, Wave+Audio）
2. 去除 Image 的准确率下降（target < 15% vs V3 的 47%）
3. 梯度贡献分析（三模态贡献是否趋于均衡）
4. 对比 V3 的 physics_only 准确率提升

### 3.3 阶段 3: 蒸馏探索（后续）

学生模型（Wave+Audio only）架构为 SACF 去掉 Stage 2 的 Image 路径：
- 保留 Stage 1 完整的 Temporal Consensus（Wave↔Audio 双向 cross-attn）
- 移除 FiLM、SideNet、image_classifier
- phys_classifier 直接作为学生输出
- 蒸馏损失使用 Stage 1 教师模型的 final logits + phys features

## 4. 需要实现的文件

### 4.1 新增

| 文件 | 内容 |
|------|------|
| `models/pipeline_sacf.py` | SACF Pipeline v2 完整实现 |
| `models/fusion/temporal_consensus.py` | Bidirectional Cross-Attn 时序融合模块 |
| `models/fusion/film_gate.py` | FiLM 调制 + Gated Residual |
| `models/modelzoo/audio_temporal.py` | AudioTemporalEncoder（保留时间维度的音频编码器） |
| `configs/sacf_fish_feeding.yaml` | SACF 训练配置 |
| `scripts/train_sacf.py` | SACF 训练脚本 |

### 4.2 修改

| 文件 | 变更 |
|------|------|
| `models/fusion/__init__.py` | 导出新模块 |

### 4.3 非目标

- 不修改 V3 pipeline（保留对比基准）
- 不修改现有 DataFactory / loaders
- 不修改蒸馏框架（阶段 3 再用）

## 5. 成功标准

1. Full 三模态 test acc ≥ 0.87（不低于 V3）
2. Wave+Audio（无 Image）test acc ≥ 0.70（当前 V3 为 0.40，提升 75%）
3. 三模态梯度贡献分布趋于均衡（Image < 60%，Wave/Audio 各 > 15%）
4. 共识损失正常收敛，无模态崩塌
5. Modal dropout 下所有组合可用
