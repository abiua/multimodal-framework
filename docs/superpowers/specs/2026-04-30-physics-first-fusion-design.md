# Physics-First Asymmetric Fusion: 鱼类摄食强度多模态分类

> 设计日期: 2026-04-30. 核心思想：Audio + IMU 是物理信号的直接测量，Video 是物理事件的视觉投影。架构应反映这种因果关系的不对称性。

## 1. 问题定义

- **任务**: 4 类鱼类摄食强度分类（无/弱/中/强）
- **模态**: Video (mp4) + Audio (wav, 16kHz stereo) + IMU (csv, 20Hz, 9 轴传感器)
- **核心约束**: 未来部署一个只有 Audio+IMU 的学生模型（无 Video）
- **数据**: ~603 样本，分布在 40 个事件 session 中

## 2. 整体架构

```
Input: IMU[B,T_imu,9]  Audio[B,2,T_aud]  Video[B,T_v,H,W,C]
         ↓                   ↓                   ↓
    ChannelSplitter    MelSpectrogram        VideoMAE
    → accel[B,T,3]     → [B,2,M,T_a]         → [B,T_v,D]
    → gyro [B,T,3]          ↓
    → angle[B,T,3]     AudioCNN
         ↓              → [B,T_a,D]
    MultiChannelTCN          ↓
    → [B,T_imu,D]     TemporalAlign ───→
         ↓                                ↓
    ┌─────────────────────────────────────┐
    │  Physical Dynamics Encoder          │
    │  内部: IMU tokens + Audio tokens    │
    │  → Cross-Attention / Transformer    │
    │  → Physical Tokens [B, T_p, D]      │
    └─────────────────┬───────────────────┘
                      ↓
            Physical Tokens [B, T_p, D]
                      ↑ (单向 query)
            Visual Tokens [B, T_v, D] ──→ Evidence Score [B, 1]
                      │                       ↓
            ┌─────────┴──────────┐   ┌──────────────────┐
            │ Asymmetric         │←──│ EvidenceGate     │
            │ Interaction        │   │ (video→physical  │
            │ (video queries     │   │  modulation)     │
            │  physical tokens)  │   └──────────────────┘
            └─────────┬──────────┘
                      ↓
            Consensus Pooling + Mid-Fusion → Classifier → [B, 4]
```

**设计原则:**

1. **物理信号优先** — IMU + Audio 在低层融合为 Physical Dynamics Encoder，构成决策主干。Video 单向查询物理 token，无法反向污染物理信号。
2. **Evidence Gate** — Video 分支输出标量 evidence score，调制视频信息注入量（水花大 → 高 evidence，鱼在水下 → 低 evidence）。
3. **学生模型接口** — Physical Dynamics Encoder 的输出 + 独立分类头 = 可直接用作 Audio+IMU 学生模型的训练目标。

## 3. 组件设计

### 3.1 IMU Multi-Channel Loader (`datasets/loaders/wave_loaders.py`)

将当前平面化的 6/9 列 IMU 数据拆分为物理上有意义的通道：

```
输入: [T, 9] — acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, angle_x, angle_y, angle_z

通道拆分:
  channel_accel:  [T, 3] — 平移加速度 (col 0-2)
  channel_gyro:   [T, 3] — 角速度 (col 3-5)
  channel_angle:  [T, 3] — 姿态角 (col 6-8)

每个通道独立归一化（各自均值和标准差），保持物理含义的独立性。
```

- 新增 loader type: `imu_multichannel_loader`
- `transform()` 返回 `Dict[str, Tensor]`: `{imu_accel, imu_gyro, imu_angle}`，每个 `[T, 3]`
- 每个 channel 独立 padding 到 `max_length`

### 3.2 Multi-Channel TCN (`models/modelzoo/wave_models.py`)

```
Input: 3个通道，每个 [B, T, 3]
  → 对每个通道应用独立的小型 1D CNN (stem_conv)
  → Concat: [B, T, 3*C_ch]
  → Shared TCN: 多层膨胀卷积
  → Output: [B, T, D]
```

新增 backbone type: `multichannel_tcn`

配置:
- `channel_dim`: 每个通道的 stem 输出维度 (default 64)
- `tcn_layers`: TCN 层配置 (default [128, 256, 256])
- `output_dim`: 输出维度 (default 256)

### 3.3 Physical Dynamics Encoder (`models/fusion/physical_encoder.py`)

IMU + Audio 早期融合模块：

```
Input:
  imu_tokens:    [B, T_imu, D] — 来自 MultiChannelTCN
  audio_tokens:  [B, T_aud, D] — 来自 AudioCNN backbone (需先将 [B, D] 扩展为 [B, T_aud, D])

Step 1 — Temporal Alignment:
  统一时间网格 T_p = max(T_imu, T_aud)
  用线性插值将两个 token 序列对齐到相同长度

Step 2 — Physical Cross-Attention:
  两个方向:
    IMU → Audio: IMU tokens query Audio tokens (运动查询声音)
    Audio → IMU: Audio tokens query IMU tokens (声音查询运动)
  结果求和 (不是替换)

Step 3 — Shared Transformer:
  Concat[imu, audio] tokens → Self-Attention → Physical Tokens [B, T_p, D]
```

这是整个架构的核心组件，实现了 "IMU+Audio 物理共识"。

### 3.4 Video Backbone (`models/modelzoo/video_models.py`)

复用现有的 `VideoMAE` backbone (`register_backbone('video_mae')`)，配置:
- `img_size`: 224
- `patch_size`: 16
- `tube_size`: 2
- `embed_dim`: 768 → 投影到 D=256

输出: video tokens `[B, T_v, D]`

### 3.5 Asymmetric Interaction (`models/fusion/asymmetric_interaction.py`)

Video 单向查询 Physical tokens 的 cross-attention block:

```
Input:
  physical_tokens: [B, T_p, D]
  visual_tokens:   [B, T_v, D]

Operation:
  Q = visual_tokens, K/V = physical_tokens  ← 单向！
  attended_visual = CrossAttn(Q=visual, K=physical, V=physical)
  visual_tokens_out = visual_tokens + attended_visual  (残差)

  physical_tokens_out = physical_tokens  (不变 — Video 不污染物理信号)
```

可堆叠多个 block。注册为 fusion type: `asymmetric_video_physical`

### 3.6 Evidence Gate (`models/fusion/evidence_gate.py`)

Video tokens → scalar evidence score per sample:

```
Input: visual_tokens [B, T_v, D]
  → Pool (mean) → [B, D]
  → MLP: D → 64 → 1 → Sigmoid
  → evidence [B, 1]

用法:
  gated_visual = evidence.unsqueeze(-1) * visual_tokens_out
  # 当视频信息不可靠时（鱼在水下），evidence → 0，visual 信息被抑制
  # 当水花大、水面变化剧烈时，evidence → 1，visual 信息充分注入
```

Evidence Gate 也作为学生模型训练的信号：高 evidence 的样本 → teacher 更有信心，低 evidence 的样本 → teacher 输出更接近 Physical-only。

### 3.7 Mid-Fusion + Classifier

```
Input:
  physical_tokens:  [B, T_p, D]  (不变)
  gated_visual:     [B, T_v, D]  (evidence 调制后)

Physical Pooling:
  phys_pooled = physical_tokens.mean(dim=1)  → [B, D]

Visual Pooling:
  vis_pooled = gated_visual.mean(dim=1)      → [B, D]

Fused:
  fused = Concat[phys_pooled, vis_pooled]    → [B, 2*D]
  → Linear(2*D → D) → GELU → Dropout        → [B, D]

Classifier:
  Linear(D → 4)  → logits [B, 4]
```

### 3.8 学生模型蒸馏接口 (预留)

设计预留接口，具体蒸馏逻辑后续实现：

```python
class DistillationInterface(nn.Module):
    """学生模型蒸馏的多级知识接口"""

    def get_teacher_knowledge(self, batch) -> Dict:
        """返回 teacher 的多级知识"""
        return {
            "logits":           ...  # [B, 4]  — 分类 logits
            "phys_features":    ...  # [B, D]  — Physical Dynamics 融合特征
            "fused_features":   ...  # [B, D]  — 全模态融合特征
            "evidence_scores":  ...  # [B, 1]  — 每个样本的 visual evidence
            "sample_relations": ...  # [B, B] — batch 内样本相似度矩阵
        }
```

蒸馏损失（后续实现）：
- L_kl: KL(teacher_logits || student_logits)
- L_feat: MSE(teacher_phys_features, student_phys_features)
- L_rel: 样本关系蒸馏 (关系矩阵的 KL)
- L_vis: 视觉补偿损失 (高 evidence 样本 → student 模仿 teacher full fusion)

## 4. Pipeline V3 (`models/pipeline_v3.py`)

新 pipeline 实现 Physics-First 架构，不修改 pipeline_v2.py。

```python
class MultimodalPipelineV3(nn.Module):
    def __init__(self, stems, physical_encoder, video_backbone,
                 asymmetric_blocks, evidence_gate, mid_fusion_dim, num_classes):
        ...

    def forward(self, batch) -> Tuple[Tensor, Optional[Dict]]:
        # 1. Stems
        imu_feats = {ch: self.stems[ch](batch[ch]) for ch in ['imu_accel', 'imu_gyro', 'imu_angle']}
        audio_feats = self.stems['audio'](batch['audio'])
        video_tokens = self.stems['video'](batch['video'])

        # 2. Multi-channel IMU → unified IMU tokens
        imu_tokens = self.imu_encoder(imu_feats)

        # 3. Physical Dynamics Encoder (IMU + Audio)
        physical_tokens = self.physical_encoder(imu_tokens, audio_tokens)

        # 4. Asymmetric Interaction (Video → Physical only)
        visual_out, physical_out = self.asymmetric_blocks(visual_tokens, physical_tokens)

        # 5. Evidence Gate
        evidence = self.evidence_gate(visual_tokens)

        # 6. Fusion + Classify
        logits = self.fuse_and_classify(physical_out, visual_out, evidence)

        return logits, None  # aux placeholder for student distillation later
```

## 5. 配置 (`configs/physics_first_fusion.yaml`)

```yaml
data:
  modalities: [imu_accel, imu_gyro, imu_angle, audio, video]
  loaders:
    imu_accel:
      type: imu_multichannel_loader
      extra_params: { channel: accel, max_length: 512 }
    imu_gyro:
      type: imu_multichannel_loader
      extra_params: { channel: gyro, max_length: 512 }
    imu_angle:
      type: imu_multichannel_loader
      extra_params: { channel: angle, max_length: 512 }
    audio:
      type: audio_loader_stereo
      extra_params: { sample_rate: 16000, max_length: 160000, n_mels: 224, time_steps: 224 }
    video:
      type: video_loader
      extra_params: { num_frames: 16, frame_size: 224 }

model:
  backbones:
    imu_accel: { type: identity_stem }    # raw data, encoded by MultiChannelTCN
    imu_gyro:  { type: identity_stem }
    imu_angle: { type: identity_stem }
    audio:     { type: audiocnn, feature_dim: 512, pretrained: false, freeze: false }
    video:     { type: video_mae, img_size: 224, patch_size: 16, tube_size: 2 }

  unified_pipeline:
    token_dim: 256
    imu_channel_dim: 64
    tcn_output_dim: 256

    physical_encoder:
      num_layers: 2
      num_heads: 8
      shared_transformer_layers: 2

    asymmetric_interaction:
      num_blocks: 2
      num_heads: 4

    evidence_gate:
      hidden_dim: 64

    mid_fusion_output_dim: 256

    decision: { type: identity }

  dropout_rate: 0.35
```

## 6. 实现顺序

1. **IMU Multi-Channel Loader** — 改造 wave_loader，支持按物理通道分组输出
2. **Multi-Channel TCN** — 新增 backbone，三通道独立 stem + 共享 TCN
3. **Physical Dynamics Encoder** — 新增 fusion 模块，IMU+Audio 时序对齐 + 双向 cross-attention
4. **Asymmetric Interaction + Evidence Gate** — 新增 fusion 模块
5. **Pipeline V3** — 组合所有组件的新 pipeline
6. **Video Loader** — 新增 video_loader（从 mp4 提取帧）
7. **Config + Train Script** — 端到端训练入口
8. **Distillation Interface** — 预留接口（仅接口，不实现训练逻辑）
