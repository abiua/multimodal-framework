# Physics-First Asymmetric Fusion 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 实现视频+音频+IMU三模态物理优先非对称融合架构，用于鱼类摄食强度四分类

**Architecture:** IMU三通道 → MultiChannelTCN → Physical Dynamics Encoder (IMU+Audio早期融合) → Asymmetric Interaction (Video单向查询Physical) → Evidence Gate → 分类

**Tech Stack:** PyTorch, torchvision, torchaudio, OmegaConf, einops

---

## Implementation Order (12 tasks, sequential dependencies)

### Task 1: Data Preparation Script
- Create: `scripts/prepare_physics_data.py`
- Convert `data/multimodal_data_original/` → `data/fish_feeding_v3/` (class-per-subdirectory layout)
- Stratified train/val/test split, copy audio.wav + imu.csv + video.mp4

### Task 2: IMU Multi-Channel Loader
- Modify: `datasets/loaders/wave_loaders.py` — append `ImuChannelLoader`
- 3 loaders (accel/gyro/angle), each extracts cols 3-5/6-8/9-12 from same imu.csv
- Independent per-channel normalization

### Task 3: Identity Stem
- Modify: `models/modelzoo/common.py` — append `IdentityStem` (pass-through, feature_dim=3)

### Task 4: Multi-Channel TCN
- Create: `models/modelzoo/multichannel_tcn.py`
- `MultiChannelTCN`: 3 independent `ChannelStem` → concat → shared TCN (dilated convs) → output projection
- Output: `[B, T, 256]`

### Task 5: Physical Dynamics Encoder
- Create: `models/fusion/physical_encoder.py`
- `PhysicalDynamicsEncoder`: temporal alignment (linear interpolate) → bidirectional cross-attention (IMU↔Audio) → shared transformer → physical tokens `[B, T_p, D]`

### Task 6: Asymmetric Interaction + Evidence Gate
- Create: `models/fusion/asymmetric_interaction.py`
- `AsymmetricInteraction`: stacked blocks, Q=video, K/V=physical, physical unchanged
- `EvidenceGate`: video tokens → mean pool → MLP → sigmoid → `[B, 1]` evidence score

### Task 7: Video Loader
- Create: `datasets/loaders/video_loaders.py`
- `VideoFrameLoader`: torchvision read_video → uniform N-frame sampling → resize → normalize

### Task 8: Pipeline V3
- Create: `models/pipeline_v3.py`
- `MultimodalPipelineV3`: assembles all components
- Includes `audio_proj` (512→256) and `video_proj` (768→256) Linear layers
- `get_teacher_knowledge()` distillation interface
- Loss: fusion_loss + 0.3 × physical_only_loss

### Task 9: Config + Train Script
- Create: `configs/physics_first_fusion.yaml` + `scripts/train_physics.py`
- AdamW, cosine LR, early stopping, label smoothing
- `build_pipeline_v3()` manual construction function

### Task 10: Dataset Extension
- Modify: `datasets/factory.py` — add `imu_accel/imu_gyro/imu_angle` → `.csv` extension mappings

### Task 11: Integration Tests
- Create: `tests/test_pipeline_v3.py`
- Test each component independently + end-to-end forward/backward + distillation interface

### Task 12: E2E Dry Run
- 5-epoch training test with real data

---

## Key Design Decisions

1. **Dimension alignment**: Pipeline V3 constructor takes explicit `audio_dim` and `video_dim` parameters (512 and 768), uses `nn.Linear` projections to unified `D=256`
2. **IMU modality registration**: Three virtual modalities share same `.csv` file, each loader extracts different columns
3. **Physical-only loss**: Auxiliary loss on `phys_classifier` (0.3× weight) ensures Audio+IMU learn independently
4. **Distillation interface**: `get_teacher_knowledge()` returns logits, phys_features, fused_features, evidence_scores, phys_logits
