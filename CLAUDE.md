# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/pythoner/abiu/multimodal-framework torchrun --nproc_per_node=1 --master_port=29501 -m tools.train --config configs/fish_feeding_unireplknet.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/home/pythoner/abiu/multimodal-framework torchrun --nproc_per_node=4 --master_port=29501 -m tools.train --config configs/fish_feeding_unireplknet.yaml
```

### Evaluation

```bash
python -m tools.eval --config configs/fish_feeding_unireplknet.yaml --checkpoint output/<experiment>/best_model.pth
```

### Running tests

```bash
PYTHONPATH=/home/pythoner/abiu/multimodal-framework python -m pytest tests/ -v --rootdir=tests
# --rootdir=tests required: the project root __init__.py has relative imports that conflict with pytest discovery
```

## Architecture

The framework has two pipeline architectures that coexist:

### V1 Pipeline (`MultimodalClassifier` — `models/builder.py`)

Traditional backbone → mid-fusion → classifier head. Supports optional **staged forward**: backbones implementing `StageableBackbone` can expose intermediate stages for cross-modal mid-fusion via `StageFusionAdapter`. Route selected when `config.model.unified_pipeline` is `None`.

### V2 Pipeline (`MultimodalPipelineV2` — `models/pipeline_v2.py`)

New unified flow: `Stem → Tokenization → InteractionBlocks → Per-Modal Pooling → Mid Fusion → Decision → Classifier`. Route selected when `config.model.unified_pipeline` is set. All components are config-driven and replaceable.

### V3 / SACF v2 Pipelines

- **V3** (`models/pipeline_v3.py`): Physics-First Asymmetric Fusion (IMU/Wave+Audio+Image). EvidenceGate confirmed dead (gradient≈0 across datasets/visual modalities).
- **SACF v2** (`models/pipeline_sacf.py`): Stage-Aware Consensus Fusion — Wave↔Audio bidirectional cross-attn → FiLM modulation + gated residual from Image → 3 independent classifiers + consensus KL loss. Modal dropout (30%) with learnable null tokens.
- SACF training: `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/home/pythoner/abiu/multimodal-framework torchrun --nproc_per_node=4 --master_port=29505 scripts/train_sacf.py --config configs/sacf_fish_feeding.yaml --output output/sacf_fish_feeding`
- SACF eval (module import to avoid `__main__` quirks): `PYTHONPATH=/home/pythoner/abiu/multimodal-framework python -B -c "import sys; sys.path.insert(0,'scripts'); sys.argv[1:]=['--config','configs/sacf_fish_feeding.yaml','--checkpoint','output/sacf_fish_feeding/best_model.pth','--gpu','0','--output','output/sacf_fish_feeding/analysis']; from evaluate_sacf import main; main()"`

### Registry Pattern

Three registries auto-discover modules from their respective directories:
- **`ModelZoo`** (`models/registry.py`) — backbones in `models/modelzoo/`, decorated with `@register_backbone(name)`
- **`LoaderRegistry`** (`datasets/registry.py`) — data loaders in `datasets/loaders/`, decorated with `@register_loader(name)`
- **`FusionRegistry`** (`models/fusion/registry.py`) — cross-modal fusion strategies for V2 interaction blocks
- **`DecisionRegistry`** (`models/decision.py`) — post-fusion decision modules (identity, evidence, etc.)

### Config System (`utils/config.py`)

OmegaConf dataclass hierarchy: `Config → {Data, Class, Model, Train, Eval, System}Config`. `load_config()` merges structured schema + user YAML + CLI overrides, then runs `validate_config()` for runtime checks.

### Data Flow

`DataFactory` creates `MultimodalDataset` instances that scan a class-per-subdirectory layout, matching samples across modalities by filename stem (e.g., `sample_001.jpg` + `sample_001.wav`). Each modality gets its own `BaseLoader` for loading + transformations.

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `models/modelzoo/` | Backbone implementations (resnet, audiocnn, tcn, unireplknet, video_mae, etc.) |
| `models/fusion/` | Fusion strategies (gate, cross_attn, token_mix, etc.) |
| `models/distillation/` | Cross-modal teacher-student distillation (contrastive loss + KL divergence) |
| `models/heads/` | Classifier head and MultimodalFusion |
| `datasets/loaders/` | Per-modality data loaders |
| `trainers/` | Trainer, DistillationTrainer, checkpoint manager |
| `utils/` | Config, distributed utils, metrics, SwanLab logger (TensorBoard deprecated) |
| `tools/` | Entry-point scripts (train, eval, train_distill) |
| `configs/` | YAML config files per experiment |
| `scripts/` | Data preparation utilities |

### Backbone Interface (`models/backbone_base.py`)

- **`BaseBackbone`**: standard backbones with `forward()` → `[B, feature_dim]`, plus optional `tokenize()` for V2 pipeline.
- **`StageableBackbone`**: backbones exposing `init_state()`, `forward_stage(state, i)`, `forward_head(state)` for staged mid-fusion in V1.

### Adding New Backbones

- New modelzoo modules MUST be imported in `models/modelzoo/__init__.py` to trigger auto-registration
- Check for name collisions with existing `@register_backbone` entries before naming
- HuggingFace downloads need `http_proxy=http://127.0.0.1:7890 https_proxy=...` env vars (same for pip GitHub installs)

### HuggingFace Backbone Integration

- Use `torchaudio` transforms (MelSpectrogram + AmplitudeToDB) instead of HF feature extractor for GPU-speed batch processing
- AudioSet mel normalization: `(x - (-4.2677393)) / (4.5689974 * 2)`
- Solo audio training: `modalities: [audio]`, `mid_fusion_enabled: false`, data paths to `data/fish_feeding_local/{train,val,test}`
- FP16 may trigger `GradScaler.unscale_()` double-call crash on some models — disable with `fp16: false` if it occurs
- Two audio loader types: `audio_raw_loader` (raw waveform for AST/BEATs) vs `audio_loader_stereo` (2ch mel spectrogram)

### Distributed Training

The `Trainer` class handles DDP, mixed precision (FP16 with `GradScaler`), gradient clipping, non-finite gradient detection/skip, and distributed metric reduction. `tools/train.py` auto-detects `torchrun` environment variables for multi-GPU setup.

- DDP with modal dropout: requires `find_unused_parameters=True` (masked modalities don't participate in loss)
- Always `torch.cuda.synchronize()` before `dist.destroy_process_group()` to avoid NCCL cleanup timeout
- Deprecated AMP API: use `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')`, not `torch.cuda.amp.*`

### Logging & Monitoring

- **SwanLab** replaced TensorBoard. Default: `swanlab_enabled: true`, project `"MM"`, TensorBoard disabled.
- API key in `swanlab_api_key.txt` (gitignored). `SwanLabLogger` reads it at init time — never set as global env var.
- Web dashboard: https://swanlab.cn/@abiu/MM

### Evaluation Gotchas

- **Ablation MUST run before gradient analysis**: gradient analysis uses `model.train()` which corrupts BatchNorm running statistics. Either reorder (ablation first) or reload model checkpoint after gradient analysis.
- Batch dict structure: `image` is direct Tensor, `audio` is `{'mel_spectrogram': Tensor}`, `wave` is `{'wave': Tensor}`. Zero-out in ablation must match nested dict structure.
- Create fresh DataLoaders per ablation scenario to avoid iterator state issues.
- File persistence: `/home/ai/data/pythoner/` and `/home/pythoner/` are same filesystem but different mount paths. If Write tool fails to persist, use Bash heredoc.

## Known Test Failures

- `test_pipeline_v2.py`: 3 failures (shape mismatch, pre-existing)
- `test_pipeline_v3.py`: 4 errors (FileNotFoundError/TypeError, pre-existing)
