# Retrain All Audio Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Retrain all audio models with improved regularization strategies + add 48kHz from-scratch experiment.

**Architecture:** Pretrained models (AST/SSLAM/BEATs) stay at 16kHz — their mel filterbank and positional encoding are locked to 128-bin 16kHz. Strategy: freeze encoder or use stronger regularization. New 48kHz experiment uses audio_resnet50 from scratch with 256-bin mel.

**Tech Stack:** PyTorch, torchrun (multi-GPU), SwanLab

---

## Constraint: Pretrained Models Locked to 16kHz

AST/SSLAM/BEATs all have hardcoded 16kHz processing:
- 128-bin mel filterbank designed for 0-8kHz Nyquist
- Patch embedding expects 128 mel bins × N time frames
- Positional encoding for specific [T, 128] grid

**Cannot feed 48kHz without rewriting the model internals.** The pretrained weights would be incompatible.

---

## Retrain Strategy

| Model | Freq | Strategy | Why |
|-------|------|----------|-----|
| AST pretrained | 16kHz | freeze encoder + higher lr head | Prevent overfitting (27% gap → target <10%) |
| SSLAM | 16kHz | freeze encoder + lower lr | Avoid catastrophic forgetting |
| BEATs | 16kHz | freeze encoder + moderate lr | Reduce overfitting |
| audio_resnet50 | **48kHz** | from scratch, 256-bin mel | Capture 8-24kHz high-freq info |

All use: higher dropout (0.5), label_smoothing (0.1), mixup (0.2), cosine LR, adamw.

---

### Task 1: AST pretrained — freeze encoder, retrain head

**Config:** `configs/solo_audio_ast_frozen.yaml` — already exists, already has freeze=true
**Output:** `output/solo_audio_ast_frozen_v2`

Run:
```bash
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/home/pythoner/abiu/multimodal-framework \
  torchrun --nproc_per_node=4 --master_port=29510 \
  bash -c 'exec -a ast_frozen_v2 python -m tools.train --config configs/solo_audio_ast_frozen.yaml' &
```
Note: change output_dir to `output/solo_audio_ast_frozen_v2` before running.

### Task 2: SSLAM — freeze encoder, retrain head

**New config:** `configs/solo_audio_sslam_frozen.yaml` (copy from sclam config, add freeze=true, lower lr=0.0001, higher dropout=0.5)
**Output:** `output/solo_audio_sslam_frozen`

Run:
```bash
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/home/pythoner/abiu/multimodal-framework \
  torchrun --nproc_per_node=4 --master_port=29511 \
  bash -c 'exec -a sslam_frozen python -m tools.train --config configs/solo_audio_sslam_frozen.yaml' &
```

### Task 3: BEATs — freeze encoder, retrain head

**New config:** `configs/solo_audio_beats_frozen.yaml`
**Output:** `output/solo_audio_beats_frozen`

Run:
```bash
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/home/pythoner/abiu/multimodal-framework \
  torchrun --nproc_per_node=4 --master_port=29512 \
  bash -c 'exec -a beats_frozen python -m tools.train --config configs/solo_audio_beats_frozen.yaml' &
```

### Task 4: 48kHz from scratch — audio_resnet50 + mono mel

**New loader needed:** Modify `audio_loader` to support 48kHz params. Already supports `sample_rate` param.

**Config:** `configs/solo_audio_48k.yaml` — use `audio_loader` (mono) at 48kHz/256mel + `audio_resnet50` from scratch
**Output:** `output/solo_audio_48k`

Run:
```bash
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/home/pythoner/abiu/multimodal-framework \
  torchrun --nproc_per_node=4 --master_port=29513 \
  bash -c 'exec -a audio48k python -m tools.train --config configs/solo_audio_48k.yaml' &
```
