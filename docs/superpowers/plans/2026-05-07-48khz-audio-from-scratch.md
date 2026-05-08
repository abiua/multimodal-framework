# 48kHz From-Scratch Audio Training (AST + BEATs)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train AST and BEATs from scratch at native 48kHz (no pretrained weights, no downsampling) on fish feeding audio, both simultaneously on 4 GPUs.

**Architecture:** Modify ASTBackbone to accept configurable mel spectrogram params (sample_rate, n_fft, hop_length, n_mels, f_max). Modify BEATsBackbone to override SpeechBrain's hardcoded 16kHz fbank for 48kHz. Create two 48kHz configs. Launch both with DDP (2 GPUs each) via separate torchrun processes.

**Tech Stack:** PyTorch, torchaudio (MelSpectrogram), HuggingFace Transformers (ASTModel without pretrained), SpeechBrain (BEATs without checkpoint), torchrun DDP

---

## File Structure

```
models/modelzoo/audio_ast.py       — MODIFY: add configurable mel params for 48kHz
models/modelzoo/audio_beats.py     — MODIFY: override preprocess for 48kHz fbank
configs/solo_audio_ast_48k.yaml    — CREATE: AST from-scratch 48kHz config
configs/solo_audio_beats_48k.yaml  — CREATE: BEATs from-scratch 48kHz config
```

---

### Task 1: Add configurable mel parameters to ASTBackbone

**Files:**
- Modify: `models/modelzoo/audio_ast.py`

The current `ASTBackbone.__init__` hardcodes `MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160, n_mels=128, f_max=8000)`. We need to make these configurable while keeping 16kHz as default.

- [ ] **Step 1: Add mel parameters to `__init__` signature**

Replace the `__init__` signature and mel transform setup:

```python
def __init__(
    self,
    feature_dim: int = 768,
    pretrained: bool = True,
    dropout: float = 0.1,
    freeze: bool = False,
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    max_length: int = 1024,
    num_mel_bins: int = 128,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    f_max: int = 8000,
    **kwargs,
):
    super().__init__()
    self.model_name = model_name
    self._max_length = max_length
    self._num_mel_bins = num_mel_bins

    # GPU-friendly mel spectrogram (configurable for 16k/48k)
    self.mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        n_mels=num_mel_bins, f_min=0, f_max=f_max,
        power=2.0, norm=None,
    )
    self.db_transform = T.AmplitudeToDB()

    from transformers import ASTModel

    if pretrained:
        self.encoder = ASTModel.from_pretrained(model_name)
    else:
        from transformers import ASTConfig
        cfg = ASTConfig.from_pretrained(model_name)
        cfg.max_length = max_length
        cfg.num_mel_bins = num_mel_bins
        cfg.hidden_dropout_prob = dropout
        cfg.attention_probs_dropout_prob = dropout
        for k in ('num_hidden_layers', 'num_attention_heads',
                   'hidden_size', 'intermediate_size'):
            if k in kwargs:
                setattr(cfg, k, kwargs[k])
        self.encoder = ASTModel(cfg)

    if 'hidden_size' in kwargs:
        self.feature_dim = kwargs['hidden_size']

    if freeze:
        for p in self.encoder.parameters():
            p.requires_grad = False
```

- [ ] **Step 2: Verify import and registration**

Run: `PYTHONPATH=/home/pythoner/abiu/multimodal-framework python -c "from models.modelzoo.audio_ast import ASTBackbone; print('OK')"`
Expected: OK

---

### Task 2: Modify BEATsBackbone for 48kHz from-scratch support

**Files:**
- Modify: `models/modelzoo/audio_beats.py`

SpeechBrain BEATs hardcodes `sample_frequency=16000` and `num_mel_bins=128` in `preprocess()`. For from-scratch 48kHz training, we need to override this method on the BEATs instance.

- [ ] **Step 1: Rewrite `audio_beats.py` with configurable preprocess**

```python
"""BEATs backbone — ICML 2023, Microsoft acoustic tokenizer SSL.

Uses SpeechBrain's BEATs implementation. Supports 16kHz (pretrained) and
48kHz (from-scratch with custom fbank params).

Input: raw waveform mono → Kaldi fbank (configurable sample_rate / n_mels) → ViT encoder
"""

from __future__ import annotations

import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi

from ..backbone_base import BaseBackbone
from ..registry import register_backbone


@register_backbone('beats', description='BEATs audio encoder (ICML 2023, AudioSet-2M pretrained)', modality='audio')
class BEATsBackbone(BaseBackbone):
    feature_dim = 768

    def __init__(
        self,
        feature_dim: int = 768,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze: bool = False,
        checkpoint_path: str | None = None,
        sample_rate: int = 16000,
        num_mel_bins: int = 128,
        **kwargs,
    ):
        super().__init__()

        from speechbrain.lobes.models.beats import BEATs

        self._sample_rate = sample_rate
        self._num_mel_bins = num_mel_bins

        if pretrained:
            if checkpoint_path is None:
                checkpoint_path = '/home/ai/data/pythoner/abiu/multimodal-framework/models/modelzoo/BEATs/BEATs_iter3_plus_AS2M.pt'

            import os
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"BEATs checkpoint not found: {checkpoint_path}\n"
                    "Download from: https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf"
                )
            self.encoder = BEATs(ckp_path=checkpoint_path, freeze=False)
        else:
            self.encoder = BEATs(ckp_path=None, freeze=False)
            # Override preprocess for custom sample_rate / n_mels
            self._patch_preprocess()

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _patch_preprocess(self):
        """Replace the instance's preprocess with a version using our sample_rate/n_mels."""
        sample_rate = self._sample_rate
        num_mel_bins = self._num_mel_bins

        def custom_preprocess(self_beats, source, fbank_mean=15.41663, fbank_std=6.55582):
            fbanks = []
            for waveform in source:
                waveform = waveform.unsqueeze(0) * 2**15
                fbank = ta_kaldi.fbank(
                    waveform,
                    num_mel_bins=num_mel_bins,
                    sample_frequency=sample_rate,
                    frame_length=25,
                    frame_shift=10,
                )
                fbanks.append(fbank)
            fbank = torch.stack(fbanks, dim=0)
            return (fbank - fbank_mean) / (2 * fbank_std)

        self.encoder.preprocess = types.MethodType(custom_preprocess, self.encoder)

    def forward(self, x=None, **inputs):
        if x is None:
            if 'waveform' in inputs:
                x = inputs['waveform']
            elif isinstance(inputs, dict):
                for v in inputs.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        x = v
                        break

        # Mono: [B, T]
        if x.dim() == 3 and x.size(1) > 1:
            x = x.mean(dim=1, keepdim=False)
        elif x.dim() == 3:
            x = x.squeeze(1)

        import torch as _torch
        B = x.shape[0]
        wav_lens = _torch.ones(B, device=x.device)
        out = self.encoder.extract_features(x, wav_lens=wav_lens)
        feats = out[0] if isinstance(out, tuple) else out
        return feats.mean(dim=1)  # [B, 768]
```

- [ ] **Step 2: Verify BEATs from-scratch with 48kHz fbank**

Run:
```bash
PYTHONPATH=/home/pythoner/abiu/multimodal-framework python -c "
import torch
from models.modelzoo.audio_beats import BEATsBackbone

# 48kHz from-scratch
m = BEATsBackbone(pretrained=False, sample_rate=48000, num_mel_bins=256)
x = torch.randn(2, 48000)
y = m(x)
print(f'Output shape: {y.shape}')  # expected: [2, 768]
print('BEATs 48kHz from-scratch OK')
"
```
Expected: `Output shape: torch.Size([2, 768])`

---

### Task 3: Create config files

**Files:**
- Create: `configs/solo_audio_ast_48k.yaml`
- Create: `configs/solo_audio_beats_48k.yaml`

- [ ] **Step 1: Create `configs/solo_audio_ast_48k.yaml`**

```yaml
# AST from-scratch 48kHz — preserves native 48kHz audio (0-24kHz)
data:
  batch_size: 32
  num_workers: 4
  pin_memory: true

  train_path: "data/fish_feeding_local/train"
  val_path: "data/fish_feeding_local/val"
  test_path: "data/fish_feeding_local/test"

  modalities:
    - audio

  loaders:
    audio:
      type: audio_raw_loader
      extra_params:
        sample_rate: 48000
        max_length: 48000

  image_size: 224

classes:
  num_classes: 3
  class_names:
    - "None"
    - "Strong"
    - "Weak"

model:
  backbones:
    audio:
      type: ast_hf
      feature_dim: 768
      pretrained: false
      freeze: false
      extra_params:
        sample_rate: 48000
        n_fft: 2048
        hop_length: 480
        num_mel_bins: 256
        f_max: 24000
        max_length: 256

  fusion_type: concat
  fusion_hidden_dim: 512
  mid_fusion_enabled: false

  dropout_rate: 0.3
  classifier_hidden_dims: [512, 256]

train:
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001

  lr_scheduler: cosine
  warmup_epochs: 5
  step_size: 30
  gamma: 0.1

  optimizer: adamw
  momentum: 0.9

  label_smoothing: 0.1
  mixup_alpha: 0.2
  cutmix_alpha: 0.0

  early_stop:
    enabled: true
    patience: 15
    min_delta: 0.001
    monitor: accuracy
    mode: max

  val_interval: 1

eval:
  metrics: [accuracy, precision, recall, f1]
  save_predictions: true
  confusion_matrix: true

system:
  seed: 42
  gpu_ids: [0, 1]
  distributed: true
  fp16: false

  log_interval: 10
  save_interval: 10

  output_dir: "output/solo_audio_ast_48k"
  resume: ""

  dist_backend: "nccl"
  dist_url: "env://"

  tensorboard_enabled: false
  experiment_name: "solo_audio_ast_48k"
```

- [ ] **Step 2: Create `configs/solo_audio_beats_48k.yaml`**

```yaml
# BEATs from-scratch 48kHz — preserves native 48kHz audio (0-24kHz)
data:
  batch_size: 16
  num_workers: 4
  pin_memory: true

  train_path: "data/fish_feeding_local/train"
  val_path: "data/fish_feeding_local/val"
  test_path: "data/fish_feeding_local/test"

  modalities:
    - audio

  loaders:
    audio:
      type: audio_raw_loader
      extra_params:
        sample_rate: 48000
        max_length: 48000

  image_size: 224

classes:
  num_classes: 3
  class_names:
    - "None"
    - "Strong"
    - "Weak"

model:
  backbones:
    audio:
      type: beats
      feature_dim: 768
      pretrained: false
      freeze: false
      extra_params:
        sample_rate: 48000
        num_mel_bins: 256

  fusion_type: concat
  fusion_hidden_dim: 512
  mid_fusion_enabled: false

  dropout_rate: 0.3
  classifier_hidden_dims: [256]

train:
  epochs: 50
  learning_rate: 0.0005
  weight_decay: 0.01

  lr_scheduler: cosine
  warmup_epochs: 5
  step_size: 30
  gamma: 0.1

  optimizer: adamw
  momentum: 0.9

  label_smoothing: 0.1
  mixup_alpha: 0.0
  cutmix_alpha: 0.0

  early_stop:
    enabled: true
    patience: 15
    min_delta: 0.001
    monitor: accuracy
    mode: max

  val_interval: 1

eval:
  metrics: [accuracy, precision, recall, f1]
  save_predictions: true
  confusion_matrix: true

system:
  seed: 42
  gpu_ids: [2, 3]
  distributed: true
  fp16: false

  log_interval: 10
  save_interval: 10

  output_dir: "output/solo_audio_beats_48k"
  resume: ""

  dist_backend: "nccl"
  dist_url: "env://"

  tensorboard_enabled: false
  experiment_name: "solo_audio_beats_48k"
```

Note: AST uses GPUs 0,1 (batch_size=32); BEATs uses GPUs 2,3 (batch_size=16 due to 90M params).

---

### Task 4: Launch both trainings simultaneously

- [ ] **Step 1: Launch AST 48kHz training on GPUs 0,1**

```bash
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/home/pythoner/abiu/multimodal-framework \
  bash -c 'exec -a ast48k_train python -m tools.train --config configs/solo_audio_ast_48k.yaml' &
```

- [ ] **Step 2: Launch BEATs 48kHz training on GPUs 2,3** (different master_port to avoid conflict)

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/pythoner/abiu/multimodal-framework \
  bash -c 'exec -a beats48k_train python -m torch.distributed.run --nproc_per_node=2 --master_port=29502 -m tools.train --config configs/solo_audio_beats_48k.yaml' &
```

Wait — `tools.train` uses internal DDP init. Let me check how it handles the torchrun environment.

Actually, looking at the CLAUDE.md commands, the standard approach is using `torchrun` directly:
```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/home/pythoner/abiu/multimodal-framework \
  torchrun --nproc_per_node=2 --master_port=29501 -m tools.train --config configs/solo_audio_ast_48k.yaml
```

So:

- [ ] **Step 1 (corrected): Launch AST 48kHz on GPUs 0,1**

```bash
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/home/pythoner/abiu/multimodal-framework \
  bash -c 'exec -a ast48k_train torchrun --nproc_per_node=2 --master_port=29501 -m tools.train --config configs/solo_audio_ast_48k.yaml' &
```

- [ ] **Step 2: Launch BEATs 48kHz on GPUs 2,3**

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/pythoner/abiu/multimodal-framework \
  bash -c 'exec -a beats48k_train torchrun --nproc_per_node=2 --master_port=29502 -m tools.train --config configs/solo_audio_beats_48k.yaml' &
```

- [ ] **Step 3: Verify both processes are running**

```bash
ps aux | grep -E "ast48k|beats48k" | grep -v grep
```

Expected: Two processes with distinct names.

---

## Self-Review

1. **Spec coverage:** All requirements covered — configurable mel params for AST, 48kHz fbank override for BEATs, two configs, simultaneous training on 4 GPUs.

2. **Placeholder scan:** No TBD, TODO, or vague instructions. All code is shown in full.

3. **Type consistency:** BEATsBackbone `_patch_preprocess` uses `self._sample_rate` and `self._num_mel_bins` which are set in `__init__`. ASTBackbone new params (`sample_rate`, `n_fft`, `hop_length`, `f_max`) are consumed immediately in the mel_transform setup.
