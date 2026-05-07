# SSLAM Solo Audio Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate SSLAM (ICLR 2025) pretrained encoder as an audio backbone for solo fish feeding audio classification.

**Architecture:** SSLAM's ViT-B encoder (12 layers, 768-dim) loaded via fairseq checkpoint, wrapped as a `BaseBackbone` in ModelZoo. Raw waveform input → Kaldi 128-mel fbank → AudioSet norm → ViT → CLS token [B, 768] → Classifier head.

**Tech Stack:** fairseq (included clone), torch, torchaudio, existing ModelZoo registry

---

### Task 1: Fix fairseq import path

**Files:**
- Modify: `models/modelzoo/SSLAM-main/SSLAM_Inference/feature_extract/feature_extract.py:14`
- Modify: `models/modelzoo/SSLAM-main/SSLAM_Inference/inference/inference.py` (line with fairseq_path)
- Modify: `models/modelzoo/SSLAM-main/SSLAM_Inference/evaluation/eval.py` (line with fairseq_path)

- [ ] **Step 1: Update fairseq_path to absolute path**

In each of the 3 files, change:
```python
fairseq_path = '/path/to/SSLAM/SSLAM_Inference/cloned_fairseq_copy/fairseq/'
```
to:
```python
fairseq_path = '/home/ai/data/pythoner/abiu/multimodal-framework/models/modelzoo/SSLAM-main/SSLAM_Inference/cloned_fairseq_copy/fairseq/'
```

- [ ] **Step 2: Verify fairseq import works**

```bash
cd /home/pythoner/abiu/multimodal-framework && python3 -c "
import sys
sys.path.append('models/modelzoo/SSLAM-main/SSLAM_Inference/cloned_fairseq_copy/fairseq/')
import fairseq
print('fairseq OK:', fairseq.__file__)
"
```
Expected: Prints fairseq path

- [ ] **Step 3: Commit**

```bash
git add models/modelzoo/SSLAM-main/SSLAM_Inference/
git commit -m "fix: update fairseq_path to project absolute path"
```

---

### Task 2: Download SSLAM pretrained checkpoint

- [ ] **Step 1: Download pretrained weights**

Google Drive: https://drive.google.com/drive/folders/1aA65-qQCHSCrkiDeLGUtn1PiEjJi5HS8

Download `SSLAM_pretrained.pt` and place at:
```
models/modelzoo/SSLAM-main/checkpoints/SSLAM_pretrained.pt
```

Note: This requires manual download. If Google Drive is inaccessible, use the HuggingFace version via proxy on port 7890:
```bash
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('ta012/SSLAM_pretrain', 'SSLAM_pretrained.pt', local_dir='models/modelzoo/SSLAM-main/checkpoints/')"
```

---

### Task 3: Create SSLAM backbone wrapper

**Files:**
- Create: `models/modelzoo/audio_sslam.py`

- [ ] **Step 1: Write the backbone**

```python
"""SSLAM backbone — ICLR 2025 audio encoder (data2vec 2.0 / EAT architecture).

Registers 'sslam' backbone. Uses pretrained SSLAM ViT encoder loaded via fairseq.
Expects raw waveform input (16kHz mono) from audio_raw_loader.
"""

from __future__ import annotations

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ..backbone_base import BaseBackbone
from ..registry import register_backbone

# AudioSet mel spectrogram normalization
AS_MEAN = -4.268
AS_STD = 4.569

# Absolute path to fairseq clone
FAIRSEQ_PATH = '/home/ai/data/pythoner/abiu/multimodal-framework/models/modelzoo/SSLAM-main/SSLAM_Inference/cloned_fairseq_copy/fairseq/'

if FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, FAIRSEQ_PATH)


@register_backbone('sslam', description='SSLAM pretrained audio encoder (ICLR 2025)', modality='audio')
class SSLAMBackbone(BaseBackbone):
    """SSLAM backbone — ViT-B encoder pretrained with mixture SSL on AudioSet-2M.

    Input: raw waveform [B, T] (16kHz mono) or {'waveform': Tensor[B, T]}
    Output: feature vector [B, 768] (CLS token)

    Args:
        checkpoint_path: path to SSLAM pretrained .pt file
        model_dir: path to SSLAM_Finetune models directory (for fairseq user_dir)
        target_length: mel spectrogram time frames (1024 for ~10s audio)
        freeze: freeze encoder weights
    """

    feature_dim = 768

    def __init__(
        self,
        feature_dim: int = 768,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze: bool = False,
        model_dir: str | None = None,
        checkpoint_path: str | None = None,
        target_length: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.target_length = target_length

        if model_dir is None:
            model_dir = '/home/ai/data/pythoner/abiu/multimodal-framework/models/modelzoo/SSLAM-main/SSLAM_Finetune/SSLAM_Ft_AS20K/'
        if checkpoint_path is None:
            checkpoint_path = '/home/ai/data/pythoner/abiu/multimodal-framework/models/modelzoo/SSLAM-main/checkpoints/SSLAM_pretrained.pt'

        self.encoder = None

        if pretrained and os.path.exists(checkpoint_path):
            import fairseq
            from fairseq.utils import import_user_module
            from dataclasses import dataclass

            @dataclass
            class UserDirModule:
                user_dir: str

            model_path = UserDirModule(model_dir)
            import_user_module(model_path)
            model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
            self.encoder = model[0].model if hasattr(model[0], 'model') else model[0]
            self.encoder.eval()
        else:
            raise FileNotFoundError(
                f"SSLAM checkpoint not found at {checkpoint_path}. "
                "Download from https://drive.google.com/drive/folders/1aA65-qQCHSCrkiDeLGUtn1PiEjJi5HS8"
            )

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _to_mel(self, x: torch.Tensor) -> torch.Tensor:
        """Convert waveform [B, T] to normalized mel [B, T_frames, 128] using Kaldi fbank."""
        B = x.shape[0]
        mels = []
        for i in range(B):
            wav = x[i]
            wav = wav - wav.mean()
            mel = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0), htk_compat=True, sample_frequency=16000,
                use_energy=False, window_type='hanning', num_mel_bins=128,
                dither=0.0, frame_shift=10,
            )  # [T_frames, 128]
            n_frames = mel.shape[0]
            diff = self.target_length - n_frames
            if diff > 0:
                mel = F.pad(mel, (0, 0, 0, diff))
            elif diff < 0:
                mel = mel[:self.target_length]
            mels.append(mel)

        mel = torch.stack(mels, dim=0).to(device=x.device, dtype=x.dtype)  # [B, T, 128]
        mel = (mel - AS_MEAN) / (AS_STD * 2)
        return mel

    def forward(self, x=None, **inputs):
        if x is None:
            if 'waveform' in inputs:
                x = inputs['waveform']
            elif isinstance(inputs, dict):
                for v in inputs.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        x = v
                        break

        if x.dim() == 3 and x.size(1) > 1:
            x = x.mean(dim=1, keepdim=False)
        elif x.dim() == 3:
            x = x.squeeze(1)

        mel = self._to_mel(x)  # [B, T, 128]
        mel = mel.unsqueeze(1)  # [B, 1, T, 128] — fairseq expects channel dim

        with torch.set_grad_enabled(self.training):
            feats = self.encoder.extract_features(
                mel, padding_mask=None, mask=False, remove_extra_tokens=False
            )
            x = feats['x'][:, 0]  # CLS token [B, 768]

        return x
```

- [ ] **Step 2: Commit**

```bash
git add models/modelzoo/audio_sslam.py
git commit -m "feat: add SSLAM backbone wrapper for solo audio training"
```

---

### Task 4: Register SSLAM in ModelZoo __init__.py

**Files:**
- Modify: `models/modelzoo/__init__.py`

- [ ] **Step 1: Import audio_sslam**

```python
from . import audio_ast
from . import audio_sslam
```

- [ ] **Step 2: Commit**

```bash
git add models/modelzoo/__init__.py
git commit -m "feat: register sslam backbone in ModelZoo"
```

---

### Task 5: Create solo audio SSLAM config

**Files:**
- Create: `configs/solo_audio_sslam.yaml`

- [ ] **Step 1: Write the config**

```yaml
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
        sample_rate: 16000
        max_length: 160000  # 10s audio for SSLAM

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
      type: sslam
      feature_dim: 768
      pretrained: true
      freeze: false
      extra_params:
        target_length: 1024

  fusion_type: concat
  fusion_hidden_dim: 512
  mid_fusion_enabled: false

  dropout_rate: 0.3
  classifier_hidden_dims: [256]

train:
  epochs: 30
  learning_rate: 0.0001
  weight_decay: 0.01

  lr_scheduler: cosine
  warmup_epochs: 3
  step_size: 30
  gamma: 0.1

  optimizer: adamw
  momentum: 0.9

  label_smoothing: 0.1
  mixup_alpha: 0.0
  cutmix_alpha: 0.0

  early_stop:
    enabled: true
    patience: 10
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
  gpu_ids: [0]
  distributed: false
  fp16: false

  log_interval: 10
  save_interval: 10

  output_dir: "output/solo_audio_sslam"
  resume: ""

  dist_backend: "nccl"
  dist_url: "env://"

  tensorboard_enabled: false
  experiment_name: "solo_audio_sslam"
```

- [ ] **Step 2: Commit**

```bash
git add configs/solo_audio_sslam.yaml
git commit -m "config: add solo audio SSLAM training config"
```

---

### Task 6: Download checkpoint and start training

- [ ] **Step 1: Download pretrained weights**

```bash
# Option A: Google Drive (manual download)
# https://drive.google.com/drive/folders/1aA65-qQCHSCrkiDeLGUtn1PiEjJi5HS8
# Place at: models/modelzoo/SSLAM-main/checkpoints/SSLAM_pretrained.pt

# Option B: HuggingFace (if proxy on 7890 is working)
mkdir -p models/modelzoo/SSLAM-main/checkpoints/
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ta012/SSLAM_pretrain', 'SSLAM_pretrained.pt',
    local_dir='models/modelzoo/SSLAM-main/checkpoints/')
"
```

- [ ] **Step 2: Test model loading**

```bash
cd /home/pythoner/abiu/multimodal-framework && PYTHONPATH=/home/pythoner/abiu/multimodal-framework python3 -c "
from models.modelzoo.audio_sslam import SSLAMBackbone
import torch
device = torch.device('cuda')
model = SSLAMBackbone(freeze=False).to(device)
print(f'Loaded: {sum(p.numel() for p in model.parameters()):,} params')
x = torch.randn(2, 160000, device=device)  # 10s @ 16kHz
with torch.no_grad():
    feat = model(x)
print(f'Forward: {x.shape} -> {feat.shape}')
print('OK!')
"
```
Expected: Prints param count and feature shape [2, 768]

- [ ] **Step 3: Start training**

```bash
export http_proxy=http://127.0.0.1:8118 https_proxy=http://127.0.0.1:8118
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/pythoner/abiu/multimodal-framework python -m tools.train --config configs/solo_audio_sslam.yaml
```

---

### Task 7: Monitor and compare

- [ ] **Step 1: Compare AST vs SSLAM results**

After training completes, compare:
- AST (86M pretrained): Val Acc ~72% (overfitted)
- AST-small (7.7M from scratch): currently training
- SSLAM (86M pretrained): TBD
