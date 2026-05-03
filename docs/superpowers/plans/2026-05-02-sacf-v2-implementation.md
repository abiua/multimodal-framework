# SACF v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement SACF v2 pipeline — Wave+Audio bidirectional cross-attention fusion with Image FiLM modulation, gated residual, consensus loss, and modal dropout. Train on 4 GPUs then run gradient + ablation analysis.

**Architecture:** 3-stage: (1) Temporal Consensus — Wave↔Audio cross-attn, (2) Image FiLM + Gated Residual, (3) Consensus KL(stopgrad(final) || single-modal) + 3 classifiers.

**Tech Stack:** PyTorch, DDP (4 GPU), NCCL, AMP, OmegaConf

---

## File Structure

| File | Role |
|------|------|
| `models/modelzoo/audio_temporal.py` | NEW — AudioTemporalEncoder preserves time dim |
| `models/fusion/temporal_consensus.py` | NEW — BidirectionalCrossAttention |
| `models/fusion/film_gate.py` | NEW — FiLMGate with gated residual |
| `models/pipeline_sacf.py` | NEW — SACFPipeline orchestrates 3 stages |
| `models/fusion/__init__.py` | MODIFY — Export new modules |
| `configs/sacf_fish_feeding.yaml` | NEW — Training config |
| `scripts/train_sacf.py` | NEW — DDP training with consensus loss |
| `scripts/evaluate_sacf.py` | NEW — Gradient analysis + ablation |
| `tests/test_sacf.py` | NEW — All tests |

---
