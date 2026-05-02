"""Bidirectional Cross-Attention — Wave<->Audio temporal consensus fusion."""

import torch
import torch.nn as nn


class CrossAttnBlock(nn.Module):
    """Cross-attention: Q from first arg, KV from second arg, pre-norm, residual + MLP."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value):
        attn_out, _ = self.attn(
            self.norm_q(query),
            self.norm_kv(key_value),
            self.norm_kv(key_value),
            need_weights=False,
        )
        x = query + attn_out
        return x + self.mlp(self.norm_out(x))


class TransformerBlock(nn.Module):
    """Standard pre-norm self-attention + MLP."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention between Wave and Audio tokens.

    Wave queries Audio and Audio queries Wave in parallel, then the
    resulting bidirectional representations are concatenated and refined
    through shared transformer blocks to produce a temporal consensus.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        num_shared_layers: Number of shared transformer blocks after cross-attn concat.
        dropout: Dropout rate.
    """

    def __init__(self, dim=256, num_heads=8, num_shared_layers=2, dropout=0.1):
        super().__init__()
        self.dim = dim

        self.wave_to_audio = CrossAttnBlock(dim, num_heads, dropout=dropout)
        self.audio_to_wave = CrossAttnBlock(dim, num_heads, dropout=dropout)

        self.shared_transformers = nn.ModuleList(
            [TransformerBlock(dim, num_heads, dropout=dropout) for _ in range(num_shared_layers)]
        )

        self.null_wave = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.null_audio = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, wave_tokens, audio_tokens, wave_masked=False, audio_masked=False):
        """Forward pass.

        Args:
            wave_tokens: [B, T_w, D] wave modality tokens.
            audio_tokens: [B, T_a, D] audio modality tokens.
            wave_masked: Whether wave is masked (for future null-token fallback).
            audio_masked: Whether audio is masked (for future null-token fallback).

        Returns:
            Consensus tokens [B, T_w + T_a, D].
        """
        # Null-token fallback: replace masked modality with learnable null tokens
        if audio_masked:
            audio_tokens = self.null_audio.expand_as(audio_tokens)
        if wave_masked:
            wave_tokens = self.null_wave.expand_as(wave_tokens)

        w2a = self.wave_to_audio(query=wave_tokens, key_value=audio_tokens)
        a2w = self.audio_to_wave(query=audio_tokens, key_value=wave_tokens)

        physical = torch.cat([w2a, a2w], dim=1)

        for block in self.shared_transformers:
            physical = block(physical)

        return physical
