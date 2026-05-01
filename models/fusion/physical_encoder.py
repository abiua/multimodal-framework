"""Physical Dynamics Encoder — IMU + Audio 早期融合。

时序对齐 → 双向 cross-attention (IMU↔Audio) → 共享 transformer → 物理共识表征。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnBlock(nn.Module):
    """Q from one modality, K/V from another."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout),
        )

    def forward(self, query, key_value):
        attn_out, _ = self.attn(self.norm_q(query), self.norm_kv(key_value),
                                self.norm_kv(key_value), need_weights=False)
        x = query + attn_out
        return x + self.mlp(self.norm_out(x))


class TransformerBlock(nn.Module):
    """Self-attention transformer block."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))


class PhysicalDynamicsEncoder(nn.Module):
    """IMU + Audio 物理信号融合编码器。

    Input:  imu_tokens [B, T_imu, D], audio_tokens [B, T_aud, D]
    Output: physical_tokens [B, T_p, D] — max(T_imu, T_aud)
    """

    def __init__(self, dim=256, num_heads=8, num_cross_attn_layers=1,
                 num_shared_transformer_layers=2, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.imu_to_audio = nn.ModuleList([
            CrossAttnBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_cross_attn_layers)
        ])
        self.audio_to_imu = nn.ModuleList([
            CrossAttnBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_cross_attn_layers)
        ])
        self.shared_transformers = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_shared_transformer_layers)
        ])

    def forward(self, imu_tokens, audio_tokens):
        # Step 1: Temporal alignment — interpolate to max(T_imu, T_aud)
        T_imu, T_aud = imu_tokens.size(1), audio_tokens.size(1)
        T_p = max(T_imu, T_aud)
        if T_imu != T_p:
            imu_tokens = F.interpolate(imu_tokens.transpose(1, 2), size=T_p,
                                       mode='linear', align_corners=False).transpose(1, 2)
        if T_aud != T_p:
            audio_tokens = F.interpolate(audio_tokens.transpose(1, 2), size=T_p,
                                         mode='linear', align_corners=False).transpose(1, 2)

        # Step 2: Bidirectional cross-attention
        imu_enh, aud_enh = imu_tokens, audio_tokens
        for layer in self.imu_to_audio:
            imu_enh = layer(query=imu_tokens, key_value=audio_tokens)
        for layer in self.audio_to_imu:
            aud_enh = layer(query=audio_tokens, key_value=imu_tokens)

        # Sum original + cross-attended
        imu_combined = imu_tokens + imu_enh
        aud_combined = audio_tokens + aud_enh

        # Step 3: Concat + shared transformer
        physical = torch.cat([imu_combined, aud_combined], dim=1)  # [B, 2*T_p, D]
        for block in self.shared_transformers:
            physical = block(physical)

        # Split concat back: sum the two halves (consensus)
        half = physical.size(1) // 2
        return physical[:, :half, :] + physical[:, half:, :]
