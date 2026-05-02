"""FiLM modulation + Gated Residual — Image conditions physical features."""
import torch
import torch.nn as nn


class FiLMGate(nn.Module):
    """Image provides FiLM modulation (scale/shift) + small gated residual.

    f_final = (scale * phys + shift) + sigmoid(gate) * r_img

    Args:
        image_dim: Image backbone output dimension (e.g., 2048 for ResNet50)
        phys_dim: Physical feature dimension (e.g., 256)
        residual_dim: SideNet bottleneck dimension (must be < phys_dim)
        r_dropout: Stochastic dropout rate for residual path (training only)
    """

    def __init__(self, image_dim=2048, phys_dim=256, residual_dim=64, r_dropout=0.3):
        super().__init__()
        assert residual_dim < phys_dim, "residual_dim must be < phys_dim to prevent shortcut"

        # FiLM modulation — initialized near identity
        self.scale_proj = nn.Sequential(nn.Linear(image_dim, phys_dim), nn.Tanh())
        nn.init.zeros_(self.scale_proj[0].weight)
        nn.init.zeros_(self.scale_proj[0].bias)

        self.shift_proj = nn.Sequential(nn.Linear(image_dim, phys_dim), nn.Tanh())
        nn.init.zeros_(self.shift_proj[0].weight)
        nn.init.zeros_(self.shift_proj[0].bias)

        # SideNet: image -> bottleneck -> small residual
        self.sidenet = nn.Sequential(
            nn.Linear(image_dim, image_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(image_dim // 4, residual_dim),
        )
        self.r_dropout = nn.Dropout(r_dropout)

        # Gate: learns when to trust Image residual
        self.gate_proj = nn.Sequential(
            nn.Linear(image_dim, image_dim // 4),
            nn.GELU(),
            nn.Linear(image_dim // 4, residual_dim),
        )

        # Expand residual from d->D
        self.residual_expand = nn.Linear(residual_dim, phys_dim)

    def forward(self, z_img, phys_pooled):
        # FiLM modulation
        scale = 1.0 + self.scale_proj(z_img)   # near 1
        shift = self.shift_proj(z_img)          # near 0
        phys_modulated = scale * phys_pooled + shift

        # Gated residual
        r_img = self.sidenet(z_img)
        r_img = self.r_dropout(r_img)
        gate = torch.sigmoid(self.gate_proj(z_img))
        residual = self.residual_expand(gate * r_img)

        f_final = phys_modulated + residual

        return f_final, {
            'film_scale': scale,
            'film_shift': shift,
            'gate': gate,
            'residual': r_img,
        }
