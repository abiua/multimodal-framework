"""Adaptive Evidence Reasoning (AER) — Dempster-Shafer decision-level fusion.

Based on PMIN (arxiv 2506.14170v3), Section 3.2.3.
Treats each branch's classification output as a piece of evidence with
learnable weight w_m and reliability r_m, fusing them via the evidential
reasoning rule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveEvidenceReasoning(nn.Module):
    """Fuse M evidence logits into a single prediction.

    Each evidence (branch) has:
      - weight w_m: preference/importance for this evidence (learned, >0 via softplus)
      - reliability r_m: credibility of this evidence source (learned, in (0,1) via sigmoid)

    Args:
        num_evidences: M, number of evidence sources (e.g., 3 for phys/image/final)
        num_classes: N, number of target classes
        init_weight: initial value for raw_w_m (default 0.0 -> softplus -> ~0.693)
        init_reliability: initial value for raw_r_m (default 0.0 -> sigmoid -> 0.5)
    """

    def __init__(self, num_evidences=3, num_classes=3,
                 init_weight=0.0, init_reliability=0.0):
        super().__init__()
        self.num_evidences = num_evidences
        self.num_classes = num_classes

        self.raw_weight = nn.Parameter(torch.full((num_evidences,), init_weight))
        self.raw_reliability = nn.Parameter(torch.full((num_evidences,), init_reliability))

    def forward(self, logits_list):
        """Fuse multiple evidence logits into log-probabilities.

        Args:
            logits_list: list of M tensors, each [B, N]

        Returns:
            log_probs: [B, N] fused log-probabilities (for CE loss)
        """
        assert len(logits_list) == self.num_evidences, \
            f"Expected {self.num_evidences} evidence tensors, got {len(logits_list)}"

        w = F.softplus(self.raw_weight)
        r = torch.sigmoid(self.raw_reliability)

        crw = 1.0 / (1.0 + w - r + 1e-8)
        K = crw * (1.0 - r)

        probs = torch.stack([F.softmax(logits, dim=-1) for logits in logits_list], dim=0)
        alpha = (crw * w).view(-1, 1, 1) * probs

        K_b = K.view(-1, 1, 1)
        prod_with = torch.prod(K_b + alpha, dim=0)
        K_prod = torch.prod(K)

        sum_prod = prod_with.sum(dim=-1)
        L = 1.0 / (sum_prod - (self.num_classes - 1) * K_prod + 1e-8)

        L_b = L.view(-1, 1)
        numerator = L_b * (prod_with - K_prod)
        denominator = 1.0 - L_b * K_prod + 1e-8
        belief = numerator / denominator

        belief = torch.clamp(belief, min=1e-8)
        belief = belief / belief.sum(dim=-1, keepdim=True)

        return torch.log(belief)
