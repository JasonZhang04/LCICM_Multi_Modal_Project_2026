"""
Regression head: maps the fused 768-dim representation to 2 diameter predictions.

Output is raw (no sigmoid/tanh) — predictions are in cm and compared directly
to the echo-derived labels. The masked Huber loss handles NaN labels.
"""

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    Input:  (B, in_dim)   — fused CLS token from FusionTransformer
    Output: (B, 2)        — [aortic_root_cm, ascending_aorta_cm], unrestricted
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 2)
