"""
v3 deep fusion model — frozen-feature early fusion with ordinal + regression heads.

All three modalities arrive as frozen precomputed vectors:
    ECG  : PCLR embedding      (320)
    CXR  : RAD-DINO embedding  (768)
    EHR  : tabular features    (F, imputed/standardized upstream)

Each is projected/encoded to a common width, concatenated, passed through a small
trunk MLP, then read out per anatomical site (root, ascending) by:
    - a CORAL ordinal head  -> K-1 cumulative logits  [>=4.0, >=4.5, >=5.0]
    - a linear regression head -> z-scored diameter

Design for n~522: deliberately small + heavily regularized (low width, dropout,
LayerNorm). Only these few-hundred-k params train; the backbones stay frozen.

CORAL (Cao et al. 2020): one shared weight vector produces a single logit per
site; K-1 learnable biases shift it into rank-consistent cumulative
probabilities P(y>k) = sigmoid(logit + b_k).
"""

import torch
import torch.nn as nn

from multimodal_aorta.data.targets import AD_CUTS

N_CUM = len(AD_CUTS)   # 3 cumulative thresholds


class EHRMLPEncoder(nn.Module):
    """Tiny MLP encoder for the tabular EHR vector (trained from scratch)."""

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoralHead(nn.Module):
    """
    CORAL ordinal head for `n_sites` targets.

    Produces (B, n_sites, N_CUM) cumulative logits: a shared per-site logit plus
    N_CUM learnable biases. Rank-consistent ordinal probabilities via sigmoid.
    """

    def __init__(self, in_dim: int, n_sites: int = 2):
        super().__init__()
        self.logit = nn.Linear(in_dim, n_sites, bias=False)
        # biases initialized descending so P(>=4.0) > P(>=4.5) > P(>=5.0) a priori
        self.bias = nn.Parameter(torch.tensor([2.0, 0.0, -2.0]).repeat(n_sites, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        base = self.logit(h).unsqueeze(-1)          # (B, n_sites, 1)
        return base + self.bias.unsqueeze(0)        # (B, n_sites, N_CUM)


class V3FusionModel(nn.Module):
    """
    Early-fusion deep model over frozen [ECG | CXR | EHR] features.

    forward(ecg, cxr, ehr) -> (ord_logits, reg)
        ord_logits : (B, n_sites, N_CUM)  cumulative logits
        reg        : (B, n_sites)         z-scored diameter predictions
    """

    def __init__(
        self,
        ecg_dim: int = 320,
        cxr_dim: int = 768,
        ehr_dim: int = 12,
        proj_dim: int = 128,
        ehr_out: int = 64,
        trunk_hidden: int = 128,
        dropout: float = 0.3,
        n_sites: int = 2,
    ):
        super().__init__()
        self.ecg_proj = nn.Linear(ecg_dim, proj_dim)
        self.cxr_proj = nn.Linear(cxr_dim, proj_dim)
        self.ehr_enc = EHRMLPEncoder(ehr_dim, hidden=ehr_out, out_dim=ehr_out, dropout=dropout)

        fused_dim = proj_dim * 2 + ehr_out
        self.trunk = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, trunk_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ord_head = CoralHead(trunk_hidden, n_sites=n_sites)
        self.reg_head = nn.Linear(trunk_hidden, n_sites)
        self.act = nn.GELU()

    def forward(self, ecg: torch.Tensor, cxr: torch.Tensor, ehr: torch.Tensor):
        e = self.act(self.ecg_proj(ecg))
        c = self.act(self.cxr_proj(cxr))
        h = self.ehr_enc(ehr)
        z = self.trunk(torch.cat([e, c, h], dim=-1))
        return self.ord_head(z), self.reg_head(z)
