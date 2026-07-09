"""
Ordinal (CORAL) loss, grade decoding, and shared v3 metrics.

Used by the deep fusion trainer; the metric helpers mirror those in
scripts/train_gbdt_fusion.py so deep vs GBDT numbers are directly comparable.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Losses (per-site, NaN-masked)
# ---------------------------------------------------------------------------

def coral_loss(ord_logits: torch.Tensor, cum_targets: torch.Tensor) -> torch.Tensor:
    """
    CORAL ordinal loss = masked BCE over the K-1 cumulative binary targets.

    ord_logits : (B, n_sites, N_CUM)
    cum_targets: (B, n_sites, N_CUM) with NaN where a site has no measurement.
    """
    mask = ~torch.isnan(cum_targets)
    if mask.sum() == 0:
        return ord_logits.sum() * 0.0
    tgt = torch.nan_to_num(cum_targets, nan=0.0)
    per = F.binary_cross_entropy_with_logits(ord_logits, tgt, reduction="none")
    return (per * mask).sum() / mask.sum().clamp(min=1)


def masked_mse(reg: torch.Tensor, z_diam: torch.Tensor) -> torch.Tensor:
    """Masked MSE on z-scored diameters. reg, z_diam: (B, n_sites)."""
    mask = ~torch.isnan(z_diam)
    if mask.sum() == 0:
        return reg.sum() * 0.0
    tgt = torch.nan_to_num(z_diam, nan=0.0)
    per = F.mse_loss(reg, tgt, reduction="none")
    return (per * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def cum_probs(ord_logits: np.ndarray) -> np.ndarray:
    """Sigmoid of cumulative logits -> P(y>=cut). (..., N_CUM)."""
    return 1.0 / (1.0 + np.exp(-ord_logits))


def grade_from_cum(probs: np.ndarray) -> np.ndarray:
    """Ordinal grade = number of cumulative probabilities > 0.5."""
    return (probs > 0.5).sum(axis=-1)


# ---------------------------------------------------------------------------
# Metrics (shared with GBDT baseline)
# ---------------------------------------------------------------------------

def binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    m = ~np.isnan(p) & ~np.isnan(y)
    yy, pp = y[m].astype(int), p[m]
    if len(np.unique(yy)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": int(m.sum()), "pos": int(yy.sum())}
    return {"auroc": float(roc_auc_score(yy, pp)),
            "auprc": float(average_precision_score(yy, pp)),
            "n": int(m.sum()), "pos": int(yy.sum())}


def reg_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score
    m = ~np.isnan(p) & ~np.isnan(y)
    if m.sum() < 2:
        return {"mae": float("nan"), "r2": float("nan"), "n": int(m.sum())}
    return {"mae": float(mean_absolute_error(y[m], p[m])),
            "r2": float(r2_score(y[m], p[m])), "n": int(m.sum())}


def ordinal_qwk(grade_true: np.ndarray, grade_hat: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    m = ~np.isnan(grade_true) & ~np.isnan(grade_hat)
    if m.sum() < 2 or len(np.unique(grade_true[m])) < 2:
        return float("nan")
    return float(cohen_kappa_score(grade_true[m].astype(int), grade_hat[m].astype(int),
                                   weights="quadratic"))
