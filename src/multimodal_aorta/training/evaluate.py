"""
Evaluation logic for aortic diameter regression.

Metrics computed per target (root, ascending):
  - MAE  (mean absolute error, cm)
  - RMSE (root mean squared error, cm)
  - R²   (coefficient of determination)
  - Pearson correlation coefficient

Classification-derived (threshold applied post-hoc, no separate training):
  - AUROC for aortic dilation (ground truth: diameter >= 4.0 cm)

Per-modality ablation: call evaluate() three times with different has_ecg/has_cxr
overrides (both, ECG-only, CXR-only) to measure each modality's contribution.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from multimodal_aorta.data.dataset import TargetStats
from multimodal_aorta.training.losses import total_loss

# Dilation threshold (cm) — used for AUROC classification
DILATION_THRESHOLD_CM = 4.0


@dataclass
class EvalMetrics:
    """All metrics for one evaluation pass."""
    # Per-target regression
    mae_root: float = float("nan")
    mae_asc:  float = float("nan")
    rmse_root: float = float("nan")
    rmse_asc:  float = float("nan")
    r2_root:   float = float("nan")
    r2_asc:    float = float("nan")
    pearson_root: float = float("nan")
    pearson_asc:  float = float("nan")

    # Classification-derived
    auroc_root: float = float("nan")
    auroc_asc:  float = float("nan")

    # Loss
    val_loss: float = float("nan")

    # Overall summary (used for early stopping and logging)
    @property
    def total_mae(self) -> float:
        """Mean of the two per-target MAEs (ignores NaN targets)."""
        vals = [v for v in [self.mae_root, self.mae_asc] if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def to_dict(self) -> Dict[str, float]:
        return {
            "val_loss": self.val_loss,
            "mae_root": self.mae_root,
            "mae_asc":  self.mae_asc,
            "rmse_root": self.rmse_root,
            "rmse_asc":  self.rmse_asc,
            "r2_root":   self.r2_root,
            "r2_asc":    self.r2_asc,
            "pearson_root": self.pearson_root,
            "pearson_asc":  self.pearson_asc,
            "auroc_root": self.auroc_root,
            "auroc_asc":  self.auroc_asc,
            "total_mae":  self.total_mae,
        }


def _mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def _r2(pred: np.ndarray, gt: np.ndarray) -> float:
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - gt.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _pearson(pred: np.ndarray, gt: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    return float(np.corrcoef(pred, gt)[0, 1])


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via Mann-Whitney U (no sklearn dependency required)."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def _compute_target_metrics(
    preds: np.ndarray,
    gts: np.ndarray,
    threshold: float = DILATION_THRESHOLD_CM,
) -> Dict[str, float]:
    """Compute all metrics for a single target given matched pred/gt arrays."""
    valid = ~np.isnan(gts)
    if valid.sum() < 2:
        return {k: float("nan") for k in ["mae", "rmse", "r2", "pearson", "auroc"]}

    p, g = preds[valid], gts[valid]
    binary_gt = (g >= threshold).astype(int)

    return {
        "mae":     _mae(p, g),
        "rmse":    _rmse(p, g),
        "r2":      _r2(p, g),
        "pearson": _pearson(p, g),
        "auroc":   _auroc(p, binary_gt),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_stats: Optional[TargetStats] = None,
    force_ecg_only: bool = False,
    force_cxr_only: bool = False,
    restrict_to_both: bool = False,
) -> EvalMetrics:
    """
    Run one full evaluation pass over `loader`.

    The loader yields batches whose `target` tensor is in the SAME scale the
    model was trained on (z-scored when `target_stats` is provided). The loss
    reported here is therefore in that same scale (z-score MSE) for direct
    comparison with `train_loss`. Predictions and targets are de-normalized to
    cm before MAE / RMSE / R² / Pearson / AUROC are computed, so reported
    regression metrics are always in human-readable cm.

    Args:
        model:          AortaModel in eval mode
        loader:         DataLoader yielding collated batches
        device:         torch device
        target_stats:   if provided, used to de-normalize predictions/targets
                        back to cm before metric computation
        force_ecg_only: override has_cxr=False for all samples (ablation)
        force_cxr_only: override has_ecg=False for all samples (ablation)
        restrict_to_both: compute metrics ONLY over samples that genuinely have
                        BOTH modalities present (before any ablation override).
                        This is required to measure each modality's marginal
                        contribution honestly: on a cohort dominated by
                        ECG-only patients, a whole-set 'cxr_only' pass mostly
                        measures the model's mean-prediction (both tokens
                        masked), not what CXR can actually do. See module docstring.

    Returns:
        EvalMetrics dataclass with all computed metrics
    """
    model.eval()

    all_preds   = []  # list of (B, 2) tensors
    all_targets = []  # list of (B, 2) tensors
    all_both    = []  # list of (B,) bool tensors — sample genuinely has both modalities

    for batch in loader:
        ecg     = batch["ecg"].to(device)
        cxr     = batch["cxr"].to(device)
        target  = batch["target"].to(device)
        has_ecg = batch["has_ecg"].to(device)
        has_cxr = batch["has_cxr"].to(device)

        # Record TRUE modality availability before any ablation override.
        both = (has_ecg & has_cxr).cpu()

        # Ablation overrides
        if force_ecg_only:
            has_cxr = torch.zeros_like(has_cxr)
        if force_cxr_only:
            has_ecg = torch.zeros_like(has_ecg)

        pred = model(ecg, cxr, has_ecg, has_cxr)

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())
        all_both.append(both)

    preds_t   = torch.cat(all_preds,   dim=0)         # (N, 2)
    targets_t = torch.cat(all_targets, dim=0)         # (N, 2)
    both_mask = torch.cat(all_both,    dim=0)         # (N,)

    if restrict_to_both:
        preds_t   = preds_t[both_mask]
        targets_t = targets_t[both_mask]

    # Loss is reported in the SAME (z-scored) scale as train_loss, computed over
    # whatever subset we are evaluating, for direct comparability.
    if preds_t.shape[0] > 0:
        val_loss = float(total_loss(preds_t, targets_t).item())
    else:
        val_loss = float("nan")

    # De-normalize back to cm for human-readable metrics. NaN targets stay NaN.
    if target_stats is not None:
        preds_t   = target_stats.denormalize(preds_t)
        targets_t = target_stats.denormalize(targets_t)

    preds   = preds_t.numpy()
    targets = targets_t.numpy()

    root_m = _compute_target_metrics(preds[:, 0], targets[:, 0])
    asc_m  = _compute_target_metrics(preds[:, 1], targets[:, 1])

    return EvalMetrics(
        val_loss    = val_loss,
        mae_root    = root_m["mae"],
        mae_asc     = asc_m["mae"],
        rmse_root   = root_m["rmse"],
        rmse_asc    = asc_m["rmse"],
        r2_root     = root_m["r2"],
        r2_asc      = asc_m["r2"],
        pearson_root = root_m["pearson"],
        pearson_asc  = asc_m["pearson"],
        auroc_root  = root_m["auroc"],
        auroc_asc   = asc_m["auroc"],
    )


def evaluate_ablation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_stats: Optional[TargetStats] = None,
    restrict_to_both: bool = False,
) -> Dict[str, EvalMetrics]:
    """
    Run three evaluation passes for per-modality ablation:
      'both'     — both modalities (full model)
      'ecg_only' — ECG only (CXR replaced with mask token)
      'cxr_only' — CXR only (ECG replaced with mask token)

    When `restrict_to_both=True`, all three passes are scored ONLY on the
    subset of patients that genuinely have both modalities, so the numbers are
    directly comparable and 'cxr_only' actually measures CXR's standalone
    signal (rather than mean-prediction on ECG-only patients).

    Returns a dict keyed by ablation name.
    """
    return {
        "both":     evaluate(model, loader, device, target_stats=target_stats,
                             restrict_to_both=restrict_to_both),
        "ecg_only": evaluate(model, loader, device, target_stats=target_stats,
                             force_ecg_only=True, restrict_to_both=restrict_to_both),
        "cxr_only": evaluate(model, loader, device, target_stats=target_stats,
                             force_cxr_only=True, restrict_to_both=restrict_to_both),
    }


@torch.no_grad()
def evaluate_mean_baseline(
    loader: DataLoader,
    target_stats: Optional[TargetStats] = None,
    restrict_to_both: bool = False,
) -> EvalMetrics:
    """
    Constant-predictor reference: always predict the train-split mean diameter.

    This is the honest "do-nothing" baseline any model must beat. Because the
    model trains on z-scored targets, the train mean corresponds to a prediction
    of 0 in z-scored space (de-normalized back to root_mean / asc_mean in cm).
    R² of this predictor is ~0 by construction; its MAE is the bar to clear.

    No model is needed — we only read targets from the loader.
    """
    all_targets = []
    all_both    = []

    for batch in loader:
        target  = batch["target"]
        has_ecg = batch["has_ecg"]
        has_cxr = batch["has_cxr"]
        all_targets.append(target)
        all_both.append((has_ecg & has_cxr))

    targets_t = torch.cat(all_targets, dim=0)          # (N, 2), z-scored if stats given
    both_mask = torch.cat(all_both,    dim=0)          # (N,)

    if restrict_to_both:
        targets_t = targets_t[both_mask]

    # Predict the mean → 0 in z-scored space (train mean in cm after de-norm).
    preds_t = torch.zeros_like(targets_t)

    if target_stats is not None:
        preds_t   = target_stats.denormalize(preds_t)
        targets_t = target_stats.denormalize(targets_t)

    preds   = preds_t.numpy()
    targets = targets_t.numpy()

    root_m = _compute_target_metrics(preds[:, 0], targets[:, 0])
    asc_m  = _compute_target_metrics(preds[:, 1], targets[:, 1])

    return EvalMetrics(
        mae_root    = root_m["mae"],
        mae_asc     = asc_m["mae"],
        rmse_root   = root_m["rmse"],
        rmse_asc    = asc_m["rmse"],
        r2_root     = root_m["r2"],
        r2_asc      = asc_m["r2"],
        pearson_root = root_m["pearson"],
        pearson_asc  = asc_m["pearson"],
        auroc_root  = root_m["auroc"],
        auroc_asc   = asc_m["auroc"],
    )
