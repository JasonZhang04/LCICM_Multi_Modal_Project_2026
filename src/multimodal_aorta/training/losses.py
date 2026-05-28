"""
Loss functions for aortic diameter regression.

Primary loss: Huber (Smooth L1) with delta=0.5 cm.
  - Quadratic for errors < 0.5 cm (precision zone — typical measurement noise)
  - Linear for errors > 0.5 cm (robust to occasional large errors)

Masked loss: handles samples where one target label is NaN (patient had only
one aortic measurement). The loss is computed only over valid (non-NaN) entries.
"""

import torch
import torch.nn.functional as F


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.5,
) -> torch.Tensor:
    """
    Huber loss averaged over valid (non-NaN) target entries only.

    Args:
        pred:   (B, 2) — model predictions [root_cm, asc_cm]
        target: (B, 2) — ground truth; NaN where label is missing
        delta:  Huber transition point in cm (default 0.5 = 5 mm)

    Returns:
        scalar loss (mean over all valid entries across both targets and batch)
    """
    valid = ~torch.isnan(target)                          # (B, 2) bool mask
    n_valid = valid.float().sum().clamp(min=1.0)

    # Replace NaN with 0 before computing loss (masked out below)
    target_filled = target.nan_to_num(0.0)

    loss_per_entry = F.smooth_l1_loss(pred, target_filled, beta=delta, reduction="none")
    loss = (loss_per_entry * valid.float()).sum() / n_valid
    return loss


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.5,
    lambda_root: float = 1.0,
    lambda_asc: float = 1.0,
) -> torch.Tensor:
    """
    Combined loss over both targets with optional per-target weighting.

    Args:
        pred:         (B, 2)  — [root_pred, asc_pred]
        target:       (B, 2)  — [root_gt,   asc_gt]
        delta:        Huber delta (cm)
        lambda_root:  weight for aortic root loss
        lambda_asc:   weight for ascending aorta loss

    Returns:
        scalar combined loss
    """
    loss_root = masked_huber_loss(pred[:, 0:1], target[:, 0:1], delta)
    loss_asc  = masked_huber_loss(pred[:, 1:2], target[:, 1:2], delta)
    return lambda_root * loss_root + lambda_asc * loss_asc
