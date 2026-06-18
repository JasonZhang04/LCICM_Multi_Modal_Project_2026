"""
Loss functions for aortic diameter regression.

Primary loss: masked MSE on z-scored targets.

Why MSE on z-scored targets (Step 1 redesign):
  - The label std is ~0.5 cm and the v1 Huber delta was also 0.5 cm, which put
    ~84% of errors in the quadratic regime — Huber bought nothing over MSE.
  - Raw-cm targets (mean ~3.2) blew up the regression head's gradient relative
    to encoder activations; the head dominated early training.
  - Z-scoring puts targets on the same scale as the rest of the network, and
    plain MSE keeps the gradient interpretation clean.

NaN handling: patients with only one aortic measurement have one NaN target
entry. The loss is computed only over valid (non-NaN) entries.
"""

import torch
import torch.nn.functional as F


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error averaged over valid (non-NaN) target entries only.

    Args:
        pred:   (B, K) — model predictions (expected on the same scale as target)
        target: (B, K) — ground truth; NaN where label is missing

    Returns:
        scalar loss (mean over all valid entries across both targets and batch)
    """
    valid = ~torch.isnan(target)
    n_valid = valid.float().sum().clamp(min=1.0)

    target_filled = target.nan_to_num(0.0)
    loss_per_entry = F.mse_loss(pred, target_filled, reduction="none")
    return (loss_per_entry * valid.float()).sum() / n_valid


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_root: float = 1.0,
    lambda_asc: float = 1.0,
) -> torch.Tensor:
    """
    Combined MSE loss over both targets with optional per-target weighting.

    Both `pred` and `target` should be in the SAME scale (z-scored in v3+).

    Args:
        pred:         (B, 2)  — [root_pred, asc_pred]
        target:       (B, 2)  — [root_gt,   asc_gt]   (z-scored if stats provided)
        lambda_root:  weight for aortic root loss
        lambda_asc:   weight for ascending aorta loss

    Returns:
        scalar combined loss
    """
    loss_root = masked_mse_loss(pred[:, 0:1], target[:, 0:1])
    loss_asc  = masked_mse_loss(pred[:, 1:2], target[:, 1:2])
    return lambda_root * loss_root + lambda_asc * loss_asc
