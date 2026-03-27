"""
Training utilities: focal loss, metrics (AUPRC, threshold tuning),
early stopping, checkpointing, study-level aggregation.
"""

import os
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Focal Loss                                                          #
# ------------------------------------------------------------------ #

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    Standard cross-entropy treats all samples equally. With 10:1 class
    imbalance, the model gets very confident on the majority class, and
    those easy samples dominate the loss. Focal loss adds a modulating
    factor (1 - p_t)^gamma that down-weights easy samples and focuses
    the optimizer on hard, misclassified examples.

    L_FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weight tensor of shape (num_classes,). Can be None.
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 reduces to standard cross-entropy.
               gamma=2.0 is a good default for medical imaging.
        reduction: 'mean' or 'sum' or 'none'.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw model output (before softmax).
            targets: (B,) integer class labels.

        Returns:
            Scalar loss.
        """
        probs = F.softmax(logits, dim=1)  # (B, C)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()  # (B, C)

        # p_t = probability of the true class
        p_t = (probs * targets_one_hot).sum(dim=1)  # (B,)

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma  # (B,)

        # Standard cross-entropy per sample
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (B,)

        # Apply focal weight
        loss = focal_weight * ce_loss  # (B,)

        # Apply class-specific alpha weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # (B,)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Compute classification metrics including AUPRC and optimal threshold.

    For imbalanced medical data, AUPRC is more informative than AUROC
    because it focuses on the model's ability to correctly identify
    positive cases without being inflated by the large number of easy
    negatives.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = 0.0

        try:
            # AUPRC — more informative for imbalanced data
            metrics["auprc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["auprc"] = 0.0

        try:
            # Find optimal threshold from precision-recall curve
            # Maximize F1 = 2 * (precision * recall) / (precision + recall)
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            metrics["optimal_threshold"] = float(thresholds[min(optimal_idx, len(thresholds) - 1)])
            metrics["f1_at_optimal"] = float(f1_scores[optimal_idx])
        except (ValueError, IndexError):
            metrics["optimal_threshold"] = 0.5
            metrics["f1_at_optimal"] = 0.0

    return metrics


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Compute metrics using a custom probability threshold instead of 0.5."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "threshold": threshold,
    }


# ------------------------------------------------------------------ #
#  Study-level aggregation                                             #
# ------------------------------------------------------------------ #

def aggregate_study_predictions(
    accessions: list,
    probs: np.ndarray,
    labels: list,
    method: str = "mean",
) -> tuple:
    """
    Aggregate clip-level predictions to study-level.

    Args:
        accessions: List of accession strings per clip.
        probs: Array of shape (N, 2) — softmax probabilities per clip.
        labels: List of integer labels per clip.
        method: "mean" (average probs) or "max" (take max positive prob).

    Returns:
        (study_y_true, study_y_pred, study_y_prob) — arrays at study level.
    """
    study_probs = defaultdict(list)
    study_labels = {}

    for i in range(len(accessions)):
        acc = accessions[i]
        study_probs[acc].append(probs[i])
        study_labels[acc] = labels[i]

    y_true, y_pred, y_prob = [], [], []
    for acc in study_probs:
        clip_probs = np.array(study_probs[acc])  # (num_clips, 2)

        if method == "mean":
            avg_prob = clip_probs.mean(axis=0)
        elif method == "max":
            # Max-pooling: take the clip with highest dilated probability
            # Clinical logic: if ANY view shows dilation, patient is dilated
            max_idx = clip_probs[:, 1].argmax()
            avg_prob = clip_probs[max_idx]
        else:
            avg_prob = clip_probs.mean(axis=0)

        pred = np.argmax(avg_prob)
        y_true.append(study_labels[acc])
        y_pred.append(pred)
        y_prob.append(avg_prob[1])  # probability of dilated

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# ------------------------------------------------------------------ #
#  Early stopping                                                      #
# ------------------------------------------------------------------ #

class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 7, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            (score < self.best_score - self.min_delta)
            if self.mode == "min"
            else (score > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered (patience={self.patience})")

        return self.should_stop


# ------------------------------------------------------------------ #
#  Checkpointing                                                       #
# ------------------------------------------------------------------ #

def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load a checkpoint. Returns the epoch number."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    logger.info(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics dict into a readable string."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{prefix}{k}: {v:.4f}")
        else:
            parts.append(f"{prefix}{k}: {v}")
    return " | ".join(parts)