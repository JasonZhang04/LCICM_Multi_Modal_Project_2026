"""
Training utilities for aortic dilation classification.

Components:
- FocalLoss: handles class imbalance at the loss level
- train_one_epoch: clip-level training with balanced sampler
- evaluate_study_level: study-level eval with mean/max pooling
- Metrics: AUPRC (primary), AUC, F1, threshold analysis
- EarlyStopping, checkpointing
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
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Focal Loss                                                          #
# ------------------------------------------------------------------ #

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for class-imbalanced classification.

    Combines two mechanisms:
    1. Alpha weighting: scales loss per class (higher for minority)
    2. Focal modulation: (1 - p_t)^gamma downweights easy/confident
       predictions so the model focuses on hard examples

    L = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = predicted probability of the TRUE class.

    For gamma=0, this reduces to standard weighted cross-entropy.
    For gamma=2 (recommended), a sample the model predicts with 0.9
    confidence has its loss scaled by (1-0.9)^2 = 0.01, effectively
    removing it from gradient computation. This forces the model to
    focus on the uncertain/misclassified samples.

    Args:
        gamma: Focusing parameter. 0 = standard CE, 2 = typical.
        alpha: Class weight tensor of shape (num_classes,), or None.
               Higher values for minority class increase its importance.
        reduction: 'mean' (default), 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Register alpha as a buffer so it moves to GPU with the model
        # and doesn't get treated as a trainable parameter
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw unnormalized scores from model.
            targets: (B,) integer class labels in [0, C-1].

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or (B,) tensor.
        """
        # Compute standard cross-entropy per sample (no reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # p_t = probability assigned to the TRUE class
        # We compute it from the CE loss: p_t = exp(-CE_loss)
        # This is numerically equivalent to softmax + gather but
        # avoids a redundant softmax computation
        p_t = torch.exp(-ce_loss)

        # Focal modulating factor
        focal_weight = (1.0 - p_t) ** self.gamma

        # Apply focal modulation
        loss = focal_weight * ce_loss

        # Apply per-class alpha weighting
        if self.alpha is not None:
            # Gather the alpha for each sample's true class
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ------------------------------------------------------------------ #
#  Early Stopping                                                      #
# ------------------------------------------------------------------ #

class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        mode: 'max' if higher metric is better (e.g., AUPRC),
              'min' if lower is better (e.g., loss).
        min_delta: Minimum change to count as improvement.
    """

    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 1e-4):
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

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping: no improvement for {self.patience} epochs "
                    f"(best={self.best_score:.4f})"
                )

        return self.should_stop


# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: (N,) ground truth binary labels.
        y_pred: (N,) predicted binary labels (at some threshold).
        y_prob: (N,) predicted probability of class 1. Optional.

    Returns:
        Dict with accuracy, precision, recall, f1, and optionally
        auc and auprc.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc"] = 0.0
        try:
            # AUPRC is our primary metric for imbalanced data
            # A random classifier scores ~prevalence (e.g., 0.08)
            # so anything above that indicates learning
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["auprc"] = 0.0
    elif y_prob is not None:
        # Only one class present in this split — can't compute AUC
        metrics["auc"] = 0.0
        metrics["auprc"] = 0.0

    return metrics


def threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Show model performance at multiple probability thresholds.

    The default 0.5 threshold may be too high. The model might be
    assigning 0.2-0.4 probability to dilated cases, which is a
    real signal but doesn't cross 0.5.

    Returns:
        Dict mapping threshold string -> metrics dict.
    """
    if len(np.unique(y_true)) < 2:
        return {}

    results = {}
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y_pred = (y_prob >= threshold).astype(int)
        n_pred_pos = int(y_pred.sum())
        n_actual_pos = int(y_true.sum())

        results[f"t{threshold:.1f}"] = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "n_predicted_pos": n_pred_pos,
            "n_actual_pos": n_actual_pos,
        }
    return results


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """
    Train for one epoch at clip level.

    With WeightedRandomSampler, each batch has roughly 50/50 class
    balance. Combined with focal loss, the model gets a strong,
    balanced gradient signal from both classes.
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        video = batch["video"].to(device)    # (B, C, T, H, W)
        labels = batch["label"].to(device)   # (B,)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(video)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(video)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = video.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        probs = torch.softmax(logits.detach(), dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    avg_loss = total_loss / max(n_samples, 1)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = avg_loss

    return metrics


# ------------------------------------------------------------------ #
#  Study-level evaluation                                              #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_study_level(model, loader, criterion, device):
    """
    Evaluate at the STUDY level with two aggregation strategies.

    For each study, we collect the predicted P(dilated) from every
    clip, then aggregate:

    1. MEAN pooling: average P(dilated) across all clips.
       Represents the "overall impression" from the full study.

    2. MAX pooling: take the highest P(dilated) across all clips.
       Clinically motivated: aortic dilation is visible in specific
       views (mainly PLAX). If ANY clip shows high probability,
       the study should be flagged. This is more appropriate for
       a screening setting where sensitivity matters.

    We report full metrics for both strategies plus threshold analysis.
    """
    model.eval()
    total_loss = 0.0
    n_clips = 0

    # Collect per-clip probabilities grouped by study
    study_clip_probs = defaultdict(list)  # accession -> list of P(dilated)
    study_labels = {}                     # accession -> ground truth label

    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["label"].to(device)
        accessions = batch["accession"]

        logits = model(video)
        loss = criterion(logits, labels)

        total_loss += loss.item() * video.size(0)
        n_clips += video.size(0)

        # Get P(dilated) = softmax probability of class 1
        probs = torch.softmax(logits, dim=1)  # (B, 2)
        prob_dilated = probs[:, 1].cpu().numpy()  # (B,)

        for i in range(len(accessions)):
            acc = accessions[i]
            study_clip_probs[acc].append(prob_dilated[i])
            study_labels[acc] = labels[i].cpu().item()

    # --- Aggregate per study ---
    y_true = []
    y_prob_mean = []
    y_prob_max = []

    for acc in study_clip_probs:
        clip_probs = np.array(study_clip_probs[acc])  # (N_clips,)
        label = study_labels[acc]

        y_true.append(label)
        y_prob_mean.append(float(np.mean(clip_probs)))
        y_prob_max.append(float(np.max(clip_probs)))

    y_true = np.array(y_true, dtype=int)
    y_prob_mean = np.array(y_prob_mean)
    y_prob_max = np.array(y_prob_max)

    # --- Mean pooling metrics ---
    y_pred_mean = (y_prob_mean >= 0.5).astype(int)
    metrics_mean = compute_metrics(y_true, y_pred_mean, y_prob_mean)

    # --- Max pooling metrics ---
    y_pred_max = (y_prob_max >= 0.5).astype(int)
    metrics_max = compute_metrics(y_true, y_pred_max, y_prob_max)

    # --- Threshold analysis for both strategies ---
    thresh_mean = threshold_analysis(y_true, y_prob_mean)
    thresh_max = threshold_analysis(y_true, y_prob_max)

    # --- Find optimal threshold for mean pooling ---
    optimal_thresh, f1_at_opt = find_optimal_threshold(y_true, y_prob_mean)

    # --- Combine into result dict ---
    # Primary metrics use mean pooling
    result = {
        "loss": total_loss / max(n_clips, 1),
        "n_studies": len(study_clip_probs),
        "n_clips": n_clips,

        # Mean pooling metrics
        "accuracy": metrics_mean["accuracy"],
        "precision": metrics_mean["precision"],
        "recall": metrics_mean["recall"],
        "f1": metrics_mean["f1"],
        "auc": metrics_mean.get("auc", 0.0),
        "auprc": metrics_mean.get("auprc", 0.0),

        # Max pooling metrics (prefixed with max_)
        "max_accuracy": metrics_max["accuracy"],
        "max_precision": metrics_max["precision"],
        "max_recall": metrics_max["recall"],
        "max_f1": metrics_max["f1"],
        "max_auc": metrics_max.get("auc", 0.0),
        "max_auprc": metrics_max.get("auprc", 0.0),

        # Threshold analysis
        "threshold_mean": thresh_mean,
        "threshold_max": thresh_max,

        # Optimal threshold metrics
        "optimal_threshold": optimal_thresh,
        "f1_at_optimal": f1_at_opt,

        # Raw probabilities for debugging
        "prob_mean_stats": {
            "min": float(y_prob_mean.min()) if len(y_prob_mean) > 0 else 0,
            "max": float(y_prob_mean.max()) if len(y_prob_mean) > 0 else 0,
            "median": float(np.median(y_prob_mean)) if len(y_prob_mean) > 0 else 0,
            "mean": float(y_prob_mean.mean()) if len(y_prob_mean) > 0 else 0,
        },
    }

    return result


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
            "metrics": {
                k: v for k, v in metrics.items()
                if not isinstance(v, dict)  # skip nested dicts for clean serialization
            },
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
    """Format metrics dict into a readable log string."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{prefix}{k}: {v:.4f}")
        elif isinstance(v, (int, np.integer)):
            parts.append(f"{prefix}{k}: {v}")
    return " | ".join(parts)


# ------------------------------------------------------------------ #
#  Missing functions needed by 03_train.py                            #
# ------------------------------------------------------------------ #

def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute classification metrics at a specific probability threshold.

    Args:
        y_true: (N,) ground truth binary labels.
        y_prob: (N,) predicted probability of class 1.
        threshold: Decision threshold for converting probabilities to predictions.

    Returns:
        Dict with precision, recall, f1 at the given threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = None,
) -> tuple:
    """
    Find the probability threshold that maximizes F1 score.

    Args:
        y_true: (N,) ground truth binary labels.
        y_prob: (N,) predicted probability of class 1.
        thresholds: Array of thresholds to try. Defaults to 0.1 to 0.9.

    Returns:
        (best_threshold, best_f1)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    best_f1 = 0.0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return float(best_thresh), float(best_f1)


def aggregate_study_predictions(
    accessions: list,
    probs: np.ndarray,
    labels: list,
    method: str = "mean",
) -> tuple:
    """
    Aggregate clip-level predictions to study level.

    Args:
        accessions: List of accession IDs (one per clip).
        probs: (N, 2) array of class probabilities, or (N,) array of P(class=1).
        labels: List of ground truth labels (one per clip).
        method: "mean" or "max" pooling of P(dilated).

    Returns:
        (y_true, y_pred, y_prob) arrays at study level.
    """
    # Handle both (N, 2) and (N,) probability formats
    probs = np.array(probs)
    if probs.ndim == 2:
        prob_positive = probs[:, 1]  # P(dilated)
    else:
        prob_positive = probs

    # Group by accession
    study_probs = defaultdict(list)
    study_labels = {}

    for i, acc in enumerate(accessions):
        study_probs[acc].append(prob_positive[i])
        study_labels[acc] = int(labels[i])

    # Aggregate
    y_true = []
    y_prob = []

    for acc in study_probs:
        clip_probs = np.array(study_probs[acc])

        if method == "max":
            agg_prob = float(np.max(clip_probs))
        else:  # mean
            agg_prob = float(np.mean(clip_probs))

        y_true.append(study_labels[acc])
        y_prob.append(agg_prob)

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    return y_true, y_pred, y_prob