#!/usr/bin/env python3
"""
Step 3: Fine-tune PanEcho for aortic dilation classification.

Improvements for class imbalance:
1. WeightedRandomSampler — balanced batches (50/50 normal/dilated)
2. Focal Loss — focuses optimizer on hard examples
3. AUPRC + threshold tuning — proper metrics for imbalanced data
4. Study-level evaluation with both mean and max pooling
5. Gradual unfreezing with differential learning rates

Usage:
    python scripts/03_train.py --config configs/default.yaml
"""

import os
import sys
import argparse
import logging
import random
import time
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.jhu_dataset import JHUEchoDataset
from models.echo_classifier import AorticDilationClassifier
from utils.training_utils import (
    FocalLoss,
    EarlyStopping,
    compute_metrics,
    compute_metrics_at_threshold,
    aggregate_study_predictions,
    save_checkpoint,
    load_checkpoint,
    format_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ #
#  Data loading with WeightedRandomSampler                             #
# ------------------------------------------------------------------ #

def create_dataloaders_with_sampler(
    labels_csv: str,
    accession_index_path: str,
    config: dict,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
):
    """
    Create dataloaders with WeightedRandomSampler for balanced training.

    The sampler ensures each mini-batch has roughly 50/50 normal/dilated
    clips. This gives consistent gradient signal for the minority class
    from the very first step.

    Splitting is still done at the study level to prevent data leakage.
    """
    from sklearn.model_selection import train_test_split

    # Load labels for study-level splitting
    labels_df = pd.read_csv(labels_csv)
    labels_df["accession"] = labels_df["accession"].astype(str)
    labels_df = labels_df.dropna(subset=["label"])

    with open(accession_index_path) as f:
        acc_index = json.load(f)

    labels_df = labels_df[labels_df["accession"].isin(acc_index.keys())]

    accessions = labels_df["accession"].values
    labels = labels_df["label"].astype(int).values

    # Study-level stratified split
    acc_train_val, acc_test, lab_train_val, lab_test = train_test_split(
        accessions, labels, test_size=test_split, stratify=labels, random_state=seed
    )
    relative_val = val_split / (1 - test_split)
    acc_train, acc_val, _, _ = train_test_split(
        acc_train_val, lab_train_val, test_size=relative_val,
        stratify=lab_train_val, random_state=seed
    )

    acc_train_set = set(acc_train)
    acc_val_set = set(acc_val)
    acc_test_set = set(acc_test)

    logger.info(f"Study-level split — train: {len(acc_train)}, val: {len(acc_val)}, test: {len(acc_test)}")

    # Create datasets
    common_kwargs = dict(
        labels_csv=labels_csv,
        accession_index_path=accession_index_path,
        num_frames=config.get("num_frames", 16),
        frame_size=config.get("frame_size", 224),
        min_frames_per_dicom=config.get("min_frames_per_dicom", 4),
        max_dicoms_per_study=config.get("max_dicoms_per_study", 10),
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std"),
    )

    train_ds = JHUEchoDataset(**common_kwargs, augment=True)
    val_ds = JHUEchoDataset(**common_kwargs, augment=False)
    test_ds = JHUEchoDataset(**common_kwargs, augment=False)

    # Split by study
    train_indices = [i for i, row in train_ds.samples.iterrows() if row["accession"] in acc_train_set]
    val_indices = [i for i, row in val_ds.samples.iterrows() if row["accession"] in acc_val_set]
    test_indices = [i for i, row in test_ds.samples.iterrows() if row["accession"] in acc_test_set]

    # --- WeightedRandomSampler for training ---
    # Assign weight to each training sample: minority class gets higher weight
    # This ensures roughly balanced batches during training
    train_labels = train_ds.samples.iloc[train_indices]["label"].values
    class_counts = np.bincount(train_labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels.astype(int)]
    sample_weights = torch.from_numpy(sample_weights).float()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    n_pos_train = (train_labels == 1).sum()
    n_neg_train = (train_labels == 0).sum()
    logger.info(
        f"Training clips — total: {len(train_indices)}, "
        f"normal: {n_neg_train}, dilated: {n_pos_train}, "
        f"ratio: {n_neg_train/max(n_pos_train,1):.1f}:1"
    )
    logger.info(
        f"WeightedRandomSampler active — each batch will be ~50/50 balanced"
    )
    logger.info(f"Val clips: {len(val_indices)}, Test clips: {len(test_indices)}")

    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 4)

    train_loader = DataLoader(
        Subset(train_ds, train_indices),
        batch_size=batch_size,
        sampler=sampler,          # balanced sampling instead of shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(test_ds, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------------ #
#  Training and evaluation loops                                       #
# ------------------------------------------------------------------ #

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch with balanced batches."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
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

        total_loss += loss.item() * video.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate_study_level(model, loader, criterion, device):
    """
    Evaluate at study level with both mean and max pooling.

    Reports:
    - Clip-level metrics (raw per-clip performance)
    - Study-level metrics with MEAN pooling (average clip probabilities)
    - Study-level metrics with MAX pooling (take highest dilated probability)
    - AUPRC and optimal threshold for both aggregation methods
    """
    model.eval()
    total_loss = 0.0

    all_probs = []
    all_labels = []
    all_accessions = []

    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["label"].to(device)
        accessions = batch["accession"]

        logits = model(video)
        loss = criterion(logits, labels)

        total_loss += loss.item() * video.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        all_accessions.extend(accessions)

    all_probs = np.array(all_probs)
    all_labels_arr = np.array(all_labels)
    n_clips = len(all_labels)

    # --- Study-level aggregation: MEAN ---
    y_true_mean, y_pred_mean, y_prob_mean = aggregate_study_predictions(
        all_accessions, all_probs, all_labels, method="mean"
    )
    metrics_mean = compute_metrics(y_true_mean, y_pred_mean, y_prob_mean)

    # --- Study-level aggregation: MAX ---
    y_true_max, y_pred_max, y_prob_max = aggregate_study_predictions(
        all_accessions, all_probs, all_labels, method="max"
    )
    metrics_max = compute_metrics(y_true_max, y_pred_max, y_prob_max)

    # --- Metrics at optimal threshold (mean pooling) ---
    if "optimal_threshold" in metrics_mean:
        metrics_at_opt = compute_metrics_at_threshold(
            y_true_mean, y_prob_mean, metrics_mean["optimal_threshold"]
        )
    else:
        metrics_at_opt = {}

    avg_loss = total_loss / max(n_clips, 1)

    # Combine into single output
    result = {
        "loss": avg_loss,
        "n_studies": len(set(all_accessions)),
        "n_clips": n_clips,
        # Mean pooling metrics (primary)
        "f1": metrics_mean.get("f1", 0),
        "auc": metrics_mean.get("auc", 0),
        "auprc": metrics_mean.get("auprc", 0),
        "accuracy": metrics_mean.get("accuracy", 0),
        "precision": metrics_mean.get("precision", 0),
        "recall": metrics_mean.get("recall", 0),
        "optimal_threshold": metrics_mean.get("optimal_threshold", 0.5),
        "f1_at_optimal": metrics_mean.get("f1_at_optimal", 0),
        # Max pooling metrics
        "max_f1": metrics_max.get("f1", 0),
        "max_auc": metrics_max.get("auc", 0),
        "max_auprc": metrics_max.get("auprc", 0),
        "max_recall": metrics_max.get("recall", 0),
    }

    return result


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train aortic dilation classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    model_cfg = config["model"]
    output_cfg = config["output"]
    data_cfg = config["data"]

    model_cfg["label_mode"] = config["labels"]["mode"]
    set_seed(train_cfg["seed"])

    # backbone
    model_cfg["backbone_lr"] = config["training"]["backbone_lr"] # Pass LR from training section

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Data with WeightedRandomSampler ---
    labels_csv = data_cfg["labels_cache"]
    accession_index_path = data_cfg["accession_index"]

    if not os.path.exists(labels_csv):
        logger.error(f"Labels CSV not found: {labels_csv}")
        sys.exit(1)
    if not os.path.exists(accession_index_path):
        logger.error(f"Accession index not found: {accession_index_path}")
        sys.exit(1)

    video_cfg = {
        **config["video"],
        "batch_size": train_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
    }

    train_loader, val_loader, test_loader = create_dataloaders_with_sampler(
        labels_csv=labels_csv,
        accession_index_path=accession_index_path,
        config=video_cfg,
        val_split=train_cfg["val_split"],
        test_split=train_cfg["test_split"],
        seed=train_cfg["seed"],
    )

    # --- Model ---
    logger.info("Building model...")
    model = AorticDilationClassifier(model_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # --- Optimizer with differential LR ---
    param_groups = model.get_param_groups(train_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg["weight_decay"])

    # --- Focal Loss ---
    # Class weights for alpha parameter
    labels_df = pd.read_csv(labels_csv)
    label_counts = labels_df["label"].dropna().astype(int).value_counts().sort_index()
    total = label_counts.sum()
    alpha = [total / (len(label_counts) * count) for count in label_counts.values]
    logger.info(f"Focal Loss alpha (class weights): {alpha}")
    logger.info(f"Focal Loss gamma: 2.0")

    criterion = FocalLoss(alpha=alpha, gamma=2.0).to(device)

    # --- Scheduler ---
    epochs = train_cfg["epochs"]
    warmup = train_cfg.get("warmup_epochs", 2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-7)

    # --- Mixed precision ---
    scaler = GradScaler() if train_cfg.get("mixed_precision") and device.type == "cuda" else None

    # --- Early stopping on AUPRC (better for imbalanced data) ---
    early_stopping = EarlyStopping(
        patience=train_cfg["early_stopping_patience"],
        mode="max",
    )

    # --- Resume ---
    start_epoch = 0
    best_val_auprc = 0.0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)

    os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(output_cfg["log_dir"], exist_ok=True)

    # --- Training loop ---
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING START")
    logger.info("=" * 70)
    logger.info(f"Loss: FocalLoss(gamma=2.0) | Sampler: WeightedRandomSampler")
    logger.info(f"Backbone frozen: {model_cfg.get('freeze_backbone', True)}")

    training_history = []

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Maybe unfreeze backbone
        model.maybe_unfreeze(epoch)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validate (study-level with both mean and max pooling)
        val_metrics = evaluate_study_level(model, val_loader, criterion, device)

        # Scheduler
        if scheduler is not None and epoch >= warmup:
            scheduler.step()

        # Logging
        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[-1]["lr"]
        logger.info(
            f"Epoch {epoch+1}/{epochs} ({elapsed:.0f}s) | LR: {lr:.2e}\n"
            f"  Train — loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.4f}, "
            f"f1: {train_metrics['f1']:.4f}\n"
            f"  Val(mean) — F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}, "
            f"AUPRC: {val_metrics['auprc']:.4f}, "
            f"thresh: {val_metrics['optimal_threshold']:.3f}, "
            f"F1@opt: {val_metrics['f1_at_optimal']:.4f}\n"
            f"  Val(max)  — F1: {val_metrics['max_f1']:.4f}, AUC: {val_metrics['max_auc']:.4f}, "
            f"AUPRC: {val_metrics['max_auprc']:.4f}, "
            f"recall: {val_metrics['max_recall']:.4f}\n"
            f"  ({val_metrics['n_studies']} studies, {val_metrics['n_clips']} clips)"
        )

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1_at_optimal": val_metrics["f1_at_optimal"],
            "val_max_f1": val_metrics["max_f1"],
            "val_max_auprc": val_metrics["max_auprc"],
            "lr": lr,
        })

        # Save best model (based on AUPRC — better for imbalanced data)
        val_auprc = val_metrics.get("auprc", 0.0)
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_path = os.path.join(output_cfg["checkpoint_dir"], "best_model.pt")
            save_checkpoint(model, optimizer, epoch, val_metrics, best_path)
            logger.info(f"  >> New best model (AUPRC={val_auprc:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(output_cfg["checkpoint_dir"], f"epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, val_metrics, ckpt_path)

        # Early stopping on AUPRC
        if early_stopping(val_auprc):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # --- Final test evaluation ---
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 70)

    best_path = os.path.join(output_cfg["checkpoint_dir"], "best_model.pt")
    if os.path.exists(best_path):
        load_checkpoint(model, None, best_path, device)

    test_metrics = evaluate_study_level(model, test_loader, criterion, device)

    logger.info(f"Test (mean pooling):")
    logger.info(f"  Accuracy:          {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision:         {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:            {test_metrics['recall']:.4f}")
    logger.info(f"  F1:                {test_metrics['f1']:.4f}")
    logger.info(f"  AUC:               {test_metrics['auc']:.4f}")
    logger.info(f"  AUPRC:             {test_metrics['auprc']:.4f}")
    logger.info(f"  Optimal threshold: {test_metrics['optimal_threshold']:.3f}")
    logger.info(f"  F1 @ optimal:      {test_metrics['f1_at_optimal']:.4f}")

    logger.info(f"\nTest (max pooling):")
    logger.info(f"  F1:                {test_metrics['max_f1']:.4f}")
    logger.info(f"  AUC:               {test_metrics['max_auc']:.4f}")
    logger.info(f"  AUPRC:             {test_metrics['max_auprc']:.4f}")
    logger.info(f"  Recall:            {test_metrics['max_recall']:.4f}")

    logger.info(f"\n  Studies: {test_metrics['n_studies']}, Clips: {test_metrics['n_clips']}")

    # Save results
    results = {
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "best_val_auprc": float(best_val_auprc),
        "training_history": training_history,
        "config": config,
    }
    results_path = os.path.join(output_cfg["log_dir"], "final_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    history_path = os.path.join(output_cfg["log_dir"], "training_history.csv")
    pd.DataFrame(training_history).to_csv(history_path, index=False)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"History saved to {history_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()