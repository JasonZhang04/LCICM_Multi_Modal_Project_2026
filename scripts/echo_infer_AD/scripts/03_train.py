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

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.jhu_dataset import create_dataloaders
from models.echo_classifier import AorticDilationClassifier
from utils.training_utils import (
    FocalLoss,
    EarlyStopping,
    compute_metrics,
    compute_metrics_at_threshold,
    aggregate_study_predictions,
    train_one_epoch,
    evaluate_study_level,
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

    # Pass backbone LR from training section
    model_cfg["backbone_lr"] = train_cfg["backbone_lr"]

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

    # Build config for dataloaders
    video_cfg = {
        **config["video"],
        "batch_size": train_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
    }

    # Use the correct create_dataloaders from jhu_dataset.py
    train_loader, val_loader, test_loader = create_dataloaders(
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
    alpha = torch.tensor([total / (len(label_counts) * count) for count in label_counts.values])
    logger.info(f"Focal Loss alpha (class weights): {alpha.tolist()}")
    logger.info(f"Focal Loss gamma: {train_cfg.get('focal_gamma', 2.0)}")

    criterion = FocalLoss(
        alpha=alpha,
        gamma=train_cfg.get("focal_gamma", 2.0)
    ).to(device)

    # --- Scheduler ---
    epochs = train_cfg["epochs"]
    warmup = train_cfg.get("warmup_epochs", 2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-7)

    # --- Mixed precision ---
    scaler = GradScaler(device_type="cuda") if train_cfg.get("mixed_precision") and device.type == "cuda" else None

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
    logger.info(f"Loss: FocalLoss(gamma={train_cfg.get('focal_gamma', 2.0)}) | Sampler: WeightedRandomSampler")
    logger.info(f"Backbone frozen: {model_cfg.get('freeze_backbone', True)}")
    logger.info(f"Unfreeze after epoch: {model_cfg.get('unfreeze_after_epoch', 3)}")

    training_history = []

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Maybe unfreeze backbone
        model.maybe_unfreeze(epoch)

        # Train (using train_one_epoch from training_utils.py)
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
            "val_optimal_threshold": val_metrics["optimal_threshold"],
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
        "test_metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v 
                         for k, v in test_metrics.items() 
                         if not isinstance(v, dict)},
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