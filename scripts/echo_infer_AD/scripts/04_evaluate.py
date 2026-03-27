#!/usr/bin/env python3
"""
Step 4: Evaluate trained model with detailed metrics.

Loads the best checkpoint and runs comprehensive evaluation on the test set,
including confusion matrix, per-class metrics, and error analysis.

Usage:
    python scripts/04_evaluate.py --config configs/default.yaml
    python scripts/04_evaluate.py --config configs/default.yaml --checkpoint /path/to/best_model.pt
"""

import os
import sys
import argparse
import logging
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.jhu_dataset import create_dataloaders
from models.echo_classifier import AorticDilationClassifier
from utils.training_utils import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def get_predictions(model, loader, device):
    """Get all predictions, probabilities, and metadata from a loader."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_accessions, all_paths = [], []

    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["label"]

        logits = model(video)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_accessions.extend(batch.get("accession", [""] * len(labels)))
        all_paths.extend(batch.get("dicom_path", [""] * len(labels)))

    return {
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
        "accessions": all_accessions,
        "paths": all_paths,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_cfg = config["model"]
    model_cfg["label_mode"] = config["labels"]["mode"]
    model = AorticDilationClassifier(model_cfg).to(device)

    ckpt_path = args.checkpoint or os.path.join(config["output"]["checkpoint_dir"], "best_model.pt")
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    load_checkpoint(model, None, ckpt_path, device)

    # Load test data
    video_cfg = {
        **config["video"],
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"]["num_workers"],
    }
    _, _, test_loader = create_dataloaders(
        labels_csv=config["data"]["labels_cache"],
        config=video_cfg,
        val_split=config["training"]["val_split"],
        test_split=config["training"]["test_split"],
        seed=config["training"]["seed"],
    )

    # Get predictions
    results = get_predictions(model, test_loader, device)
    y_true = results["labels"]
    y_pred = results["preds"]
    y_prob = results["probs"]

    # --- Classification Report ---
    if config["labels"]["mode"] == "binary":
        target_names = ["Normal", "Dilated"]
    else:
        target_names = ["Normal", "Mild", "Moderate", "Severe"]

    logger.info("\n" + "="*60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*60)
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    logger.info(f"\n{report}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")

    # --- ROC AUC (binary) ---
    if config["labels"]["mode"] == "binary" and y_prob.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        logger.info(f"\nROC AUC: {roc_auc:.4f}")

        # Find optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        logger.info(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")

    # --- Error Analysis ---
    logger.info("\n" + "="*60)
    logger.info("ERROR ANALYSIS")
    logger.info("="*60)

    errors = y_true != y_pred
    n_errors = errors.sum()
    logger.info(f"Total errors: {n_errors}/{len(y_true)} ({100*n_errors/len(y_true):.1f}%)")

    # False positives and false negatives
    if config["labels"]["mode"] == "binary":
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        logger.info(f"False positives (predicted dilated, actually normal): {fp}")
        logger.info(f"False negatives (predicted normal, actually dilated): {fn}")

    # Save detailed results
    output_dir = config["output"]["log_dir"]
    os.makedirs(output_dir, exist_ok=True)

    eval_df = pd.DataFrame({
        "accession": results["accessions"],
        "true_label": y_true,
        "predicted_label": y_pred,
        "prob_class_0": y_prob[:, 0],
        "prob_class_1": y_prob[:, 1] if y_prob.shape[1] > 1 else None,
        "correct": ~errors,
    })
    eval_path = os.path.join(output_dir, "evaluation_details.csv")
    eval_df.to_csv(eval_path, index=False)
    logger.info(f"\nDetailed results saved to {eval_path}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()