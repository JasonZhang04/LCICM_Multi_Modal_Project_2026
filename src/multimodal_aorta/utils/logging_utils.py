"""
Logging and checkpointing utilities.

CSVLogger      — writes one row per epoch to training_log.csv
save_checkpoint — saves model + optimizer state with metadata
load_checkpoint — restores from a checkpoint file
"""

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV training logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """
    Appends one row per epoch to a CSV file.

    Usage:
        csv_log = CSVLogger(output_dir / "training_log.csv")
        csv_log.write(epoch=1, train_loss=0.42, val_loss=0.38, ...)
    """

    # Column order in the output file
    COLUMNS = [
        "epoch",
        "train_loss",
        "val_loss",
        "total_mae",
        "mae_root",
        "mae_asc",
        "rmse_root",
        "rmse_asc",
        "r2_root",
        "r2_asc",
        "pearson_root",
        "pearson_asc",
        "auroc_root",
        "auroc_asc",
        "lr_ecg",
        "lr_cxr",
        "lr_fusion",
    ]

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = self.path.exists()

    def write(self, **kwargs) -> None:
        """Write one row. Unknown columns are accepted and appended at the end."""
        write_header = not self._initialized
        with open(self.path, "a", newline="") as f:
            # Build ordered row; unknown keys go after the defined columns
            all_keys = self.COLUMNS + [k for k in kwargs if k not in self.COLUMNS]
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._initialized = True
            row = {k: kwargs.get(k, "") for k in all_keys}
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> None:
    """
    Save a full training checkpoint.

    Stored keys:
        epoch, model_state_dict, optimizer_state_dict, metrics, config,
        scheduler_state_dict (if scheduler provided),
        scaler_state_dict     (if GradScaler provided)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":             metrics,
        "config":              config,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    torch.save(payload, path)
    logger.info("Checkpoint saved → %s  (epoch %d)", path, epoch)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Returns the full checkpoint dict so callers can read epoch, metrics, config.
    """
    map_loc = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_loc, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    logger.info("Checkpoint loaded ← %s  (epoch %d)", path, ckpt.get("epoch", "?"))
    return ckpt


# ---------------------------------------------------------------------------
# Console logging setup
# ---------------------------------------------------------------------------

def plot_training_curves(log_csv: str, output_dir: str) -> None:
    """
    Read training_log.csv and save a multi-panel PNG of training curves.
    Safe to call even if matplotlib is unavailable (logs a warning and returns).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for SLURM
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib/pandas not available — skipping training curve plots.")
        return

    csv_path = Path(log_csv)
    if not csv_path.exists():
        logger.warning("Training log not found at %s — skipping plots.", csv_path)
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    epochs = df["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training curves", fontsize=13)

    # --- Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, df["train_loss"], label="train")
    ax.plot(epochs, df["val_loss"],   label="val")
    ax.set_title("Huber Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- MAE ---
    ax = axes[0, 1]
    for col, lbl in [("mae_root", "root"), ("mae_asc", "asc")]:
        if col in df: ax.plot(epochs, df[col], label=lbl)
    ax.set_title("Validation MAE (cm)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (cm)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- R² ---
    ax = axes[0, 2]
    for col, lbl in [("r2_root", "root"), ("r2_asc", "asc")]:
        if col in df: ax.plot(epochs, df[col], label=lbl)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Validation R²")
    ax.set_xlabel("Epoch"); ax.set_ylabel("R²")
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- AUROC ---
    ax = axes[1, 0]
    for col, lbl in [("auroc_root", "root"), ("auroc_asc", "asc")]:
        if col in df: ax.plot(epochs, df[col], label=lbl)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Validation AUROC (threshold 4.0 cm)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- RMSE ---
    ax = axes[1, 1]
    for col, lbl in [("rmse_root", "root"), ("rmse_asc", "asc")]:
        if col in df: ax.plot(epochs, df[col], label=lbl)
    ax.set_title("Validation RMSE (cm)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE (cm)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Learning rates ---
    ax = axes[1, 2]
    for col, lbl in [("lr_ecg", "ECG enc"), ("lr_cxr", "CXR enc"), ("lr_fusion", "fusion")]:
        if col in df: ax.plot(epochs, df[col], label=lbl)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = out / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training curves saved → %s", save_path)


def setup_logging(output_dir: str, level: int = logging.INFO) -> None:
    """Configure root logger: console + file handler."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
