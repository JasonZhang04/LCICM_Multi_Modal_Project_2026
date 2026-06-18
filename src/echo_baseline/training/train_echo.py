"""
Training loop for the single-modality echo baseline model.
Mirrors the multimodal train.py structure: AdamW + warmup-cosine LR +
mixed precision + early stopping + CSV logging + checkpointing.
"""

import dataclasses
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EchoTrainConfig:
    # Paths
    echo_labels_path: str = (
        "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
        "/data/echo/aortic_labels.csv"
    )
    echo_structured_path: str = (
        "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
        "/data/echo/structured-measurement.csv"
    )
    echo_study_list: str = "/scratch4/rsteven1/Echo_Data/echo-study-list.csv"
    echo_record_list: str = "/scratch4/rsteven1/Echo_Data/echo-record-list.csv"
    echo_root: str = "/scratch4/rsteven1/Echo_Data"

    # Clip / frame sampling (match PanEcho training resolution)
    n_clips:  int = 8
    n_frames: int = 16
    img_size: int = 224

    # Training
    batch_size: int = 16
    num_epochs: int = 50
    early_stop_patience: int = 10

    lr_backbone: float = 1e-5
    lr_head:     float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_epochs: int = 3
    freeze_backbone_epochs: int = 5

    # Data normalization
    target_normalize: bool = True

    # Splits
    train_frac: float = 0.70
    val_frac: float   = 0.15
    split_seed: int   = 42    # same seed as multimodal pipeline

    # I/O
    num_workers: int = 4
    pin_memory: bool = True
    output_dir: str = "outputs/echo_baseline"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = ~torch.isnan(target)
    n = valid.float().sum().clamp(min=1.0)
    return ((pred - target.nan_to_num(0.0)) ** 2 * valid.float()).sum() / n


def total_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _masked_mse(pred[:, 0:1], target[:, 0:1]) + \
           _masked_mse(pred[:, 1:2], target[:, 1:2])


@torch.no_grad()
def evaluate_echo(model, loader, device, target_stats=None):
    model.eval()
    all_preds, all_targets = [], []
    total_loss_val = 0.0

    for batch in loader:
        video  = batch["video"].to(device)
        target = batch["target"].to(device)
        pred   = model(video)
        total_loss_val += total_loss(pred, target).item()
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    preds_t   = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)

    if target_stats is not None:
        preds_t   = target_stats.denormalize(preds_t)
        targets_t = target_stats.denormalize(targets_t)

    p = preds_t.numpy()
    t = targets_t.numpy()

    def mae(pi, ti):
        v = ~np.isnan(ti)
        return float(np.mean(np.abs(pi[v] - ti[v]))) if v.sum() > 0 else float("nan")

    def r2(pi, ti):
        v = ~np.isnan(ti)
        if v.sum() < 2:
            return float("nan")
        ss_res = np.sum((ti[v] - pi[v]) ** 2)
        ss_tot = np.sum((ti[v] - ti[v].mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "val_loss": total_loss_val / max(len(loader), 1),
        "mae_root": mae(p[:, 0], t[:, 0]),
        "mae_asc":  mae(p[:, 1], t[:, 1]),
        "r2_root":  r2(p[:, 0],  t[:, 0]),
        "r2_asc":   r2(p[:, 1],  t[:, 1]),
    }


# ---------------------------------------------------------------------------
# Warmup + cosine scheduler
# ---------------------------------------------------------------------------

def _lr_schedule(step: int, warmup: int, total: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_echo(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: EchoTrainConfig,
    output_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    target_stats=None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    param_groups = model.get_param_groups(
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
    )
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    use_amp = device.type == "cuda"
    scaler  = GradScaler("cuda", enabled=use_amp)

    total_steps  = cfg.num_epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)

    # CSV logger
    import csv
    log_path = out_dir / "training_log.csv"
    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss",
                         "mae_root", "mae_asc", "r2_root", "r2_asc",
                         "lr_backbone", "lr_head"])

    best_val_mae = float("inf")
    no_improve   = 0
    best_ckpt    = str(out_dir / "best_model.pt")
    global_step  = 0

    logger.info("Starting echo training: %d epochs, batch_size=%d", cfg.num_epochs, cfg.batch_size)

    for epoch in range(cfg.num_epochs):
        model.on_epoch_start(epoch)
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            video  = batch["video"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            # Per-batch LR schedule
            lr_b = _lr_schedule(global_step, warmup_steps, total_steps, cfg.lr_backbone)
            lr_h = _lr_schedule(global_step, warmup_steps, total_steps, cfg.lr_head)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_b if pg["name"] == "backbone" else lr_h

            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                pred = model(video)
                loss = total_loss(pred, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()
            global_step += 1

        train_loss = train_loss_accum / max(len(train_loader), 1)
        metrics = evaluate_echo(model, val_loader, device, target_stats)

        val_mae = (
            float(np.nanmean([metrics["mae_root"], metrics["mae_asc"]]))
            if not (np.isnan(metrics["mae_root"]) and np.isnan(metrics["mae_asc"]))
            else float("nan")
        )

        logger.info(
            "Epoch %3d/%d | train=%.4f  val=%.4f  "
            "MAE root=%.4f asc=%.4f  R² root=%.3f asc=%.3f",
            epoch + 1, cfg.num_epochs,
            train_loss, metrics["val_loss"],
            metrics["mae_root"], metrics["mae_asc"],
            metrics["r2_root"],  metrics["r2_asc"],
        )

        csv_writer.writerow([
            epoch + 1, train_loss, metrics["val_loss"],
            metrics["mae_root"], metrics["mae_asc"],
            metrics["r2_root"],  metrics["r2_asc"],
            lr_b, lr_h,
        ])
        csv_file.flush()

        if not np.isnan(val_mae) and val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            }, best_ckpt)
            logger.info("  ↳ New best val MAE=%.4f — saved.", best_val_mae)
        else:
            no_improve += 1

        if no_improve >= cfg.early_stop_patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs).",
                epoch + 1, cfg.early_stop_patience,
            )
            break

    csv_file.close()
    logger.info("Echo training complete. Best val MAE: %.4f", best_val_mae)
    logger.info("Best checkpoint: %s", best_ckpt)
