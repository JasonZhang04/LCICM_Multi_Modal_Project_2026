"""
Main training loop for the multimodal aortic diameter prediction model.

Key features:
  - AdamW with per-component learning rates (ECG enc / CXR enc / fusion+head)
  - Linear warmup for first `warmup_epochs` epochs, then CosineAnnealingLR
  - Mixed-precision training (torch.amp) with GradScaler
  - Gradient clipping (max_norm=1.0)
  - Early stopping on validation total MAE (patience=10)
  - Saves best-val-MAE checkpoint and final-epoch checkpoint
  - Logs all metrics to CSV and console each epoch
"""

import dataclasses
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from multimodal_aorta.configs.default_config import Config
from multimodal_aorta.data.dataset import TargetStats
from multimodal_aorta.models.full_model import AortaModel
from multimodal_aorta.training.losses import total_loss
from multimodal_aorta.training.evaluate import (
    evaluate,
    evaluate_ablation,
    evaluate_mean_baseline,
)
from multimodal_aorta.utils.logging_utils import CSVLogger, save_checkpoint, plot_training_curves

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warmup + Cosine scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs` then CosineAnnealingLR.
    Wraps the base CosineAnnealingLR and applies a warmup scale on top.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 1e-6,
    ):
        self.warmup_epochs = warmup_epochs
        self._cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=eta_min,
        )
        self._optimizer = optimizer
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._epoch = 0

    def step(self) -> None:
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            # Linear warmup: scale LR from 0 → base_lr
            scale = self._epoch / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self._optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * scale
        else:
            self._cosine.step()

    def get_last_lrs(self):
        return [pg["lr"] for pg in self._optimizer.param_groups]

    def state_dict(self):
        return {"epoch": self._epoch, "cosine": self._cosine.state_dict()}

    def load_state_dict(self, d):
        self._epoch = d["epoch"]
        self._cosine.load_state_dict(d["cosine"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_dual_modality(loader: DataLoader) -> int:
    """Count samples in a loader that genuinely have BOTH ECG and CXR present."""
    n = 0
    for batch in loader:
        n += int((batch["has_ecg"] & batch["has_cxr"]).sum().item())
    return n


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: AortaModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    output_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    target_stats: Optional[TargetStats] = None,
) -> None:
    """
    Full training loop.

    Args:
        model:        AortaModel instance (on CPU — moved to device inside)
        train_loader: DataLoader for training split
        val_loader:   DataLoader for validation split
        cfg:          Config dataclass (cfg.train, cfg.model used)
        output_dir:   Where to save checkpoints and logs (overrides cfg.train.output_dir)
        device:       torch.device (auto-detects CUDA if None)
        target_stats: train-split z-score stats; passed to evaluate() so
                      reported MAE/R²/AUROC are de-normalized back to cm.
                      When None, predictions and targets are assumed to
                      already be in cm.
    """
    tc = cfg.train
    mc = cfg.model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    out_dir = Path(output_dir or tc.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # --- Optimizer (per-component LR groups) ---
    param_groups = model.get_param_groups()
    optimizer = optim.AdamW(param_groups, weight_decay=tc.weight_decay)

    # --- Scheduler ---
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=tc.warmup_epochs,
        total_epochs=tc.num_epochs,
    )

    # --- Mixed precision ---
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # --- Logging ---
    csv_log = CSVLogger(str(out_dir / "training_log.csv"))
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
        logger.info("TensorBoard writer → %s/tensorboard", out_dir)
    except Exception:
        tb_writer = None

    # --- Config snapshot for checkpoints ---
    config_dict = {
        "model": dataclasses.asdict(mc),
        "train": dataclasses.asdict(tc),
        "target_stats": dataclasses.asdict(target_stats) if target_stats is not None else None,
    }

    # --- Early stopping state ---
    best_val_mae = float("inf")
    epochs_no_improve = 0
    best_ckpt_path = str(out_dir / "best_model.pt")
    final_ckpt_path = str(out_dir / "final_model.pt")

    logger.info("Starting training: %d epochs, batch_size=%d", tc.num_epochs, tc.batch_size)
    logger.info("Output dir: %s", out_dir)

    for epoch in range(tc.num_epochs):

        # CXR freeze schedule
        model.on_epoch_start(epoch)

        # ----------------------------------------------------------------
        # Training epoch
        # ----------------------------------------------------------------
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            ecg     = batch["ecg"].to(device, non_blocking=True)
            cxr     = batch["cxr"].to(device, non_blocking=True)
            target  = batch["target"].to(device, non_blocking=True)
            has_ecg = batch["has_ecg"].to(device, non_blocking=True)
            has_cxr = batch["has_cxr"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast("cuda", enabled=use_amp):
                pred = model(ecg, cxr, has_ecg, has_cxr)
                loss = total_loss(pred, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()

        scheduler.step()
        train_loss = train_loss_accum / max(len(train_loader), 1)
        lrs = scheduler.get_last_lrs()  # [lr_ecg, lr_cxr, lr_fusion]

        # ----------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------
        val_metrics = evaluate(model, val_loader, device, target_stats=target_stats)
        val_mae = val_metrics.total_mae

        # ----------------------------------------------------------------
        # Console log
        # ----------------------------------------------------------------
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f  val_loss=%.4f  "
            "MAE root=%.4f asc=%.4f  R² root=%.3f asc=%.3f  "
            "AUROC root=%.3f asc=%.3f",
            epoch + 1, tc.num_epochs,
            train_loss, val_metrics.val_loss,
            val_metrics.mae_root, val_metrics.mae_asc,
            val_metrics.r2_root,  val_metrics.r2_asc,
            val_metrics.auroc_root, val_metrics.auroc_asc,
        )

        # ----------------------------------------------------------------
        # CSV log
        # ----------------------------------------------------------------
        csv_log.write(
            epoch=epoch + 1,
            train_loss=train_loss,
            lr_ecg=lrs[0],
            lr_cxr=lrs[1],
            lr_fusion=lrs[2],
            **val_metrics.to_dict(),
        )

        if tb_writer is not None:
            ep = epoch + 1
            tb_writer.add_scalars("Loss", {"train": train_loss, "val": val_metrics.val_loss}, ep)
            tb_writer.add_scalars("MAE/root", {"val": val_metrics.mae_root}, ep)
            tb_writer.add_scalars("MAE/asc",  {"val": val_metrics.mae_asc},  ep)
            tb_writer.add_scalars("R2/root",  {"val": val_metrics.r2_root},  ep)
            tb_writer.add_scalars("R2/asc",   {"val": val_metrics.r2_asc},   ep)
            tb_writer.add_scalars("AUROC/root", {"val": val_metrics.auroc_root}, ep)
            tb_writer.add_scalars("AUROC/asc",  {"val": val_metrics.auroc_asc},  ep)
            tb_writer.add_scalar("LR/ecg",    lrs[0], ep)
            tb_writer.add_scalar("LR/cxr",    lrs[1], ep)
            tb_writer.add_scalar("LR/fusion", lrs[2], ep)

        # ----------------------------------------------------------------
        # Checkpointing
        # ----------------------------------------------------------------
        metrics_dict = {"val_mae": val_mae, **val_metrics.to_dict()}

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            save_checkpoint(
                best_ckpt_path, model, optimizer, epoch + 1,
                metrics_dict, config_dict, scheduler, scaler,
            )
            logger.info("  ↳ New best val MAE=%.4f — checkpoint saved.", best_val_mae)
        else:
            epochs_no_improve += 1

        # ----------------------------------------------------------------
        # Early stopping
        # ----------------------------------------------------------------
        if epochs_no_improve >= tc.early_stop_patience:
            logger.info(
                "Early stopping triggered at epoch %d "
                "(no improvement for %d epochs).",
                epoch + 1, tc.early_stop_patience,
            )
            break

    # Save final epoch checkpoint regardless
    final_metrics = evaluate(model, val_loader, device, target_stats=target_stats)
    save_checkpoint(
        final_ckpt_path, model, optimizer, epoch + 1,
        {"val_mae": final_metrics.total_mae, **final_metrics.to_dict()},
        config_dict, scheduler, scaler,
    )
    logger.info("Final checkpoint saved → %s", final_ckpt_path)

    # ----------------------------------------------------------------
    # Validation ablation + baselines (load best model)
    # ----------------------------------------------------------------
    logger.info("Running per-modality ablation on validation set with best model...")
    from multimodal_aorta.utils.logging_utils import load_checkpoint
    load_checkpoint(best_ckpt_path, model, device=device)

    # --- Reference baseline: constant train-mean predictor (the bar to beat) ---
    base_full = evaluate_mean_baseline(val_loader, target_stats=target_stats)
    logger.info(
        "Baseline [mean-predictor, full val] — MAE root=%.4f asc=%.4f  "
        "R² root=%.3f asc=%.3f  (R²~0 by construction; MAE is the bar to beat)",
        base_full.mae_root, base_full.mae_asc, base_full.r2_root, base_full.r2_asc,
    )

    # --- Whole-val ablation (NOTE: confounded by ECG-only patients; kept for
    #     backward-comparison with earlier runs only) ---
    logger.info(
        "Per-modality ablation on FULL val set "
        "(confounded by ECG-only patients — see dual-modality block below):"
    )
    ablation = evaluate_ablation(model, val_loader, device, target_stats=target_stats)
    for name, m in ablation.items():
        logger.info(
            "Ablation [%s, full val] — MAE root=%.4f asc=%.4f  R² root=%.3f asc=%.3f",
            name, m.mae_root, m.mae_asc, m.r2_root, m.r2_asc,
        )

    # --- Dual-modality ablation (THE honest multimodal comparison) ---
    # Scored ONLY on patients who genuinely have BOTH ECG and CXR, so
    # 'both' vs 'ecg_only' vs 'cxr_only' isolates each modality's contribution.
    n_both = _count_dual_modality(val_loader)
    logger.info(
        "Per-modality ablation on DUAL-MODALITY subset (n=%d patients with both ECG+CXR) "
        "— this is the honest test of whether CXR adds signal:",
        n_both,
    )
    if n_both < 2:
        logger.warning(
            "Dual-modality subset has <2 patients (n=%d); skipping dual-modality "
            "ablation. CXR contribution cannot be measured on this split.", n_both,
        )
    else:
        dual_ablation = evaluate_ablation(
            model, val_loader, device, target_stats=target_stats, restrict_to_both=True,
        )
        base_dual = evaluate_mean_baseline(
            val_loader, target_stats=target_stats, restrict_to_both=True,
        )
        logger.info(
            "Baseline [mean-predictor, dual subset] — MAE root=%.4f asc=%.4f  "
            "R² root=%.3f asc=%.3f",
            base_dual.mae_root, base_dual.mae_asc, base_dual.r2_root, base_dual.r2_asc,
        )
        for name, m in dual_ablation.items():
            logger.info(
                "Ablation [%s, dual subset] — MAE root=%.4f asc=%.4f  "
                "R² root=%.3f asc=%.3f  AUROC root=%.3f asc=%.3f",
                name, m.mae_root, m.mae_asc, m.r2_root, m.r2_asc,
                m.auroc_root, m.auroc_asc,
            )
        d_both = dual_ablation["both"]
        d_ecg  = dual_ablation["ecg_only"]
        logger.info(
            "CXR marginal contribution (dual subset, both − ecg_only) — "
            "ΔR² root=%+.3f asc=%+.3f  ΔMAE root=%+.4f asc=%+.4f",
            d_both.r2_root - d_ecg.r2_root, d_both.r2_asc - d_ecg.r2_asc,
            d_both.mae_root - d_ecg.mae_root, d_both.mae_asc - d_ecg.mae_asc,
        )

    if tb_writer is not None:
        tb_writer.close()

    plot_training_curves(str(out_dir / "training_log.csv"), str(out_dir))
    logger.info("Training complete. Best val MAE: %.4f", best_val_mae)
