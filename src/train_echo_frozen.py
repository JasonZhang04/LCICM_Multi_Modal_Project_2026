"""
Train a small MLP head on pre-extracted, frozen PanEcho embeddings.

This is Option A — mirrors the PCLR approach used in the multimodal pipeline:
  1. PanEcho ran once offline  →  {subject_id: tensor(768,)}  (pretrained_checkpoints/panecho_embeddings.pt)
  2. This script trains only the regression head on those 768-dim features.

Advantages:
  - No DICOM I/O during training (runs in minutes)
  - Same split/seed as the main pipeline for fair comparison
  - Serves as a fast upper-bound estimate on what frozen PanEcho features can do

Usage:
    python src/train_echo_frozen.py
    python src/train_echo_frozen.py --output_dir outputs/echo_frozen_v2 --num_epochs 300
    python src/train_echo_frozen.py --debug
"""

import argparse
import csv
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))

from echo_baseline.data.echo_dataset import build_echo_cohort
from echo_baseline.training.train_echo import EchoTrainConfig
from multimodal_aorta.data.dataset import TargetStats, compute_target_stats
from multimodal_aorta.data.splits import make_splits
from multimodal_aorta.utils.logging_utils import setup_logging

log = logging.getLogger(__name__)

EMBEDDINGS_PATH = "pretrained_checkpoints/panecho_embeddings.pt"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """
    Serves (embedding_768, target_2) pairs from pre-extracted PanEcho features.
    Fast: no DICOM loading, pure tensor ops.
    """

    def __init__(self, cohort, embeddings: dict, target_stats: Optional[TargetStats] = None):
        self.rows = cohort.reset_index(drop=True)
        self.embeddings = embeddings
        self.target_stats = target_stats

        # Filter to patients that actually have embeddings
        mask = self.rows["subject_id"].isin(embeddings)
        n_missing = (~mask).sum()
        if n_missing:
            log.warning("%d patients in split have no embedding — skipped", n_missing)
        self.rows = self.rows[mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        sid = int(row["subject_id"])

        emb = self.embeddings[sid].float()  # (768,)

        target = torch.tensor(
            [row["target_root"], row["target_asc"]], dtype=torch.float32
        )
        if self.target_stats is not None:
            target = self.target_stats.normalize(target)

        return {"emb": emb, "target": target, "subject_id": sid}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FrozenEchoHead(nn.Module):
    """
    Lightweight regression head on top of frozen PanEcho 768-dim embeddings.
    LayerNorm  →  Linear(768→256)  →  GELU  →  Dropout  →  Linear(256→2)
    """

    def __init__(self, emb_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def masked_mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    valid = ~torch.isnan(tgt)
    n = valid.float().sum().clamp(min=1.0)
    return ((pred - tgt.nan_to_num(0.0)) ** 2 * valid.float()).sum() / n


def total_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return masked_mse(pred[:, 0:1], tgt[:, 0:1]) + masked_mse(pred[:, 1:2], tgt[:, 1:2])


@torch.no_grad()
def evaluate(model, loader, device, target_stats=None):
    model.eval()
    preds, targets = [], []
    loss_sum = 0.0

    for batch in loader:
        emb = batch["emb"].to(device)
        tgt = batch["target"].to(device)
        pred = model(emb)
        loss_sum += total_loss(pred, tgt).item()
        preds.append(pred.cpu())
        targets.append(tgt.cpu())

    preds_t   = torch.cat(preds)
    targets_t = torch.cat(targets)

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
        "val_loss": loss_sum / max(len(loader), 1),
        "mae_root": mae(p[:, 0], t[:, 0]),
        "mae_asc":  mae(p[:, 1], t[:, 1]),
        "r2_root":  r2(p[:, 0],  t[:, 0]),
        "r2_asc":   r2(p[:, 1],  t[:, 1]),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_frozen(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    early_stop_patience: int,
    out_dir: Path,
    device: torch.device,
    target_stats=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps  = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "mae_root", "mae_asc", "r2_root", "r2_asc", "lr"]
        )

    best_val_mae = float("inf")
    no_improve   = 0
    best_ckpt    = str(out_dir / "best_model.pt")
    global_step  = 0

    log.info("Frozen-head training: %d epochs  lr=%.4g  device=%s", num_epochs, lr, device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            emb = batch["emb"].to(device)
            tgt = batch["target"].to(device)
            pred = model(emb)
            loss = total_loss(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            global_step += 1

        train_loss /= max(len(train_loader), 1)
        metrics = evaluate(model, val_loader, device, target_stats)

        cur_lr = scheduler.get_last_lr()[0] * lr
        log.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | "
            "mae_root=%.4f mae_asc=%.4f | r2_root=%.4f r2_asc=%.4f",
            epoch, train_loss, metrics["val_loss"],
            metrics["mae_root"], metrics["mae_asc"],
            metrics["r2_root"],  metrics["r2_asc"],
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, metrics["val_loss"],
                metrics["mae_root"], metrics["mae_asc"],
                metrics["r2_root"], metrics["r2_asc"], cur_lr,
            ])

        mean_mae = np.nanmean([metrics["mae_root"], metrics["mae_asc"]])
        if mean_mae < best_val_mae:
            best_val_mae = mean_mae
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            log.info("  ✓ New best val MAE=%.4f — saved checkpoint", best_val_mae)
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                log.info("Early stopping at epoch %d (no improve for %d epochs)",
                         epoch, early_stop_patience)
                break

    log.info("Training complete. Best val MAE: %.4f  Checkpoint: %s", best_val_mae, best_ckpt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Frozen PanEcho embedding head training")
    p.add_argument("--output_dir",          type=str,   default="outputs/echo_frozen")
    p.add_argument("--embeddings_path",     type=str,   default=None)
    p.add_argument("--num_epochs",          type=int,   default=200)
    p.add_argument("--lr",                  type=float, default=1e-3)
    p.add_argument("--weight_decay",        type=float, default=0.01)
    p.add_argument("--batch_size",          type=int,   default=256)
    p.add_argument("--warmup_epochs",       type=int,   default=5)
    p.add_argument("--early_stop_patience", type=int,   default=30)
    p.add_argument("--hidden_dim",          type=int,   default=256)
    p.add_argument("--dropout",             type=float, default=0.2)
    p.add_argument("--debug", action="store_true",
                   help="10 epochs, subset of patients")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    setup_logging(args.output_dir)

    emb_path = args.embeddings_path or os.path.join(project_root, EMBEDDINGS_PATH)
    if not os.path.exists(emb_path):
        log.error(
            "Embeddings not found at %s — run extraction first:\n"
            "  sbatch scripts/slurm_extract_panecho.sh", emb_path
        )
        sys.exit(1)

    log.info("Loading PanEcho embeddings from %s...", emb_path)
    embeddings = torch.load(emb_path, map_location="cpu")
    log.info("Loaded %d patient embeddings", len(embeddings))

    # ── Build cohort (same pipeline as train_echo.py) ──────────────────────
    cfg = EchoTrainConfig()
    cohort = build_echo_cohort(
        echo_labels_path     = cfg.echo_labels_path,
        echo_structured_path = cfg.echo_structured_path,
        echo_study_list      = cfg.echo_study_list,
        echo_record_list     = cfg.echo_record_list,
    )

    if args.debug:
        cohort = cohort.head(200).reset_index(drop=True)
        args.num_epochs = 10
        log.info("Debug mode: 200 patients, 10 epochs")

    # ── Splits — identical seed/fractions as main pipeline ────────────────
    train_ids, val_ids, test_ids = make_splits(
        cohort,
        col_root   = "target_root",
        train_frac = cfg.train_frac,
        val_frac   = cfg.val_frac,
        seed       = cfg.split_seed,
    )
    log.info("Splits — train: %d  val: %d  test: %d", len(train_ids), len(val_ids), len(test_ids))

    # ── Target normalization ──────────────────────────────────────────────
    train_cohort = cohort[cohort["subject_id"].isin(train_ids)].reset_index(drop=True)
    target_stats = compute_target_stats(train_cohort)
    log.info(
        "Target stats — root: mean=%.4f std=%.4f | asc: mean=%.4f std=%.4f",
        target_stats.root_mean, target_stats.root_std,
        target_stats.asc_mean,  target_stats.asc_std,
    )

    def make_ds(ids):
        sub = cohort[cohort["subject_id"].isin(ids)].reset_index(drop=True)
        return EmbeddingDataset(sub, embeddings, target_stats)

    train_ds = make_ds(train_ids)
    val_ds   = make_ds(val_ids)
    test_ds  = make_ds(test_ids)
    log.info("Embedding dataset sizes — train: %d  val: %d  test: %d",
             len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────
    model = FrozenEchoHead(emb_dim=768, hidden_dim=args.hidden_dim, dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("FrozenEchoHead: %d parameters", n_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    train_frozen(
        model, train_loader, val_loader,
        num_epochs          = args.num_epochs,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        warmup_epochs       = args.warmup_epochs,
        early_stop_patience = args.early_stop_patience,
        out_dir             = out_dir,
        device              = device,
        target_stats        = target_stats,
    )

    # ── Final test evaluation ─────────────────────────────────────────────
    log.info("Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(str(out_dir / "best_model.pt"), map_location=device))
    test_metrics = evaluate(model, test_loader, device, target_stats)
    log.info(
        "Test results — mae_root=%.4f  mae_asc=%.4f  r2_root=%.4f  r2_asc=%.4f",
        test_metrics["mae_root"], test_metrics["mae_asc"],
        test_metrics["r2_root"],  test_metrics["r2_asc"],
    )

    # Save test metrics to file
    import json
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    log.info("Test metrics saved → %s/test_metrics.json", out_dir)


if __name__ == "__main__":
    main()
