"""
CLI entry point for the single-modality echo baseline model.

Usage (from project root):
    python src/train_echo.py
    python src/train_echo.py --batch_size 8 --num_epochs 30 --output_dir outputs/echo_v2
    python src/train_echo.py --debug   # 2 epochs, 50 patients, fast
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from echo_baseline.data.echo_dataset import build_echo_cohort, EchoDataset
from echo_baseline.models.echo_model import EchoModel
from echo_baseline.training.train_echo import EchoTrainConfig, train_echo
from multimodal_aorta.data.splits import make_splits
from multimodal_aorta.data.dataset import compute_target_stats
from multimodal_aorta.utils.logging_utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser(description="Single-modality echo baseline")
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--num_epochs",   type=int,   default=None)
    p.add_argument("--output_dir",   type=str,   default=None)
    p.add_argument("--n_clips",      type=int,   default=None)
    p.add_argument("--lr_backbone",  type=float, default=None)
    p.add_argument("--lr_head",      type=float, default=None)
    p.add_argument("--debug", action="store_true",
                   help="2 epochs, 50 patients, batch_size=4, 0 workers")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = EchoTrainConfig()

    if args.batch_size  is not None: cfg.batch_size  = args.batch_size
    if args.num_epochs  is not None: cfg.num_epochs  = args.num_epochs
    if args.output_dir  is not None: cfg.output_dir  = args.output_dir
    if args.n_clips     is not None: cfg.n_clips     = args.n_clips
    if args.lr_backbone is not None: cfg.lr_backbone = args.lr_backbone
    if args.lr_head     is not None: cfg.lr_head     = args.lr_head

    if args.debug:
        cfg.num_epochs            = 2
        cfg.batch_size            = 4
        cfg.num_workers           = 0
        cfg.early_stop_patience   = 999
        cfg.freeze_backbone_epochs = 0
        cfg.output_dir            = cfg.output_dir + "_debug"

    setup_logging(cfg.output_dir)
    log = logging.getLogger(__name__)
    log.info("EchoTrainConfig: %s", cfg)

    # ── Build cohort ─────────────────────────────────────────────────────
    log.info("Building echo cohort...")
    cohort = build_echo_cohort(
        echo_labels_path    = cfg.echo_labels_path,
        echo_structured_path= cfg.echo_structured_path,
        echo_study_list     = cfg.echo_study_list,
        echo_record_list    = cfg.echo_record_list,
    )

    if args.debug:
        cohort = cohort.head(50).reset_index(drop=True)
        log.info("Debug mode: truncated to 50 patients")

    # ── Splits (same seed as multimodal pipeline for fair comparison) ─────
    train_ids, val_ids, _ = make_splits(
        cohort,
        col_root   = "target_root",
        train_frac = cfg.train_frac,
        val_frac   = cfg.val_frac,
        seed       = cfg.split_seed,
    )
    log.info("Splits — train: %d  val: %d", len(train_ids), len(val_ids))

    # ── Target normalization ──────────────────────────────────────────────
    target_stats = None
    if cfg.target_normalize:
        train_cohort = cohort[cohort["subject_id"].isin(train_ids)].reset_index(drop=True)
        target_stats = compute_target_stats(train_cohort)
        log.info(
            "Target stats — root: mean=%.4f std=%.4f | asc: mean=%.4f std=%.4f",
            target_stats.root_mean, target_stats.root_std,
            target_stats.asc_mean,  target_stats.asc_std,
        )

    def make_split(ids):
        sub = cohort[cohort["subject_id"].isin(ids)].reset_index(drop=True)
        return EchoDataset(
            sub,
            echo_root    = cfg.echo_root,
            n_clips      = cfg.n_clips,
            n_frames     = cfg.n_frames,
            img_size     = cfg.img_size,
            target_stats = target_stats,
        )

    train_ds = make_split(train_ids)
    val_ds   = make_split(val_ids)
    log.info("Dataset sizes — train: %d  val: %d", len(train_ds), len(val_ds))

    num_workers = 0 if args.debug else cfg.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    )

    # ── Build model ───────────────────────────────────────────────────────
    log.info("Building EchoModel (R2Plus1D-18 backbone)...")
    model = EchoModel(
        hidden_dim             = 128,
        dropout                = 0.3,
        freeze_backbone_epochs = cfg.freeze_backbone_epochs,
    )
    total_p    = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Params — total: %s  trainable: %s", f"{total_p:,}", f"{trainable:,}")

    # ── Train ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_echo(model, train_loader, val_loader, cfg,
               output_dir=cfg.output_dir, device=device, target_stats=target_stats)


if __name__ == "__main__":
    main()
