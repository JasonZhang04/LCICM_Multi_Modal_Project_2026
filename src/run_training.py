"""
Entry point for training the multimodal aortic diameter prediction model.

Usage (from the src/ directory):
    # Standard run
    python run_training.py

    # Override specific config values
    python run_training.py --batch_size 16 --num_epochs 30 --max_days_offset 365

    # Resume from a checkpoint
    python run_training.py --resume /path/to/checkpoint.pt

    # Debug: 2 epochs, tiny batch, CPU only
    python run_training.py --debug
"""

import argparse
import logging
import os
import sys

import torch

# Ensure the src/ directory is on the Python path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from multimodal_aorta.configs.default_config import Config
from multimodal_aorta.data.dataset import build_cohort, AortaDataset
from multimodal_aorta.data.splits import make_splits
from multimodal_aorta.data.preprocessing import load_ecg, load_cxr, collate_fn
from multimodal_aorta.models.full_model import AortaModel
from multimodal_aorta.training.train import train
from multimodal_aorta.utils.logging_utils import setup_logging, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Train multimodal aortic diameter model")

    # Config overrides
    p.add_argument("--batch_size",       type=int,   default=None)
    p.add_argument("--num_epochs",       type=int,   default=None)
    p.add_argument("--max_days_offset",  type=int,   default=None)
    p.add_argument("--output_dir",       type=str,   default=None)
    p.add_argument("--ecg_encoder",      type=str,   default=None,
                   choices=["resnet1d", "ecgfm"],
                   help="ECG encoder backend (resnet1d=baseline, ecgfm=future)")

    # Resume
    p.add_argument("--resume",           type=str,   default=None,
                   help="Path to checkpoint to resume training from")

    # Debug mode
    p.add_argument("--debug", action="store_true",
                   help="2 epochs, batch_size=4, no workers")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    # --- Apply CLI overrides ---
    if args.batch_size    is not None: cfg.train.batch_size    = args.batch_size
    if args.num_epochs    is not None: cfg.train.num_epochs    = args.num_epochs
    if args.max_days_offset is not None: cfg.data.max_days_offset = args.max_days_offset
    if args.output_dir    is not None: cfg.train.output_dir    = args.output_dir
    if args.ecg_encoder   is not None: cfg.model.ecg_encoder   = args.ecg_encoder

    if args.debug:
        cfg.train.num_epochs          = 2
        cfg.train.batch_size          = 4
        cfg.train.num_workers         = 0
        cfg.train.early_stop_patience = 999
        cfg.train.output_dir          = cfg.train.output_dir + "_debug"

    # --- Logging ---
    setup_logging(cfg.train.output_dir)
    log = logging.getLogger(__name__)
    log.info("Config:\n  data=%s\n  model=%s\n  train=%s",
             cfg.data, cfg.model, cfg.train)

    # --- Build cohort ---
    log.info("Building cohort...")
    cohort = build_cohort(
        echo_labels_path    = cfg.data.echo_labels_path,
        echo_structured_path= cfg.data.echo_structured_path,
        ecg_root            = cfg.data.ecg_root,
        cxr_record_list     = cfg.data.cxr_record_list,
        cxr_png_root        = cfg.data.cxr_png_root,
        cxr_dicom_root      = cfg.data.cxr_dicom_root,
        cxr_metadata_cache  = cfg.data.cxr_metadata_cache,
        col_root            = cfg.data.col_root,
        col_asc             = cfg.data.col_asc,
        root_range          = cfg.data.root_range,
        asc_range           = cfg.data.asc_range,
        max_days_offset     = cfg.data.max_days_offset,
        cxr_extra_png_root  = cfg.data.cxr_extra_png_root,
        cxr_metadata_csv    = cfg.data.cxr_metadata_csv,
    )

    train_ids, val_ids, test_ids = make_splits(
        cohort,
        col_root       = "target_root",
        train_frac     = cfg.data.train_frac,
        val_frac       = cfg.data.val_frac,
        n_bins         = cfg.data.stratify_bins,
        seed           = cfg.data.split_seed,
    )

    log.info("Cohort splits — train: %d  val: %d  test: %d", len(train_ids), len(val_ids), len(test_ids))

    def make_subset(ids, is_train):
        sub = cohort[cohort["subject_id"].isin(ids)].reset_index(drop=True)
        return AortaDataset(
            sub,
            ecg_transform = load_ecg,
            cxr_transform = load_cxr,
            ecg_cfg       = cfg.data,
            cxr_cfg       = cfg.data,
            is_train      = is_train,
        )

    train_ds = make_subset(train_ids, is_train=True)
    val_ds   = make_subset(val_ids,   is_train=False)

    num_workers = 0 if args.debug else cfg.train.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        shuffle=True,  num_workers=num_workers,
        pin_memory=cfg.train.pin_memory and torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=cfg.train.pin_memory and torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    log.info("Dataset sizes — train: %d  val: %d", len(train_ds), len(val_ds))

    # --- Build model ---
    log.info("Building model (ecg_encoder=%s, cxr=%s)...",
             cfg.model.ecg_encoder, cfg.model.cxr_model_name)
    model = AortaModel(cfg.model, cfg.train)

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model params — total: %s  trainable: %s  frozen: %s",
             f"{total_p:,}", f"{trainable_p:,}", f"{total_p - trainable_p:,}")

    # --- Resume from checkpoint ---
    start_epoch = 0
    if args.resume:
        log.info("Resuming from checkpoint: %s", args.resume)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_checkpoint(args.resume, model, device=device)
        start_epoch = ckpt.get("epoch", 0)
        log.info("Resumed at epoch %d", start_epoch)

    # --- Train ---
    # Always auto-detect device (uses GPU if available, CPU otherwise).
    # debug mode only changes epochs/batch_size, not the device.
    train(model, train_loader, val_loader, cfg, output_dir=cfg.train.output_dir, device=None)


if __name__ == "__main__":
    main()
