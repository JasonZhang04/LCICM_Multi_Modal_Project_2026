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
from multimodal_aorta.data.dataset import build_cohort, AortaDataset, compute_target_stats
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
    p.add_argument("--ecg_pretrain_ckpt", type=str,  default=None,
                   help="Path to SimCLR-pretrained ECG encoder weights (from pretrain_ecg.py)")
    p.add_argument("--pclr_embeddings",  type=str,  default=None,
                   help="Path to PCLR embeddings .pt file (from scripts/extract_pclr_embeddings.py). "
                        "Automatically sets ecg_encoder=pclr_frozen.")
    p.add_argument("--cxr_embeddings",   type=str,  default=None,
                   help="Path to RAD-DINO embeddings .pt file (from scripts/extract_raddino_embeddings.py). "
                        "Automatically sets cxr_encoder=raddino_frozen (frozen precomputed CXR features).")

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
    if args.ecg_encoder       is not None: cfg.model.ecg_encoder       = args.ecg_encoder
    if args.ecg_pretrain_ckpt is not None: cfg.model.ecg_pretrain_ckpt = args.ecg_pretrain_ckpt
    if args.pclr_embeddings   is not None:
        cfg.data.pclr_embeddings_path = args.pclr_embeddings
        cfg.model.ecg_encoder = "pclr_frozen"
    if args.cxr_embeddings    is not None:
        cfg.data.cxr_embeddings_path = args.cxr_embeddings
        cfg.model.cxr_encoder = "raddino_frozen"

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

    # --- Target normalization (Step 1) ---
    # Compute z-score stats on the TRAIN split only, then apply to all splits
    # in the dataset's __getitem__. evaluate() de-normalizes predictions to cm
    # before reporting MAE/R²/AUROC.
    target_stats = None
    if cfg.data.target_normalize:
        train_cohort = cohort[cohort["subject_id"].isin(train_ids)].reset_index(drop=True)
        target_stats = compute_target_stats(train_cohort)
        log.info(
            "Target stats (train split, cm) — root: mean=%.4f std=%.4f | asc: mean=%.4f std=%.4f",
            target_stats.root_mean, target_stats.root_std,
            target_stats.asc_mean,  target_stats.asc_std,
        )

    # --- PCLR precomputed embeddings (optional) ---
    ecg_embeddings = None
    if cfg.data.pclr_embeddings_path:
        from multimodal_aorta.models.ecg_encoder import PCLREmbeddingEncoder
        ecg_embeddings = PCLREmbeddingEncoder.load_embeddings(cfg.data.pclr_embeddings_path)
        log.info(
            "PCLR embeddings loaded: %d subjects — ecg_encoder set to 'pclr_frozen'",
            len(ecg_embeddings),
        )

    # --- RAD-DINO precomputed CXR embeddings (optional) ---
    cxr_embeddings = None
    if cfg.data.cxr_embeddings_path:
        from multimodal_aorta.models.cxr_encoder import CXREmbeddingEncoder
        cxr_embeddings = CXREmbeddingEncoder.load_embeddings(cfg.data.cxr_embeddings_path)
        log.info(
            "RAD-DINO embeddings loaded: %d subjects — cxr_encoder set to 'raddino_frozen'",
            len(cxr_embeddings),
        )

    def make_subset(ids, is_train):
        sub = cohort[cohort["subject_id"].isin(ids)].reset_index(drop=True)
        return AortaDataset(
            sub,
            ecg_transform  = load_ecg,
            cxr_transform  = load_cxr,
            ecg_cfg        = cfg.data,
            cxr_cfg        = cfg.data,
            is_train       = is_train,
            target_stats   = target_stats,
            ecg_embeddings = ecg_embeddings,
            cxr_embeddings = cxr_embeddings,
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
    train(
        model, train_loader, val_loader, cfg,
        output_dir=cfg.train.output_dir,
        device=None,
        target_stats=target_stats,
    )


if __name__ == "__main__":
    main()
