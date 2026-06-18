"""
CLI entry point for SimCLR self-supervised ECG pretraining.

Usage (from the project root or src/ directory):
    # Standard 100-epoch pretraining run (submit via SLURM, see scripts/slurm_pretrain_ecg.sh)
    python src/pretrain_ecg.py

    # Override hyperparameters
    python src/pretrain_ecg.py --epochs 50 --batch_size 256

    # Debug: 2 epochs, 1000 records, no GPU required
    python src/pretrain_ecg.py --debug
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from multimodal_aorta.models.ecg_encoder import ResNet1DEncoder
from multimodal_aorta.training.pretrain_ecg import PretrainConfig, pretrain
from multimodal_aorta.utils.logging_utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser(description="SimCLR self-supervised ECG pretraining")
    p.add_argument("--ecg_root",     type=str,   default=None)
    p.add_argument("--record_list",  type=str,   default=None)
    p.add_argument("--output_ckpt",  type=str,   default=None,
                   help="Path for saved encoder weights (default: pretrained_checkpoints/ecg_pretrain.pt)")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--tau",          type=float, default=None)
    p.add_argument("--num_workers",  type=int,   default=None)
    p.add_argument("--debug", action="store_true",
                   help="2 epochs, 1000 records, batch_size=64, 0 workers")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PretrainConfig()

    if args.ecg_root    is not None: cfg.ecg_root    = args.ecg_root
    if args.record_list is not None: cfg.record_list = args.record_list
    if args.output_ckpt is not None: cfg.output_ckpt = args.output_ckpt
    if args.epochs      is not None: cfg.epochs      = args.epochs
    if args.batch_size  is not None: cfg.batch_size  = args.batch_size
    if args.lr          is not None: cfg.lr          = args.lr
    if args.tau         is not None: cfg.tau         = args.tau
    if args.num_workers is not None: cfg.num_workers = args.num_workers

    if args.debug:
        cfg.epochs        = 2
        cfg.batch_size    = 64
        cfg.max_records   = 1000
        cfg.num_workers   = 0
        cfg.warmup_epochs = 1
        cfg.output_ckpt   = cfg.output_ckpt.replace(".pt", "_debug.pt")

    import pathlib
    log_dir = str(pathlib.Path(cfg.output_ckpt).parent)
    setup_logging(log_dir)

    log = logging.getLogger(__name__)
    log.info("PretrainConfig: %s", cfg)

    encoder = ResNet1DEncoder(out_dim=768)
    total_p = sum(p.numel() for p in encoder.parameters())
    log.info("Encoder params: %s", f"{total_p:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain(encoder, cfg, device=device)


if __name__ == "__main__":
    main()
