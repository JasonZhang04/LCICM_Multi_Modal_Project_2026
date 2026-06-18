"""
SimCLR self-supervised pretraining for the ECG encoder.

Trains ResNet1DEncoder + ProjectionHead with NT-Xent loss on MIMIC-IV-ECG.
After training, only encoder weights are saved; the projection head is discarded
(standard SimCLR procedure — the head improves training but hurts transfer).

Loss reference:
  Chen et al., "A Simple Framework for Contrastive Learning of Visual
  Representations", ICML 2020. https://arxiv.org/abs/2002.05709
"""

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class PretrainConfig:
    ecg_root: str = (
        "/scratch4/rsteven1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
    )
    record_list: str = (
        "/scratch4/rsteven1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
        "/record_list.csv"
    )
    output_ckpt: str = "pretrained_checkpoints/ecg_pretrain.pt"

    # Training
    epochs: int = 100
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # NT-Xent temperature
    tau: float = 0.07

    # Data loading
    num_workers: int = 8
    pin_memory: bool = True

    # Set for debugging (limits dataset size)
    max_records: Optional[int] = None


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head for SimCLR.
    Architecture: Linear(in) → BN → GELU → Linear(out).
    Discarded after pretraining; only the encoder weights are kept.
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    NT-Xent (normalized temperature-scaled cross entropy) loss for SimCLR.

    Given two batches of projections z1, z2 ∈ R^(B × D), each pair (z1[i], z2[i])
    is a positive pair; all other 2(B-1) combinations are negatives.

    Args:
        z1, z2: (B, D) projection embeddings. L2-normalized internally.
        tau:    Temperature. Lower τ → harder negatives. Default 0.07.

    Returns:
        Scalar loss. Chance level ≈ log(B); well-trained ≈ 2–3.
    """
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)    # (2B, D)

    sim = torch.mm(z, z.T) / tau                           # (2B, 2B)

    # Mask self-similarity (diagonal) — no sample is its own positive
    eye = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))

    # Positive indices: for row i in [0, B) the positive is i+B; vice versa
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])

    return F.cross_entropy(sim, labels)


def _cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def pretrain(
    encoder: nn.Module,
    cfg: PretrainConfig,
    device: Optional[torch.device] = None,
) -> None:
    """
    Run SimCLR pretraining on the full MIMIC-IV-ECG corpus.

    Saves encoder-only state_dict to cfg.output_ckpt every 10 epochs and
    at the end of training. The projection head is never saved.

    Args:
        encoder: ResNet1DEncoder (or any ECGEncoder with .out_dim attribute).
        cfg:     PretrainConfig with all hyperparameters and paths.
        device:  torch.device (auto-detects CUDA if None).
    """
    from multimodal_aorta.data.ecg_pretrain_dataset import ECGPretrainDataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Pretraining device: %s", device)

    # --- Dataset & loader ---
    dataset = ECGPretrainDataset(
        ecg_root=cfg.ecg_root,
        record_list=cfg.record_list,
        max_records=cfg.max_records,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        drop_last=True,   # NT-Xent is unstable with partial batches
        persistent_workers=cfg.num_workers > 0,
    )
    n_batches = len(loader)
    logger.info(
        "Loader: %d records / %d batches per epoch (drop_last=True, batch_size=%d)",
        len(dataset), n_batches, cfg.batch_size,
    )

    # --- Model: encoder + projection head ---
    head = ProjectionHead(in_dim=encoder.out_dim)
    encoder = encoder.to(device)
    head = head.to(device)
    all_params = list(encoder.parameters()) + list(head.parameters())

    # --- Optimizer (no per-group LRs needed; all params share the same schedule) ---
    optimizer = optim.AdamW(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- Mixed precision ---
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # --- Output path ---
    out_path = Path(cfg.output_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = cfg.epochs * n_batches
    warmup_steps = cfg.warmup_epochs * n_batches

    chance_loss = math.log(cfg.batch_size)
    logger.info(
        "Starting SimCLR: %d epochs, batch_size=%d, τ=%.3f, "
        "warmup=%d epochs, chance_loss≈%.3f",
        cfg.epochs, cfg.batch_size, cfg.tau,
        cfg.warmup_epochs, chance_loss,
    )

    global_step = 0
    for epoch in range(cfg.epochs):
        encoder.train()
        head.train()
        epoch_loss = 0.0

        for v1, v2 in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            # Manual warmup + cosine LR schedule (per batch)
            lr = _cosine_lr(global_step, warmup_steps, total_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                h1 = encoder(v1)
                h2 = encoder(v2)
                z1 = head(h1)
                z2 = head(h2)
                loss = nt_xent_loss(z1, z2, tau=cfg.tau)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(
            "Epoch %3d/%d | loss=%.4f  (chance=%.3f)  lr=%.2e",
            epoch + 1, cfg.epochs, avg_loss, chance_loss, lr,
        )

        # Save encoder weights every 10 epochs and at the end
        if (epoch + 1) % 10 == 0 or epoch + 1 == cfg.epochs:
            _save_encoder(encoder, out_path, epoch + 1)

    logger.info("Pretraining complete. Encoder saved → %s", out_path)


def _save_encoder(encoder: nn.Module, path: Path, epoch: int) -> None:
    torch.save(encoder.state_dict(), str(path))
    logger.info("Encoder checkpoint → %s  (epoch %d)", path, epoch)
