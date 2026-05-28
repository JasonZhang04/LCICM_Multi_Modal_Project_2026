"""
CXR encoder: RAD-DINO wrapper (microsoft/rad-dino).

RAD-DINO is a ViT-B/16 pretrained on 882K chest X-rays via DINOv2 (Microsoft Research).
It outputs a 768-dim CLS token that captures chest X-ray–specific visual features.

Fine-tuning strategy:
  - Freeze the first `freeze_blocks` transformer blocks for the first
    `unfreeze_epoch` training epochs to protect pretrained CXR representations.
  - Unfreeze everything afterwards for full fine-tuning.
"""

import logging

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

_RAD_DINO_ID = "microsoft/rad-dino"


class CXREncoder(nn.Module):
    """
    Wraps RAD-DINO (ViT-B/16, DINOv2-pretrained on chest X-rays).

    Input:  (B, 3, H, W)  — ImageNet-normalised, any resolution (224×224 standard)
    Output: (B, 768)      — CLS token from last ViT layer

    RAD-DINO uses interpolated position embeddings so it handles 224×224 input
    even though it was pretrained at 518×518.
    """

    def __init__(self, model_name: str = _RAD_DINO_ID, freeze_blocks: int = 6):
        super().__init__()
        logger.info("Loading CXR encoder: %s", model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim: int = self.model.config.hidden_size  # 768 for ViT-B
        self._freeze_blocks = freeze_blocks
        self._freeze_first_n_blocks(freeze_blocks)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) float32, ImageNet-normalised
        Returns:
            (B, 768) CLS token embedding
        """
        out = self.model(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0, :]   # (B, 768)
        return cls_token

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers (called from training loop)
    # ------------------------------------------------------------------

    def _freeze_first_n_blocks(self, n: int) -> None:
        """Freeze patch embedding + first n transformer blocks."""
        # Always freeze patch embedding and positional params
        for p in self.model.embeddings.parameters():
            p.requires_grad = False

        encoder_layers = self.model.encoder.layer
        for i, block in enumerate(encoder_layers):
            freeze = i < n
            for p in block.parameters():
                p.requires_grad = not freeze

        n_frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        logger.info(
            "CXR encoder: froze embeddings + blocks 0–%d (%d params frozen).", n - 1, n_frozen
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        for p in self.model.parameters():
            p.requires_grad = True
        logger.info("CXR encoder: all parameters unfrozen.")

    def set_frozen_for_epoch(self, epoch: int, unfreeze_epoch: int) -> None:
        """
        Called at the start of each training epoch.
        Freezes first `_freeze_blocks` blocks before `unfreeze_epoch`,
        unfreezes everything from `unfreeze_epoch` onward.
        """
        if epoch < unfreeze_epoch:
            self._freeze_first_n_blocks(self._freeze_blocks)
        else:
            if epoch == unfreeze_epoch:
                self.unfreeze_all()
