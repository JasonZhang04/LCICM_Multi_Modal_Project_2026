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


class CXREmbeddingEncoder(nn.Module):
    """
    Frozen CXR encoder backed by precomputed RAD-DINO embeddings.

    RAD-DINO is run ONCE offline (scripts/extract_raddino_embeddings.py) and the
    resulting {subject_id: tensor(768,)} dict is stored on disk. This class loads
    that dict (via AortaDataset) and applies a learnable Linear(768 → out_dim)
    projection during supervised training — exactly mirroring PCLREmbeddingEncoder
    on the ECG side.

    Why: only ~520 cohort patients have a CXR. Fine-tuning the 44M-param ViT on
    that many images overfit badly. Freezing it to precomputed features makes the
    trainable model tiny and training seconds-fast.

    Input (forward): (B, 768) float32 — precomputed RAD-DINO CLS embedding
    Output:          (B, out_dim) float32
    """

    def __init__(self, cxr_dim: int = 768, out_dim: int = 768):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(cxr_dim, out_dim)
        logger.info(
            "CXREmbeddingEncoder: Linear(%d → %d) projection (trained during fine-tuning)",
            cxr_dim, out_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 768) precomputed RAD-DINO embedding
        return self.proj(x)   # (B, out_dim)

    # No-op freeze hooks so the training loop can treat all CXR encoders uniformly.
    def set_frozen_for_epoch(self, epoch: int, unfreeze_epoch: int) -> None:
        pass

    @staticmethod
    def load_embeddings(path: str) -> dict:
        """Load precomputed RAD-DINO embeddings. Returns {subject_id (int): tensor(768,)}."""
        embs = torch.load(path, map_location="cpu", weights_only=False)
        logger.info("Loaded %d RAD-DINO embeddings from %s", len(embs), path)
        return embs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cxr_encoder(cfg) -> nn.Module:
    """Instantiate the CXR encoder specified in ModelConfig."""
    kind = getattr(cfg, "cxr_encoder", "rad_dino")
    if kind == "rad_dino":
        return CXREncoder(
            model_name=cfg.cxr_model_name,
            freeze_blocks=cfg.cxr_freeze_blocks,
        )
    if kind == "raddino_frozen":
        return CXREmbeddingEncoder(cxr_dim=cfg.cxr_out_dim, out_dim=cfg.d_model)
    raise ValueError(
        f"Unknown cxr_encoder: {kind!r}. Choose 'rad_dino' or 'raddino_frozen'."
    )
