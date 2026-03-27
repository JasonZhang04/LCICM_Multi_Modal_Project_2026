"""
Aortic dilation classifier.

Wraps a pretrained echo backbone (PanEcho or EchoNet) with a
classification head for binary aortic dilation prediction.

For PanEcho, the backbone outputs 768-dim embeddings per video clip.
During training, each DICOM clip gets its own prediction. At study
level, predictions are aggregated (averaged) across all clips in a
study — this mirrors how PanEcho was designed to work.
"""

import logging

import torch
import torch.nn as nn

from models.backbones import load_backbone

logger = logging.getLogger(__name__)


class AorticDilationClassifier(nn.Module):
    """
    Fine-tunable echo classifier for aortic dilation.

    Architecture:
        [Pretrained Echo Backbone] -> [Classification Head]

    The backbone is initially frozen and can be unfrozen after warmup.
    The classification head is a small MLP that maps backbone embeddings
    to binary (dilated/normal) predictions.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The full `model` section of the YAML config.
        """
        super().__init__()

        # Load pretrained backbone
        self.backbone = load_backbone(config)
        feat_dim = self.backbone.feature_dim  # 768 for PanEcho, 512 for EchoNet

        hidden_dim = config.get("hidden_dim", 256)
        dropout = config.get("dropout", 0.3)

        # Determine number of output classes
        label_mode = config.get("label_mode", "binary")
        if label_mode == "binary":
            self.num_classes = 2
        else:
            self.num_classes = 4

        # Classification head — kept simple to avoid overfitting
        # on our small dataset (535 studies)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

        # Track config for unfreezing
        self._freeze_backbone = config.get("freeze_backbone", True)
        self._unfreeze_epoch = config.get("unfreeze_after_epoch", 5)

        logger.info(
            f"AorticDilationClassifier: backbone_dim={feat_dim}, "
            f"hidden={hidden_dim}, classes={self.num_classes}, "
            f"frozen={self._freeze_backbone}"
        )

        # backbone learning rate
        self.backbone_lr = config.get("backbone_lr", 5.0e-6)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single video clip.

        Args:
            video: Tensor of shape (B, C, T, H, W).
                   For PanEcho: C=3, T=16, H=W=224.

        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.backbone(video)  # (B, feat_dim)
        logits = self.head(features)     # (B, num_classes)
        return logits

    def maybe_unfreeze(self, current_epoch: int):
        """
        Unfreeze backbone if we've passed the warmup period.
        Call this at the start of each epoch.
        """
        if self._freeze_backbone and current_epoch >= self._unfreeze_epoch:
            self.backbone.unfreeze()
            self._freeze_backbone = False
            logger.info(f"Backbone unfrozen at epoch {current_epoch}")

    def get_param_groups(self, lr: float) -> list[dict]:
        """
        Get parameter groups with different learning rates.

        Backbone gets a lower LR (lr/10) to prevent catastrophic forgetting.
        Head gets the full LR since it's training from scratch.
        """
        return [
            {"params": self.backbone.parameters(), "lr": self.backbone_lr},
            {"params": self.head.parameters(), "lr": lr},
        ]