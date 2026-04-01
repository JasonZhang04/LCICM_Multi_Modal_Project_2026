"""
Pre-trained echocardiogram model backbones.

Supports loading:
- PanEcho (view-agnostic multi-task model from Yale, JAMA 2025)
- EchoNet-Dynamic (R2+1D based, trained on EF prediction)

Each backbone returns a feature extractor that maps:
    video tensor (B, C, T, H, W) -> feature vector (B, D)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PanEchoBackbone(nn.Module):
    """
    PanEcho backbone for transfer learning.

    PanEcho is a view-agnostic, multi-task echo model (JAMA 2025).
    We load it with backbone_only=True, which gives us a ConvNeXt-Tiny
    image encoder + temporal Transformer that outputs a 768-dim
    video embedding.

    Key properties:
    - View-agnostic: works with any echo view (PLAX, A4C, etc.)
    - Video-based: processes 16-frame clips (configurable)
    - Input: (B, 3, 16, 224, 224) — ImageNet-normalized
    - Output: (B, 768) embedding

    The model is loaded via PyTorch Hub from the CarDS-Yale/PanEcho repo.
    Weights are downloaded automatically on first use.
    """

    def __init__(self, freeze: bool = True, clip_len: int = 16):
        super().__init__()

        logger.info("Loading PanEcho backbone via PyTorch Hub...")
        self.model = torch.hub.load(
            "CarDS-Yale/PanEcho",
            "PanEcho",
            force_reload=False,
            backbone_only=True,
            clip_len=clip_len,
        )
        self.feature_dim = 768  # PanEcho outputs 768-dim embeddings
        self.clip_len = clip_len

        if freeze:
            self._freeze()

        # Count parameters
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"PanEcho backbone loaded: {total:,} params, {trainable:,} trainable")

    def _freeze(self):
        """Freeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("PanEcho backbone frozen")

    def unfreeze(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone parameters.

        Args:
            num_layers: If None, unfreeze all. Otherwise, unfreeze the last
                        N named children.
        """
        if num_layers is None:
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("Full PanEcho backbone unfrozen")
        else:
            children = list(self.model.named_children())
            for name, module in reversed(children[-num_layers:]):
                for param in module.parameters():
                    param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape (B, C, T, H, W).
               C=3, T=clip_len (default 16), H=W=224.
               Should be ImageNet-normalized.

        Returns:
            Feature vector of shape (B, 768).
        """
        return self.model(x)


class EchoNetDynamicBackbone(nn.Module):
    """
    EchoNet-Dynamic backbone based on R(2+1)D-18.
    Kept as a fallback option.
    """

    def __init__(
        self,
        pretrained_weights_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

        self.model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)

        if pretrained_weights_path:
            self._load_echo_weights(pretrained_weights_path)

        self.feature_dim = self.model.fc.in_features  # 512
        self.model.fc = nn.Identity()

        if freeze:
            self._freeze()

    def _load_echo_weights(self, path: str):
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
            logger.info(f"Loaded EchoNet weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            logger.warning(f"Could not load EchoNet weights: {e}. Using Kinetics-400.")

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("EchoNet backbone frozen")

    def unfreeze(self, num_layers: Optional[int] = None):
        if num_layers is None:
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("Full EchoNet backbone unfrozen")
        else:
            children = list(self.model.named_children())
            for name, module in reversed(children[-num_layers:]):
                for param in module.parameters():
                    param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_backbone(config: dict) -> nn.Module:
    """
    Factory function to load the appropriate backbone.

    Args:
        config: The `model` section of the YAML config.

    Returns:
        A backbone nn.Module with a `feature_dim` attribute.
    """
    backbone_name = config.get("backbone", "panecho")
    freeze = config.get("freeze_backbone", True)

    if backbone_name == "panecho":
        clip_len = config.get("clip_len", 16)
        logger.info(f"Loading PanEcho backbone (clip_len={clip_len})")
        return PanEchoBackbone(freeze=freeze, clip_len=clip_len)

    elif backbone_name == "echonet_dynamic":
        weights_path = config.get("pretrained_weights")
        logger.info("Loading EchoNet-Dynamic backbone (R2+1D-18)")
        return EchoNetDynamicBackbone(
            pretrained_weights_path=weights_path,
            freeze=freeze,
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")