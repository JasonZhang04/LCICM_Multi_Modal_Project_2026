"""
EchoModel: PanEcho encoder + clip pooling + regression head.

Encoder choice
--------------
Primary:  PanEcho (CarDS-Yale/PanEcho) — the right choice for this task.
  - ConvNeXt-Tiny image encoder + 4-layer Frame Transformer per clip
  - Pretrained on Yale echo cohort with 39 echo reporting tasks including
    AORoot diameter — i.e. the encoder has explicitly learned aortic anatomy
  - View-agnostic: works across PLAX, A4C, PSAX, etc.
  - Input: (B, 3, 16, 224, 224) per clip  →  output: (B, 768)
  - Weights already cached at ~/.cache/torch/hub/checkpoints/panecho.pt
  - Code already cached at ~/.cache/torch/hub/CarDS-Yale_PanEcho_main/

Backup:   ResNet-50 (ImageNet) + mean-pool over frames — no download needed.
Future:   Echo-FM (SekeunKim/EchoFM) when weights are downloaded.

Architecture
------------
    N_CLIPS clips × (3, 16, 224, 224)
        ↓  PanEcho FrameTransformer (per clip)
    N_CLIPS × (768,) clip embeddings
        ↓  mean-pool over N_CLIPS
    (768,) study embedding
        ↓  Linear(768→256) → GELU → Dropout → Linear(256→2)
    [root_cm, asc_cm]
"""

import logging
import os
import sys
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Respect TORCH_HOME if set; otherwise default to a persistent scratch location
# so the hub cache survives home-directory quota cleanups.
_TORCH_HUB_BASE = os.path.join(
    os.environ.get("TORCH_HOME", "/scratch4/rsteven1/torch_hub"), "hub"
)
PANECHO_HUB_DIR = os.path.join(_TORCH_HUB_BASE, "CarDS-Yale_PanEcho_main")
PANECHO_CKPT    = os.path.join(_TORCH_HUB_BASE, "checkpoints", "panecho.pt")


# ---------------------------------------------------------------------------
# PanEcho encoder wrapper
# ---------------------------------------------------------------------------

class PanEchoEncoder(nn.Module):
    """
    Loads the PanEcho backbone (FrameTransformer, backbone_only=True) from the
    locally cached repo and checkpoint — no internet required.

    Input:  (B, 3, clip_len, 224, 224)  float32, ImageNet-normalized
    Output: (B, 768)
    """

    OUT_DIM   = 768
    CLIP_LEN  = 16
    IMG_SIZE  = 224

    def __init__(self, ckpt_path: str = PANECHO_CKPT, clip_len: int = CLIP_LEN):
        super().__init__()

        # Import FrameTransformer from the locally-cached PanEcho repo
        if PANECHO_HUB_DIR not in sys.path:
            sys.path.insert(0, PANECHO_HUB_DIR)
        from src.models import FrameTransformer   # noqa: E402

        self.frame_transformer = FrameTransformer(
            arch="convnext_tiny",
            n_heads=8,
            n_layers=4,
            transformer_dropout=0.0,
            pooling="mean",
            clip_len=clip_len,
        )

        # Load pretrained encoder weights from the checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Checkpoint has 'encoder.*' (FrameTransformer) and '*_head' (task heads).
        # We want only encoder weights, with the 'encoder.' prefix stripped.
        encoder_weights = {
            k[len("encoder."):]: v
            for k, v in ckpt["weights"].items()
            if k.startswith("encoder.")
            and k != "encoder.time_encoder.pe"   # positional encoding is re-init'd
        }
        missing, unexpected = self.frame_transformer.load_state_dict(
            encoder_weights, strict=False
        )
        real_missing = [k for k in missing if "time_encoder.pe" not in k]
        if real_missing:
            logger.warning("PanEcho: missing keys after weight load: %s", real_missing)
        if unexpected:
            logger.warning("PanEcho: unexpected keys: %s", unexpected)
        logger.info("PanEchoEncoder loaded from %s", ckpt_path)

        self.out_dim = self.OUT_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, clip_len, 224, 224)
        return self.frame_transformer(x)   # (B, 768)


# ---------------------------------------------------------------------------
# ResNet-50 fallback (no domain-specific pretraining, no download needed)
# ---------------------------------------------------------------------------

class ResNet50Encoder(nn.Module):
    """
    ResNet-50 (ImageNet) 2D per-frame encoder.
    Each frame is encoded independently; mean-pool over all frames gives the
    clip embedding. Less suitable than PanEcho for echo but available without
    any echo-specific weights.

    Input:  (B, 3, T, H, W) — processes each of the T frames independently
    Output: (B, 2048)
    """

    OUT_DIM = 2048

    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            m = tvm.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # (B, 2048, 1, 1)
        self.out_dim = self.OUT_DIM
        logger.info("ResNet50Encoder loaded (ImageNet pretrained)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, H, W) — encode each frame, then mean-pool
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, 3, H, W)
        feat = self.backbone(x)                                  # (B*T, 2048, 1, 1)
        feat = feat.view(B, T, -1)                               # (B, T, 2048)
        return feat.mean(dim=1)                                  # (B, 2048)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class EchoModel(nn.Module):
    """
    Single-modality echocardiography regression model.

    Input:  video (B, N_clips, 3, T, H, W)
    Output: (B, 2)  —  [aortic_root_cm, ascending_aorta_cm]
    """

    def __init__(
        self,
        backbone: str = "panecho",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone_epochs: int = 5,
        panecho_ckpt: str = PANECHO_CKPT,
    ):
        """
        Parameters
        ----------
        backbone : "panecho" (default) | "resnet50"
        hidden_dim : regression head hidden dimension
        dropout : dropout in regression head
        freeze_backbone_epochs : freeze backbone for this many epochs, then unfreeze
            with lr_backbone to prevent catastrophic forgetting
        panecho_ckpt : path to panecho.pt (defaults to hub cache location)
        """
        super().__init__()
        self.backbone_name = backbone
        self.freeze_backbone_epochs = freeze_backbone_epochs

        if backbone == "panecho":
            self.encoder = PanEchoEncoder(ckpt_path=panecho_ckpt)
        elif backbone == "resnet50":
            self.encoder = ResNet50Encoder()
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}. Choose 'panecho' or 'resnet50'.")

        feat_dim = self.encoder.out_dim

        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self._backbone_frozen = False

    # ------------------------------------------------------------------
    # Freeze schedule (called each epoch from training loop)
    # ------------------------------------------------------------------

    def on_epoch_start(self, epoch: int) -> None:
        should_freeze = (epoch < self.freeze_backbone_epochs)
        if should_freeze and not self._backbone_frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self._backbone_frozen = True
            logger.info("Epoch %d: echo backbone frozen", epoch)
        elif not should_freeze and self._backbone_frozen:
            for p in self.encoder.parameters():
                p.requires_grad = True
            self._backbone_frozen = False
            logger.info("Epoch %d: echo backbone unfrozen", epoch)

    # ------------------------------------------------------------------
    # Parameter groups for AdamW
    # ------------------------------------------------------------------

    def get_param_groups(
        self,
        lr_backbone: float = 5e-6,
        lr_head: float = 1e-4,
    ) -> List[dict]:
        return [
            {"params": list(self.encoder.parameters()), "lr": lr_backbone, "name": "backbone"},
            {"params": list(self.head.parameters()),    "lr": lr_head,     "name": "head"},
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, N_clips, 3, T, H, W)  — ImageNet-normalized float32
        Returns:
            (B, 2)
        """
        B, N, C, T, H, W = video.shape

        # Process all clips in one batched forward pass
        x = video.view(B * N, C, T, H, W)   # (B*N, 3, T, H, W)
        feat = self.encoder(x)               # (B*N, feat_dim)
        feat = feat.view(B, N, -1)           # (B, N, feat_dim)

        # Mean-pool over clips → study-level embedding
        study_emb = feat.mean(dim=1)         # (B, feat_dim)

        return self.head(study_emb)          # (B, 2)
