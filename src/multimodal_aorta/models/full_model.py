"""
AortaModel: assembles ECG encoder, CXR encoder, fusion transformer, and
regression head into a single nn.Module.

Key responsibilities beyond wiring components:
  - Training-time modality dropout (randomly mask one modality per batch).
  - Per-component parameter groups for AdamW (different LRs per component).
  - Epoch-level CXR encoder freeze/unfreeze scheduling.
"""

import random
from typing import List

import torch
import torch.nn as nn

from multimodal_aorta.configs.default_config import ModelConfig, TrainConfig
from multimodal_aorta.models.ecg_encoder import build_ecg_encoder
from multimodal_aorta.models.cxr_encoder import CXREncoder
from multimodal_aorta.models.fusion import FusionTransformer
from multimodal_aorta.models.regression_head import RegressionHead


class AortaModel(nn.Module):
    """
    End-to-end multimodal aortic diameter prediction model.

    forward() input:
        ecg      : (B, 12, 5000)  — ECG waveform (zeros if ECG absent)
        cxr      : (B, 3, H, W)   — CXR image    (zeros if CXR absent)
        has_ecg  : (B,) bool      — True if the ECG for that sample is real
        has_cxr  : (B,) bool      — True if the CXR for that sample is real

    forward() output:
        (B, 2) float32            — [aortic_root_cm, ascending_aorta_cm]
    """

    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.ecg_encoder = build_ecg_encoder(model_cfg)

        self.cxr_encoder = CXREncoder(
            model_name=model_cfg.cxr_model_name,
            freeze_blocks=model_cfg.cxr_freeze_blocks,
        )

        self.fusion = FusionTransformer(
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            num_layers=model_cfg.num_fusion_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            dropout=model_cfg.fusion_dropout,
        )

        self.head = RegressionHead(
            in_dim=model_cfg.d_model,
            hidden_dim=model_cfg.head_hidden_dim,
            dropout=model_cfg.head_dropout,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        ecg: torch.Tensor,
        cxr: torch.Tensor,
        has_ecg: torch.Tensor,
        has_cxr: torch.Tensor,
    ) -> torch.Tensor:
        # Training-time modality dropout: randomly suppress one modality
        # so the model learns to function with incomplete inputs.
        if self.training:
            has_ecg, has_cxr = self._apply_modality_dropout(has_ecg, has_cxr)

        ecg_emb = self.ecg_encoder(ecg)           # (B, 768)
        cxr_emb = self.cxr_encoder(cxr)           # (B, 768)

        fused = self.fusion(ecg_emb, cxr_emb, has_ecg, has_cxr)  # (B, 768)
        return self.head(fused)                    # (B, 2)

    def _apply_modality_dropout(
        self,
        has_ecg: torch.Tensor,
        has_cxr: torch.Tensor,
    ):
        """
        Independently suppress each modality with probability p_drop_*.
        Never suppresses both modalities for the same sample.
        Returns updated has_ecg, has_cxr boolean tensors.
        """
        p_ecg = self.train_cfg.p_drop_ecg
        p_cxr = self.train_cfg.p_drop_cxr

        has_ecg = has_ecg.clone()
        has_cxr = has_cxr.clone()

        B = has_ecg.size(0)
        for i in range(B):
            drop_ecg = has_ecg[i] and random.random() < p_ecg
            drop_cxr = has_cxr[i] and random.random() < p_cxr
            # Ensure we never drop both
            if drop_ecg and drop_cxr:
                if random.random() < 0.5:
                    drop_ecg = False
                else:
                    drop_cxr = False
            if drop_ecg:
                has_ecg[i] = False
            if drop_cxr:
                has_cxr[i] = False

        return has_ecg, has_cxr

    # ------------------------------------------------------------------
    # Optimizer parameter groups
    # ------------------------------------------------------------------

    def get_param_groups(self) -> List[dict]:
        """
        Returns AdamW parameter groups with per-component learning rates.

            ECG encoder:  lr_ecg_encoder  (trains from scratch — higher LR ok)
            CXR encoder:  lr_cxr_encoder  (pretrained — conservative LR)
            Fusion + head: lr_fusion      (new layers — highest LR)
        """
        tc = self.train_cfg
        return [
            {
                "params": list(self.ecg_encoder.parameters()),
                "lr": tc.lr_ecg_encoder,
                "name": "ecg_encoder",
            },
            {
                "params": list(self.cxr_encoder.parameters()),
                "lr": tc.lr_cxr_encoder,
                "name": "cxr_encoder",
            },
            {
                "params": (
                    list(self.fusion.parameters()) + list(self.head.parameters())
                ),
                "lr": tc.lr_fusion,
                "name": "fusion_head",
            },
        ]

    # ------------------------------------------------------------------
    # CXR freeze scheduling (called from training loop each epoch)
    # ------------------------------------------------------------------

    def on_epoch_start(self, epoch: int) -> None:
        """Call at the beginning of each epoch to apply freeze schedule."""
        self.cxr_encoder.set_frozen_for_epoch(epoch, self.model_cfg.cxr_unfreeze_epoch)
