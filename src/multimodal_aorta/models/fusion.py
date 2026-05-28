"""
Cross-attention fusion transformer.

Takes ECG and CXR embeddings (both 768-dim) and fuses them via a small
transformer, returning a single 768-dim fused representation.

Three tokens attend to each other:
  0 — ECG embedding  (or ecg_mask_token when ECG is absent)
  1 — CXR embedding  (or cxr_mask_token when CXR is absent)
  2 — Learned [CLS]  (aggregates both; its output is the fused representation)

Each token gets a learned modality-type embedding added (like BERT segment IDs)
so the transformer knows which modality each token represents.
"""

import torch
import torch.nn as nn


class FusionTransformer(nn.Module):
    """
    Input:
        ecg_emb:  (B, d_model)  — ECG encoder output
        cxr_emb:  (B, d_model)  — CXR encoder output
        has_ecg:  (B,) bool     — True if ECG is present for that sample
        has_cxr:  (B,) bool     — True if CXR is present for that sample
    Output:
        (B, d_model)            — CLS token after transformer fusion
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Learnable modality-type embeddings (analogous to BERT segment embeddings)
        self.ecg_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cxr_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learnable [CLS] token — learns to aggregate both modalities
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learned mask tokens — substituted when a modality is absent at inference
        # or during training-time modality dropout. Initialized to zero; the
        # model learns a meaningful "null" representation through training.
        self.ecg_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cxr_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learnable linear projections (768→768): even when dims match, these
        # provide learned modality-specific alignment transforms.
        self.ecg_proj = nn.Linear(d_model, d_model)
        self.cxr_proj = nn.Linear(d_model, d_model)

        # Transformer encoder (pre-norm for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # expects (B, seq, d_model)
            norm_first=True,    # pre-norm (more stable than post-norm)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.ecg_type_emb, std=0.02)
        nn.init.normal_(self.cxr_type_emb, std=0.02)
        nn.init.normal_(self.cls_type_emb, std=0.02)
        # mask tokens and projections stay at default init

    def forward(
        self,
        ecg_emb: torch.Tensor,   # (B, d_model)
        cxr_emb: torch.Tensor,   # (B, d_model)
        has_ecg: torch.Tensor,   # (B,) bool
        has_cxr: torch.Tensor,   # (B,) bool
    ) -> torch.Tensor:
        B = ecg_emb.size(0)

        # --- Substitute mask tokens for absent modalities (per-sample) ---
        # Expand mask tokens to batch, then replace where modality is absent.
        ecg_mask = self.ecg_mask_token.expand(B, 1, -1)   # (B, 1, d)
        cxr_mask = self.cxr_mask_token.expand(B, 1, -1)   # (B, 1, d)

        ecg_in = ecg_emb.unsqueeze(1)   # (B, 1, d)
        cxr_in = cxr_emb.unsqueeze(1)   # (B, 1, d)

        # Replace with mask token where modality is absent
        has_ecg_ = has_ecg.view(B, 1, 1).to(ecg_emb.dtype)
        has_cxr_ = has_cxr.view(B, 1, 1).to(cxr_emb.dtype)
        ecg_in = ecg_in * has_ecg_ + ecg_mask * (1.0 - has_ecg_)
        cxr_in = cxr_in * has_cxr_ + cxr_mask * (1.0 - has_cxr_)

        # --- Linear projections ---
        ecg_in = self.ecg_proj(ecg_in)   # (B, 1, d)
        cxr_in = self.cxr_proj(cxr_in)   # (B, 1, d)

        # --- Add modality-type embeddings ---
        ecg_in = ecg_in + self.ecg_type_emb
        cxr_in = cxr_in + self.cxr_type_emb

        # --- Prepend [CLS] token ---
        cls = self.cls_token.expand(B, 1, -1) + self.cls_type_emb   # (B, 1, d)

        # Sequence: [CLS, ECG, CXR] — shape (B, 3, d_model)
        tokens = torch.cat([cls, ecg_in, cxr_in], dim=1)

        # --- Transformer (all tokens attend to each other) ---
        out = self.transformer(tokens)   # (B, 3, d_model)

        # Return CLS token (position 0) as the fused representation
        return out[:, 0, :]   # (B, d_model)
