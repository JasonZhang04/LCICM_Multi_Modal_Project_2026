"""
ECG encoder interface and implementations.

ECGEncoder (abstract)
  └── ResNet1DEncoder  — 1D ResNet-34, trained from scratch (baseline)
  └── ECGFMEncoder     — ECG-FM via fairseq_signals (stub, future implementation)

All implementations share the same contract:
    Input:  (batch, 12, 5000)  — 12-lead ECG, 10 s at 500 Hz
    Output: (batch, out_dim)   — fixed-size embedding (default out_dim=768)
"""

import abc

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ECGEncoder(nn.Module, abc.ABC):
    """
    Abstract base for ECG encoders.
    All subclasses must produce (batch, out_dim) from (batch, 12, 5000).
    """

    def __init__(self, out_dim: int = 768):
        super().__init__()
        self.out_dim = out_dim

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, 5000) float32 ECG waveform, z-score normalised per lead
        Returns:
            (B, out_dim) float32 embedding
        """


# ---------------------------------------------------------------------------
# Residual block (1D)
# ---------------------------------------------------------------------------

class _ResBlock1D(nn.Module):
    """
    Pre-activation residual block for 1D signals.
    BN → GELU → Conv → BN → GELU → Conv + skip.
    Downsampling via stride in the first conv when stride > 1.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(self.act1(self.bn1(x)))
        x = self.conv2(self.act2(self.bn2(x)))
        return x + residual


# ---------------------------------------------------------------------------
# ResNet-34 1D encoder
# ---------------------------------------------------------------------------

class ResNet1DEncoder(ECGEncoder):
    """
    1D ResNet-34 ECG encoder (trained from scratch).

    Architecture:
        Stem:    Conv1d(12→64, k=15, s=2) + BN + GELU + MaxPool(k=3,s=2)
        Stage 1: 3 × ResBlock(64→64,   k=7, s=1)
        Stage 2: 4 × ResBlock(64→128,  k=7, s=2 on first)
        Stage 3: 6 × ResBlock(128→256, k=7, s=2 on first)
        Stage 4: 3 × ResBlock(256→768, k=7, s=2 on first)
        Head:    GlobalAvgPool → (B, 768)

    Input:  (B, 12, 5000)
    Output: (B, 768)
    """

    def __init__(self, out_dim: int = 768):
        super().__init__(out_dim=out_dim)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )  # → (B, 64, 1250)

        # Stages (channels, n_blocks, stride of first block)
        stage_cfg = [
            (64,  64,  3, 1),
            (64,  128, 4, 2),
            (128, 256, 6, 2),
            (256, out_dim, 3, 2),
        ]
        stages = []
        for in_ch, out_ch, n_blocks, first_stride in stage_cfg:
            blocks = [_ResBlock1D(in_ch, out_ch, kernel=7, stride=first_stride)]
            for _ in range(1, n_blocks):
                blocks.append(_ResBlock1D(out_ch, out_ch, kernel=7, stride=1))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool1d(1)   # → (B, out_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)   # (B, out_dim)
        return x


# ---------------------------------------------------------------------------
# ECG-FM stub (future implementation)
# ---------------------------------------------------------------------------

class ECGFMEncoder(ECGEncoder):
    """
    Placeholder for ECG-FM (foundation model via fairseq_signals).

    To implement: load the pretrained fairseq_signals checkpoint and wrap
    the encoder in this class so it satisfies the ECGEncoder interface.
    The fusion transformer and everything downstream are fully agnostic to
    which concrete encoder is used — only out_dim must match.
    """

    def __init__(self, checkpoint_path: str, out_dim: int = 768):
        super().__init__(out_dim=out_dim)
        self.checkpoint_path = checkpoint_path
        raise NotImplementedError(
            "ECGFMEncoder is not yet implemented. "
            "Install fairseq_signals, load the checkpoint, and implement forward(). "
            "Use ResNet1DEncoder as the baseline in the meantime."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ecg_encoder(cfg) -> ECGEncoder:
    """Instantiate the ECG encoder specified in ModelConfig."""
    if cfg.ecg_encoder == "resnet1d":
        return ResNet1DEncoder(out_dim=cfg.ecg_out_dim)
    if cfg.ecg_encoder == "ecgfm":
        return ECGFMEncoder(
            checkpoint_path=getattr(cfg, "ecgfm_checkpoint", ""),
            out_dim=cfg.ecg_out_dim,
        )
    raise ValueError(f"Unknown ecg_encoder: {cfg.ecg_encoder!r}. Choose 'resnet1d' or 'ecgfm'.")
