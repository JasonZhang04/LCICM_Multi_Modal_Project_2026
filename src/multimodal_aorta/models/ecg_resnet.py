"""
ECGAI-TAA-style 1D ResNet for aortic-diameter regression from the raw 12-lead ECG.

Reimplemented from Demarais/Olgin/Pirruccello, "Estimating Ascending Aortic Diameter
From the Electrocardiogram" (JACC 2026;87(1):105-108 / medRxiv 2025.08.19.25333786),
since no weights are publicly released. Trained from scratch on MIMIC-IV-ECG against
our own TTE labels (so no MRI/UK-Biobank domain gap).

Faithful to the described architecture:
  input 10 s, 12-lead, 500 Hz (12 x 5000), left in mV with no preprocessing;
  stem conv (kernel 15); three residual stages with group normalization and
  downsampling between stages; dilations in alternating blocks; a residual
  multi-head self-attention block; global average pooling; dropout; a linear head.
Our head predicts [root diameter, ascending diameter, heart rate] (HR auxiliary),
all MSE on normalized targets.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1, dilation=1, k=9):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.gn1 = _gn(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, stride=1, padding=k // 2, bias=False)
        self.gn2 = _gn(cout)
        self.act = nn.GELU()
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), _gn(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + idt)


class AttnBlock(nn.Module):
    """Residual multi-head self-attention over the temporal sequence."""
    def __init__(self, c, heads=4, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, dropout=drop, batch_first=True)

    def forward(self, x):                      # x: (B, C, T)
        h = x.transpose(1, 2)                  # (B, T, C)
        hn = self.norm(h)
        a, _ = self.attn(hn, hn, hn, need_weights=False)
        return (h + a).transpose(1, 2)


class ECGResNet(nn.Module):
    def __init__(self, n_leads=12, channels=(64, 128, 256), blocks=2,
                 n_out=3, dropout=0.3, heads=4):
        super().__init__()
        c0 = channels[0]
        # stem (kernel 15) + a maxpool to cut the 5000-long sequence early
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, c0, 15, stride=2, padding=7, bias=False),
            _gn(c0), nn.GELU(), nn.MaxPool1d(2))          # 5000 -> 1250
        stages, cin = [], c0
        for si, cout in enumerate(channels):
            for bi in range(blocks):
                stride = 2 if (bi == 0 and si > 0) else 1   # downsample at the start of stages 2,3
                dil = 2 if (bi % 2 == 1) else 1             # dilations in alternating blocks
                stages.append(ResBlock1D(cin, cout, stride=stride, dilation=dil))
                cin = cout
        self.stages = nn.Sequential(*stages)
        self.attn = AttnBlock(channels[-1], heads=heads)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], n_out)

    def forward(self, x):                        # x: (B, 12, 5000)
        x = self.stem(x)
        x = self.stages(x)
        x = self.attn(x)
        x = self.pool(x).squeeze(-1)             # (B, C)
        return self.head(self.drop(x))           # (B, n_out)
