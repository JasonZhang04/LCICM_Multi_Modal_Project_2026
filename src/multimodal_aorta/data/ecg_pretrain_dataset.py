"""
Dataset for SimCLR self-supervised ECG pretraining.

Loads all 800k ECG records from MIMIC-IV-ECG and returns two independently
augmented views of each waveform for contrastive learning.

Augmentation pipeline per view (applied in order):
  1. Time crop  — random 4000-sample window, zero-pad tail
  2. Amplitude scale — U[0.7, 1.3] per lead
  3. Gaussian noise  — σ = U[0, 0.02] × lead std (non-padded region only)
  4. Baseline wander — low-frequency sinusoid f ∈ U[0.05, 1] Hz (non-padded region)
  5. Lead masking    — zero out 1–2 randomly chosen leads
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_TARGET_FS  = 500    # Hz
_TARGET_LEN = 5000   # samples (10 s)
_N_LEADS    = 12
_CROP_LEN   = 4000   # kept samples per view


def _load_raw_ecg(path: str) -> Optional[np.ndarray]:
    """
    Load a WFDB record and return (12, 5000) float32, or None on failure.
    Resamples to 500 Hz, crops/pads to 5000 samples, z-scores per lead.
    Same logic as preprocessing.load_ecg but without DataConfig dependency.
    """
    try:
        import wfdb
        record = wfdb.rdrecord(path)
        signal = record.p_signal  # (n_samples, n_leads)
        fs = record.fs
    except Exception:
        return None

    if signal is None or signal.ndim != 2:
        return None

    if fs != _TARGET_FS:
        from scipy.signal import resample
        n_target = int(round(signal.shape[0] * _TARGET_FS / fs))
        signal = resample(signal, n_target, axis=0)

    n_samples = signal.shape[0]
    if n_samples < _TARGET_LEN:
        pad = np.zeros((_TARGET_LEN - n_samples, signal.shape[1]), dtype=np.float32)
        signal = np.concatenate([signal, pad], axis=0)
    elif n_samples > _TARGET_LEN:
        start = (n_samples - _TARGET_LEN) // 2
        signal = signal[start : start + _TARGET_LEN]

    out = np.zeros((_N_LEADS, _TARGET_LEN), dtype=np.float32)
    n_leads = min(signal.shape[1], _N_LEADS)
    for i in range(n_leads):
        lead = signal[:, i].astype(np.float32)
        nan_mask = np.isnan(lead)
        if nan_mask.all():
            continue
        if nan_mask.any():
            lead[nan_mask] = 0.0
        mu = np.nanmean(lead)
        std = np.nanstd(lead) + 1e-8
        lead_clip = np.clip(lead, mu - 4.0 * std, mu + 4.0 * std)
        mu2 = lead_clip.mean()
        std2 = lead_clip.std() + 1e-8
        out[i] = (lead - mu2) / std2

    return out


def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply one set of SimCLR augmentations to a (12, 5000) ECG array.
    Calling this twice with the same rng produces two different views
    because the rng state advances between calls.
    """
    x = x.copy()
    n_leads = x.shape[0]

    # 1. Time crop — random window of _CROP_LEN samples, zero-pad tail
    start = int(rng.integers(0, _TARGET_LEN - _CROP_LEN + 1))
    cropped = x[:, start : start + _CROP_LEN].copy()
    x = np.zeros((_N_LEADS, _TARGET_LEN), dtype=np.float32)
    x[:, :_CROP_LEN] = cropped

    # 2. Per-lead amplitude scale ∈ U[0.7, 1.3] — zeros in tail stay 0
    scales = rng.uniform(0.7, 1.3, size=(n_leads, 1)).astype(np.float32)
    x = x * scales

    # 3. Gaussian noise — applied to non-padded region only
    for i in range(n_leads):
        lead_std = float(x[i, :_CROP_LEN].std()) + 1e-8
        sigma = float(rng.uniform(0.0, 0.02)) * lead_std
        if sigma > 0:
            x[i, :_CROP_LEN] += rng.normal(0.0, sigma, size=_CROP_LEN).astype(np.float32)

    # 4. Baseline wander — sinusoid on non-padded region
    t = np.arange(_CROP_LEN, dtype=np.float32) / _TARGET_FS
    for i in range(n_leads):
        lead_std = float(x[i, :_CROP_LEN].std()) + 1e-8
        freq  = float(rng.uniform(0.05, 1.0))
        amp   = float(rng.uniform(0.0, 0.1)) * lead_std
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        x[i, :_CROP_LEN] += (amp * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)

    # 5. Lead masking — zero out 1 or 2 leads entirely
    n_drop = int(rng.integers(1, 3))
    drop_leads = rng.choice(n_leads, size=n_drop, replace=False)
    x[drop_leads] = 0.0

    return x


class ECGPretrainDataset(Dataset):
    """
    SimCLR pretraining dataset over all MIMIC-IV-ECG records.

    Each item returns (view1, view2): two differently-augmented tensors of
    shape (12, 5000) float32. If a waveform fails to load, zeros are returned.

    Args:
        ecg_root:    Root directory of the MIMIC-IV-ECG dataset.
        record_list: Path to record_list.csv (must have a `path` column with
                     paths relative to ecg_root, no extension — WFDB adds .hea/.dat).
        max_records: Subsample to this many records (for debugging).
    """

    def __init__(
        self,
        ecg_root: str,
        record_list: str,
        max_records: Optional[int] = None,
    ):
        df = pd.read_csv(record_list, usecols=["path"])
        if max_records is not None:
            df = df.iloc[:max_records].reset_index(drop=True)
        self._paths = [os.path.join(ecg_root, p) for p in df["path"]]
        logger.info(
            "ECGPretrainDataset: %d records from %s",
            len(self._paths), ecg_root,
        )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _load_raw_ecg(self._paths[idx])
        if x is None:
            x = np.zeros((_N_LEADS, _TARGET_LEN), dtype=np.float32)

        # Single RNG; two calls to _augment advance state → different views
        rng = np.random.default_rng()
        v1 = _augment(x, rng)
        v2 = _augment(x, rng)

        return torch.from_numpy(v1), torch.from_numpy(v2)
