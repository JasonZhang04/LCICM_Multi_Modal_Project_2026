"""
EchoDataset: loads MIMIC-IV echocardiography DICOMs for single-modality
aortic diameter regression.

Each item corresponds to one patient study and returns:
    video    : (N_CLIPS, 3, N_FRAMES, IMG_SIZE, IMG_SIZE) float32
    target   : (2,) float32  [root_cm, asc_cm], NaN where label missing
    subject_id : int

Requires pylibjpeg for JPEG-compressed DICOMs:
    pip install pylibjpeg pylibjpeg-libjpeg
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Default clip / frame sampling parameters
# PanEcho was trained on 16-frame clips at 224×224 — use same defaults
N_CLIPS   = 8
N_FRAMES  = 16
IMG_SIZE  = 224


# ---------------------------------------------------------------------------
# Cohort building
# ---------------------------------------------------------------------------

def build_echo_cohort(
    echo_labels_path: str,
    echo_structured_path: str,
    echo_study_list: str,
    echo_record_list: str,
    col_root: str = "sinus_diam_cm",
    col_asc: str  = "ascending_diam_cm",
    root_range: Tuple[float, float] = (1.5, 6.0),
    asc_range:  Tuple[float, float] = (1.5, 7.0),
    max_days_offset: int = 365,
) -> pd.DataFrame:
    """
    Build a per-patient DataFrame linking aortic labels to echo study IDs.

    Columns returned:
        subject_id, study_id, target_root, target_asc, echo_date, study_date

    Strategy: for each patient, pick the echo study whose study_datetime
    is closest to their aortic measurement date (within max_days_offset).
    """
    # ── Labels ──────────────────────────────────────────────────────────────
    labels = pd.read_csv(echo_labels_path)
    labels = labels.rename(columns={col_root: "target_root", col_asc: "target_asc"})

    # Unit check
    for col in ["target_root", "target_asc"]:
        if labels[col].median() > 20:
            labels[col] /= 10.0

    # Drop rows with both targets missing
    both_nan = labels["target_root"].isna() & labels["target_asc"].isna()
    labels = labels[~both_nan].copy()

    # Clip outliers
    for col, (lo, hi) in [("target_root", root_range), ("target_asc", asc_range)]:
        out = labels[col].notna() & ((labels[col] < lo) | (labels[col] > hi))
        labels.loc[out, col] = np.nan
    both_nan2 = labels["target_root"].isna() & labels["target_asc"].isna()
    labels = labels[~both_nan2].copy()

    # ── Echo measurement dates ───────────────────────────────────────────────
    struct = pd.read_csv(
        echo_structured_path,
        usecols=["subject_id", "measurement_datetime"],
    )
    echo_dates = (
        struct.groupby("subject_id")["measurement_datetime"]
        .min()
        .reset_index()
        .rename(columns={"measurement_datetime": "echo_date"})
    )
    echo_dates["echo_date"] = pd.to_datetime(echo_dates["echo_date"])
    labels = labels.merge(echo_dates, on="subject_id", how="left")

    # ── Echo studies ─────────────────────────────────────────────────────────
    studies = pd.read_csv(echo_study_list, usecols=["subject_id", "study_id", "study_datetime"])
    studies["study_date"] = pd.to_datetime(studies["study_datetime"])

    # ── Match each patient to best echo study ────────────────────────────────
    rows = []
    for _, lab in labels.iterrows():
        sid = lab["subject_id"]
        pat_studies = studies[studies["subject_id"] == sid].copy()
        if pat_studies.empty:
            continue

        echo_date = lab["echo_date"]
        if pd.isna(echo_date):
            # No measurement date — use most recent study
            best = pat_studies.sort_values("study_date").iloc[-1]
        else:
            pat_studies["days_off"] = (
                (pat_studies["study_date"] - echo_date)
                .dt.total_seconds().abs() / 86400
            )
            within = pat_studies[pat_studies["days_off"] <= max_days_offset]
            if within.empty:
                best = pat_studies.loc[pat_studies["days_off"].idxmin()]
            else:
                best = within.loc[within["days_off"].idxmin()]

        rows.append({
            "subject_id":  sid,
            "study_id":    int(best["study_id"]),
            "target_root": lab["target_root"],
            "target_asc":  lab["target_asc"],
            "echo_date":   echo_date,
            "study_date":  best["study_date"],
        })

    cohort = pd.DataFrame(rows).reset_index(drop=True)
    logger.info(
        "Echo cohort: %d patients with study linkage (from %d labels)",
        len(cohort), len(labels),
    )

    # ── Attach clip lists (records per study) ────────────────────────────────
    records = pd.read_csv(
        echo_record_list,
        usecols=["subject_id", "study_id", "dicom_filepath"],
    )
    clips_by_study = (
        records.groupby("study_id")["dicom_filepath"]
        .apply(list)
        .to_dict()
    )
    cohort["clips"] = cohort["study_id"].map(clips_by_study).apply(
        lambda x: x if isinstance(x, list) else []
    )

    n_empty = (cohort["clips"].apply(len) == 0).sum()
    if n_empty:
        logger.warning("%d patients have no clips in study — they will return zeros", n_empty)

    logger.info(
        "Clips per study — median: %.0f  mean: %.1f",
        cohort["clips"].apply(len).median(),
        cohort["clips"].apply(len).mean(),
    )

    return cohort


# ---------------------------------------------------------------------------
# DICOM clip loading
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _load_clip_frames(
    dicom_path: str,
    n_frames: int,
    img_size: int,
) -> Optional[torch.Tensor]:
    """
    Load one echo DICOM clip and return (3, n_frames, img_size, img_size) float32.
    Returns None on failure.

    Output is ImageNet-normalized (mean/std of [0.485,0.456,0.406] / [0.229,0.224,0.225])
    to match PanEcho's ConvNeXt-Tiny backbone training distribution.

    Requires pylibjpeg for JPEG-compressed DICOMs:
        pip install pylibjpeg pylibjpeg-libjpeg
    """
    try:
        import pydicom
        from PIL import Image

        ds = pydicom.dcmread(dicom_path)
        px = ds.pixel_array   # (T, H, W), (T, H, W, C), or (H, W)
    except Exception as e:
        logger.debug("DICOM load failed %s: %s", dicom_path, e)
        return None

    if px.ndim == 2:
        px = px[np.newaxis]     # (H, W) → (1, H, W)
    T = px.shape[0]

    # Sample n_frames uniformly across the clip
    indices = np.linspace(0, T - 1, n_frames, dtype=int)
    frames = px[indices]       # (n_frames, H, W) or (n_frames, H, W, C)

    # Resize each frame to img_size × img_size and produce (3, n_frames, H, W)
    # Handles both grayscale (H, W) and RGB (H, W, 3) DICOM pixel arrays.
    out3 = np.zeros((3, n_frames, img_size, img_size), dtype=np.float32)
    for i, f in enumerate(frames):
        img = Image.fromarray(f.astype(np.uint8)).convert("RGB").resize(
            (img_size, img_size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        out3[:, i, :, :] = arr.transpose(2, 0, 1)         # (3, H, W)

    # ImageNet normalize per channel
    for c in range(3):
        out3[c] = (out3[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]

    return torch.from_numpy(out3)  # (3, n_frames, H, W)


def _sample_clips(clips: List[str], n: int) -> List[str]:
    """Return n clips sampled uniformly across the sorted clip list."""
    if len(clips) <= n:
        return clips
    indices = np.linspace(0, len(clips) - 1, n, dtype=int)
    return [clips[i] for i in indices]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EchoDataset(Dataset):
    """
    Single-modality echocardiography dataset for aortic diameter regression.

    Each item returns one patient's echo study as a stack of N_CLIPS × N_FRAMES
    individual grayscale frames (normalized to [0, 1]).
    The 2D backbone in EchoModel encodes each frame independently.
    """

    def __init__(
        self,
        cohort: pd.DataFrame,
        echo_root: str,
        n_clips: int = N_CLIPS,
        n_frames: int = N_FRAMES,
        img_size: int = IMG_SIZE,
        target_stats=None,
    ):
        """
        Parameters
        ----------
        cohort     : output of build_echo_cohort(), filtered to a split
        echo_root  : root directory of Echo_Data (files/ is a sub-dir)
        n_clips    : clips to sample per study
        n_frames   : frames to sample per clip
        img_size   : spatial resolution (square)
        target_stats : TargetStats for z-scoring; pass None for raw cm
        """
        self.cohort = cohort.reset_index(drop=True)
        self.echo_root = echo_root
        self.n_clips = n_clips
        self.n_frames = n_frames
        self.img_size = img_size
        self.target_stats = target_stats

        self._zero_clip = torch.zeros(3, n_frames, img_size, img_size)

    def __len__(self) -> int:
        return len(self.cohort)

    def __getitem__(self, idx: int) -> Dict:
        row = self.cohort.iloc[idx]

        # ── Sample clips ───────────────────────────────────────────────────
        clips = row["clips"]
        selected = _sample_clips(clips, self.n_clips)

        clip_tensors = []
        for rel_path in selected:
            # Record list stores paths as "files/p10/..." but echo_root already
            # points to the directory that contains p10/, p11/, etc. directly.
            rel_path = rel_path.removeprefix("files/")
            full_path = os.path.join(self.echo_root, rel_path)
            t = _load_clip_frames(full_path, self.n_frames, self.img_size)
            clip_tensors.append(t if t is not None else self._zero_clip)

        # Pad with zeros if fewer clips than requested
        while len(clip_tensors) < self.n_clips:
            clip_tensors.append(self._zero_clip)

        video = torch.stack(clip_tensors)   # (N_CLIPS, 3, N_FRAMES, H, W)

        # ── Target ─────────────────────────────────────────────────────────
        target = torch.tensor(
            [row["target_root"], row["target_asc"]], dtype=torch.float32
        )
        if self.target_stats is not None:
            target = self.target_stats.normalize(target)

        return {
            "video":      video,
            "target":     target,
            "subject_id": int(row["subject_id"]),
        }
