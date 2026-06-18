"""
AortaDataset: links echo labels → MIMIC IV-ECG → MIMIC-CXR,
applies QC, temporal alignment, and exposes a modality-masked
PyTorch Dataset.

Cohort construction is done once (build_cohort) and the resulting
DataFrame is stored on the Dataset instance. The actual waveform /
image loading happens in __getitem__ via the preprocessing module.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target normalization (Step 1 redesign)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetStats:
    """Per-target mean and std (cm), computed on the train split only."""
    root_mean: float
    root_std: float
    asc_mean: float
    asc_std: float

    def normalize(self, target: torch.Tensor) -> torch.Tensor:
        """Z-score a (..., 2) tensor of [root, asc] values. Preserves NaN."""
        means = torch.tensor([self.root_mean, self.asc_mean], dtype=target.dtype)
        stds = torch.tensor([self.root_std, self.asc_std], dtype=target.dtype)
        means = means.to(target.device)
        stds = stds.to(target.device)
        return (target - means) / stds

    def denormalize(self, target: torch.Tensor) -> torch.Tensor:
        """Inverse of normalize(): map z-scored values back to cm."""
        means = torch.tensor([self.root_mean, self.asc_mean], dtype=target.dtype)
        stds = torch.tensor([self.root_std, self.asc_std], dtype=target.dtype)
        means = means.to(target.device)
        stds = stds.to(target.device)
        return target * stds + means


def compute_target_stats(cohort: pd.DataFrame) -> TargetStats:
    """
    Compute z-score statistics on the train cohort.

    NaN entries are ignored (each target's mean/std is computed over its own
    non-null values). The cohort is expected to have already been QC-clipped
    and renamed to use the `target_root` / `target_asc` columns.
    """
    root = cohort["target_root"].to_numpy(dtype=np.float64)
    asc = cohort["target_asc"].to_numpy(dtype=np.float64)

    root_valid = root[~np.isnan(root)]
    asc_valid = asc[~np.isnan(asc)]

    if len(root_valid) < 2 or len(asc_valid) < 2:
        raise ValueError(
            f"Not enough valid targets to compute stats "
            f"(root n={len(root_valid)}, asc n={len(asc_valid)})"
        )

    return TargetStats(
        root_mean=float(root_valid.mean()),
        root_std=float(root_valid.std(ddof=0)),
        asc_mean=float(asc_valid.mean()),
        asc_std=float(asc_valid.std(ddof=0)),
    )


# ---------------------------------------------------------------------------
# Cohort-building helpers
# ---------------------------------------------------------------------------

def _load_echo_labels(
    labels_path: str,
    structured_path: str,
    col_root: str,
    col_asc: str,
    root_range: Tuple[float, float],
    asc_range: Tuple[float, float],
) -> pd.DataFrame:
    """
    Load aortic_labels.csv, join echo study dates from structured-measurement.csv,
    apply unit checks and outlier clipping.

    Returns DataFrame with columns:
      subject_id, target_root, target_asc, echo_date
    """
    labels = pd.read_csv(labels_path)
    logger.info("Echo labels loaded: %d rows, columns: %s", len(labels), labels.columns.tolist())
    logger.info("Label dtypes:\n%s", labels.dtypes.to_string())
    logger.info("Label statistics:\n%s", labels[[col_root, col_asc]].describe().to_string())

    n_start = len(labels)

    # --- Unit check: if median > 20, values are likely in mm ---
    for col, name in [(col_root, "root"), (col_asc, "ascending")]:
        median_val = labels[col].median()
        if median_val > 20:
            logger.warning(
                "%s median=%.1f looks like mm; dividing by 10 to convert to cm", name, median_val
            )
            labels[col] = labels[col] / 10.0

    # --- Drop rows where BOTH targets are NaN ---
    both_nan = labels[col_root].isna() & labels[col_asc].isna()
    n_both_nan = both_nan.sum()
    labels = labels[~both_nan].reset_index(drop=True)
    logger.info("Dropped %d rows with both targets NaN. Remaining: %d", n_both_nan, len(labels))

    # --- Clip outliers ---
    for col, name, (lo, hi) in [
        (col_root, "root", root_range),
        (col_asc,  "asc",  asc_range),
    ]:
        valid = labels[col].notna()
        out_of_range = valid & ((labels[col] < lo) | (labels[col] > hi))
        n_outlier = out_of_range.sum()
        if n_outlier > 0:
            logger.warning(
                "Clipping %d %s values outside [%.1f, %.1f] cm to NaN",
                n_outlier, name, lo, hi,
            )
            labels.loc[out_of_range, col] = np.nan

    # After outlier clipping, drop rows where both are now NaN
    both_nan2 = labels[col_root].isna() & labels[col_asc].isna()
    labels = labels[~both_nan2].reset_index(drop=True)
    logger.info(
        "After outlier clipping: %d rows remain (dropped %d more)",
        len(labels),
        n_start - len(labels),
    )

    # Rename target columns for clarity
    labels = labels.rename(columns={col_root: "target_root", col_asc: "target_asc"})

    # --- Join echo study dates from structured measurements ---
    struct = pd.read_csv(structured_path, usecols=["subject_id", "measurement_id", "measurement_datetime"])
    # Each measurement_id corresponds to one echo study; take the earliest date per patient
    # (aortic_labels.csv was pre-filtered to one row per patient, so we want the
    # measurement_datetime that corresponds to the aortic measurement — use min date
    # per patient as the best proxy for the echo study date)
    echo_dates = (
        struct.groupby("subject_id")["measurement_datetime"]
        .min()
        .reset_index()
        .rename(columns={"measurement_datetime": "echo_date"})
    )
    echo_dates["echo_date"] = pd.to_datetime(echo_dates["echo_date"])

    labels = labels.merge(echo_dates, on="subject_id", how="left")
    n_no_date = labels["echo_date"].isna().sum()
    logger.info(
        "Echo dates joined: %d/%d patients have a date",
        len(labels) - n_no_date,
        len(labels),
    )

    return labels[["subject_id", "target_root", "target_asc", "echo_date"]]


def _load_ecg_index(ecg_root: str) -> pd.DataFrame:
    """
    Load MIMIC IV-ECG record list.
    Returns DataFrame: subject_id (int), study_id, ecg_time (datetime), ecg_path (str).
    ecg_path is the full absolute path prefix (without .hea/.dat extension).
    """
    record_list_path = os.path.join(ecg_root, "record_list.csv")
    ecg = pd.read_csv(record_list_path)
    ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"])
    # Construct full path
    ecg["ecg_path"] = ecg["path"].apply(
        lambda p: os.path.join(ecg_root, p)
    )
    logger.info("ECG record list loaded: %d records for %d patients",
                len(ecg), ecg["subject_id"].nunique())
    return ecg[["subject_id", "study_id", "ecg_time", "ecg_path"]]


def _load_mimic_cxr_metadata(metadata_csv_path: str) -> dict:
    """
    Load mimic-cxr-2.0.0-metadata.csv.gz and return a dict keyed by dicom_id.

    Each value is (view_position: str|None, study_date: pd.Timestamp|NaT).
    StudyDate in the CSV is an int YYYYMMDD; we convert to Timestamp here.
    """
    logger.info("Loading MIMIC-CXR metadata from %s ...", metadata_csv_path)
    meta = pd.read_csv(metadata_csv_path, usecols=["dicom_id", "ViewPosition", "StudyDate"])

    lookup: dict = {}
    for row in meta.itertuples(index=False):
        view = row.ViewPosition if pd.notna(row.ViewPosition) and str(row.ViewPosition).strip() else None
        try:
            date_str = str(int(row.StudyDate))
            if len(date_str) == 8:
                study_date = pd.to_datetime(date_str, format="%Y%m%d")
            else:
                study_date = pd.NaT
        except (ValueError, TypeError):
            study_date = pd.NaT
        lookup[str(row.dicom_id)] = (view, study_date)

    logger.info("MIMIC-CXR metadata loaded: %d dicom entries", len(lookup))
    return lookup


def _build_cxr_metadata_cache(
    record_list_path: str,
    cache_path: str,
    png_root: str,
    dicom_root: str,
    subject_ids_filter: Optional[set] = None,
    extra_png_root: Optional[str] = None,
    metadata_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scan CXR files listed in cxr-record-list.csv.

    For each image, checks PNG/JPG first (primary png_root, then extra_png_root),
    then DICOM (dicom_root). Skips entries where no file exists on disk.

    ViewPosition and StudyDate come from the official MIMIC-CXR metadata CSV
    (mimic-cxr-2.0.0-metadata.csv.gz) when metadata_csv_path is provided —
    this covers all 377K images including JPG-only files that have no DICOM on disk.
    Falls back to reading DICOM headers when the metadata CSV is not provided.

    subject_ids_filter: if provided, only scan records for these patients.
    """
    # Load metadata lookup once (preferred path: metadata CSV)
    meta_lookup: Optional[dict] = None
    if metadata_csv_path and os.path.exists(metadata_csv_path):
        meta_lookup = _load_mimic_cxr_metadata(metadata_csv_path)
    else:
        if metadata_csv_path:
            logger.warning(
                "metadata_csv_path not found: %s — will fall back to DICOM headers",
                metadata_csv_path,
            )
        import pydicom  # noqa: F401  (only needed for fallback)

    record_list = pd.read_csv(record_list_path)
    if subject_ids_filter is not None:
        n_total = len(record_list)
        record_list = record_list[record_list["subject_id"].isin(subject_ids_filter)]
        logger.info(
            "Building CXR metadata cache for %d records (cohort-filtered from %d; one-time)...",
            len(record_list), n_total,
        )
    else:
        logger.info(
            "Building CXR metadata cache for %d records (one-time)...", len(record_list)
        )

    rows = []
    n_missing = 0
    for i, row in enumerate(record_list.itertuples(), 1):
        sid = str(row.subject_id)
        partition = "p" + sid[:2]

        # --- Resolve image path: primary root → extra root, prefer PNG/JPG over DICOM ---
        img_path = None
        img_fmt = None
        for root in ([png_root] + ([extra_png_root] if extra_png_root else [])):
            base = os.path.join(root, partition, f"p{sid}", f"s{row.study_id}", row.dicom_id)
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = base + ext
                if os.path.exists(candidate):
                    img_path = candidate
                    img_fmt = ext.lstrip(".")
                    break
            if img_path:
                break

        # --- Resolve DICOM path (relative path from record_list) ---
        dicom_path = os.path.join(dicom_root, row.path)

        has_png = img_path is not None
        has_dcm = os.path.exists(dicom_path)

        if not has_png and not has_dcm:
            n_missing += 1
            continue

        # Prefer raster image (PNG/JPG) for loading; fall back to DICOM
        cxr_path = img_path if has_png else dicom_path
        cxr_fmt = img_fmt if has_png else "dcm"

        # --- ViewPosition + StudyDate ---
        if meta_lookup is not None:
            view, study_date = meta_lookup.get(str(row.dicom_id), (None, pd.NaT))
        else:
            # Fallback: read DICOM header (slow, only for DICOM files)
            view = None
            study_date = pd.NaT
            if has_dcm:
                import pydicom
                try:
                    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                    view = ds.get("ViewPosition", None)
                    study_date_str = ds.get("StudyDate", None)
                    if study_date_str and len(str(study_date_str)) == 8:
                        study_date = pd.to_datetime(str(study_date_str), format="%Y%m%d")
                except Exception as e:
                    logger.debug("DICOM header read failed for %s: %s", dicom_path, e)

        rows.append({
            "subject_id": row.subject_id,
            "study_id": row.study_id,
            "dicom_id": row.dicom_id,
            "cxr_path": cxr_path,
            "cxr_fmt": cxr_fmt,
            "view_position": view,
            "cxr_date": study_date,
        })

        if i % 10000 == 0:
            logger.info("  Scanned %d / %d candidate records...", i, len(record_list))

    n_raster = sum(1 for r in rows if r["cxr_fmt"] != "dcm")
    n_dcm = sum(1 for r in rows if r["cxr_fmt"] == "dcm")
    n_with_date = sum(1 for r in rows if not pd.isna(r["cxr_date"]))
    logger.info(
        "CXR scan complete: %d files found (%d PNG/JPG, %d DICOM), %d with date, %d missing.",
        len(rows), n_raster, n_dcm, n_with_date, n_missing,
    )

    cache = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache.to_csv(cache_path, index=False)
    logger.info("CXR metadata cache written to %s", cache_path)
    return cache


def _load_cxr_index(
    record_list_path: str,
    cache_path: str,
    png_root: str,
    dicom_root: str,
    subject_ids_filter: Optional[set] = None,
    extra_png_root: Optional[str] = None,
    metadata_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load or build CXR metadata index.
    Returns DataFrame: subject_id, study_id, dicom_id, cxr_path, cxr_fmt,
                       view_position, cxr_date.
    """
    if os.path.exists(cache_path):
        logger.info("Loading CXR metadata from cache: %s", cache_path)
        cxr = pd.read_csv(cache_path, parse_dates=["cxr_date"])
        # Stale if old schema or built before metadata-CSV fix (>20% missing both date+view)
        stale = (
            "cxr_fmt" not in cxr.columns
            or (cxr["cxr_date"].isna() & cxr["view_position"].isna()).mean() > 0.2
        )
        if stale:
            logger.warning(
                "CXR cache is stale (old schema or >20%% missing date+view). Rebuilding..."
            )
            cxr = _build_cxr_metadata_cache(
                record_list_path, cache_path, png_root, dicom_root,
                subject_ids_filter, extra_png_root, metadata_csv_path,
            )
    else:
        cxr = _build_cxr_metadata_cache(
            record_list_path, cache_path, png_root, dicom_root,
            subject_ids_filter, extra_png_root, metadata_csv_path,
        )

    n_with_date = cxr["cxr_date"].notna().sum()
    logger.info(
        "CXR index: %d files for %d patients (%d with study date)",
        len(cxr), cxr["subject_id"].nunique(), n_with_date,
    )
    return cxr


def _view_priority(view: Optional[str]) -> int:
    """PA=0 (best), AP=1, anything else=2 (worst)."""
    if view is None:
        return 2
    v = str(view).upper()
    if v == "PA":
        return 0
    if v == "AP":
        return 1
    return 2


def _select_best_ecg(
    patient_ecgs: pd.DataFrame,
    echo_date: Optional[pd.Timestamp],
    max_days: int,
) -> Optional[str]:
    """
    From a patient's ECG records, pick the one closest in time to echo_date
    within max_days. Falls back to most recent if echo_date is NaT.
    Returns the ecg_path string or None if no valid record.
    """
    if patient_ecgs.empty:
        return None

    if pd.isna(echo_date):
        # No echo date — use most recent ECG
        idx = patient_ecgs["ecg_time"].idxmax()
        return patient_ecgs.loc[idx, "ecg_path"]

    patient_ecgs = patient_ecgs.copy()
    patient_ecgs["days_off"] = (
        (patient_ecgs["ecg_time"] - echo_date).dt.total_seconds().abs() / 86400
    )
    within = patient_ecgs[patient_ecgs["days_off"] <= max_days]
    if within.empty:
        return None
    idx = within["days_off"].idxmin()
    return within.loc[idx, "ecg_path"]


def _select_best_cxr(
    patient_cxrs: pd.DataFrame,
    echo_date: Optional[pd.Timestamp],
    max_days: int,
) -> Optional[Tuple[str, str]]:
    """
    From a patient's CXR records, pick the best view (PA > AP > other)
    closest in time to echo_date within max_days.
    Returns (cxr_path, view_position) or None.
    """
    if patient_cxrs.empty:
        return None

    patient_cxrs = patient_cxrs.copy()
    patient_cxrs["view_pri"] = patient_cxrs["view_position"].apply(_view_priority)

    if pd.isna(echo_date):
        # No echo date — use best view from most recent study
        valid = patient_cxrs.dropna(subset=["cxr_date"])
        if valid.empty:
            sub = patient_cxrs
        else:
            latest_date = valid["cxr_date"].max()
            sub = valid[valid["cxr_date"] == latest_date]
        sub = sub.sort_values("view_pri")
        row = sub.iloc[0]
        return row["cxr_path"], row["view_position"]

    patient_cxrs["days_off"] = (
        (patient_cxrs["cxr_date"] - echo_date).dt.total_seconds().abs() / 86400
    )
    within = patient_cxrs[patient_cxrs["days_off"] <= max_days].copy()
    if within.empty:
        return None

    # Sort by (view_priority, days_offset) — best view and closest in time
    within = within.sort_values(["view_pri", "days_off"])
    row = within.iloc[0]
    return row["cxr_path"], row["view_position"]


def build_cohort(
    echo_labels_path: str,
    echo_structured_path: str,
    ecg_root: str,
    cxr_record_list: str,
    cxr_png_root: str,
    cxr_dicom_root: str,
    cxr_metadata_cache: str,
    col_root: str,
    col_asc: str,
    root_range: Tuple[float, float],
    asc_range: Tuple[float, float],
    max_days_offset: int = 180,
    cxr_extra_png_root: Optional[str] = None,
    cxr_metadata_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the final multi-modal cohort DataFrame.

    Columns in output:
      subject_id, target_root, target_asc, echo_date,
      ecg_path, cxr_path, cxr_view,
      has_ecg, has_cxr

    Logs drop counts at each linkage step.
    """
    # 1. Load echo labels
    labels = _load_echo_labels(
        echo_labels_path, echo_structured_path,
        col_root, col_asc, root_range, asc_range,
    )
    n_echo = len(labels)
    logger.info("=== Cohort linkage ===")
    logger.info("Starting cohort size (echo labels): %d patients", n_echo)

    # 2. Load ECG and CXR indexes (CXR cache filtered to cohort patients)
    cohort_subject_ids = set(labels["subject_id"].tolist())
    ecg_idx = _load_ecg_index(ecg_root)
    cxr_idx = _load_cxr_index(
        cxr_record_list, cxr_metadata_cache,
        png_root=cxr_png_root, dicom_root=cxr_dicom_root,
        subject_ids_filter=cohort_subject_ids,
        extra_png_root=cxr_extra_png_root,
        metadata_csv_path=cxr_metadata_csv,
    )

    ecg_by_subj = ecg_idx.groupby("subject_id")
    cxr_by_subj = cxr_idx.groupby("subject_id")

    # 3. For each patient, find best ECG and best CXR
    rows = []
    offset_log_ecg = []
    offset_log_cxr = []

    for row in labels.itertuples(index=False):
        sid = row.subject_id
        echo_date = row.echo_date if not pd.isna(row.echo_date) else None

        # ECG
        ecg_path = None
        if sid in ecg_by_subj.groups:
            patient_ecgs = ecg_by_subj.get_group(sid)
            ecg_path = _select_best_ecg(patient_ecgs, echo_date, max_days_offset)
            if ecg_path is not None and echo_date is not None:
                # Log offset for distribution analysis
                t = patient_ecgs.loc[patient_ecgs["ecg_path"] == ecg_path, "ecg_time"]
                if not t.empty:
                    off = abs((t.iloc[0] - echo_date).total_seconds() / 86400)
                    offset_log_ecg.append(off)

        # CXR
        cxr_path = None
        cxr_view = None
        if sid in cxr_by_subj.groups:
            patient_cxrs = cxr_by_subj.get_group(sid)
            result = _select_best_cxr(patient_cxrs, echo_date, max_days_offset)
            if result is not None:
                cxr_path, cxr_view = result
                if echo_date is not None and not pd.isna(
                    cxr_idx.loc[cxr_idx["cxr_path"] == cxr_path, "cxr_date"].squeeze()
                    if not cxr_idx[cxr_idx["cxr_path"] == cxr_path].empty else pd.NaT
                ):
                    cxr_date_val = cxr_idx.loc[
                        cxr_idx["cxr_path"] == cxr_path, "cxr_date"
                    ].iloc[0]
                    if not pd.isna(cxr_date_val):
                        off = abs((cxr_date_val - echo_date).total_seconds() / 86400)
                        offset_log_cxr.append(off)

        rows.append({
            "subject_id": sid,
            "target_root": row.target_root,
            "target_asc": row.target_asc,
            "echo_date": row.echo_date,
            "ecg_path": ecg_path,
            "cxr_path": cxr_path,
            "cxr_view": cxr_view,
            "has_ecg": ecg_path is not None,
            "has_cxr": cxr_path is not None,
        })

    cohort = pd.DataFrame(rows)

    # 4. Log drop statistics
    n_both = (cohort["has_ecg"] & cohort["has_cxr"]).sum()
    n_ecg_only = (cohort["has_ecg"] & ~cohort["has_cxr"]).sum()
    n_cxr_only = (~cohort["has_ecg"] & cohort["has_cxr"]).sum()
    n_neither = (~cohort["has_ecg"] & ~cohort["has_cxr"]).sum()

    logger.info(
        "Modality breakdown — both: %d | ECG-only: %d | CXR-only: %d | neither: %d",
        n_both, n_ecg_only, n_cxr_only, n_neither,
    )

    # Drop patients with no modalities at all
    cohort = cohort[cohort["has_ecg"] | cohort["has_cxr"]].reset_index(drop=True)
    logger.info(
        "After dropping no-modality patients: %d remain (dropped %d)",
        len(cohort),
        n_neither,
    )

    # 5. Log temporal offset distributions
    if offset_log_ecg:
        arr = np.array(offset_log_ecg)
        logger.info(
            "ECG offset (days) — median: %.1f  90th pct: %.1f  max: %.1f",
            np.median(arr), np.percentile(arr, 90), arr.max(),
        )
    if offset_log_cxr:
        arr = np.array(offset_log_cxr)
        logger.info(
            "CXR offset (days) — median: %.1f  90th pct: %.1f  max: %.1f",
            np.median(arr), np.percentile(arr, 90), arr.max(),
        )

    # 6. Final label stats
    logger.info("Final cohort: %d patients", len(cohort))
    for col, name in [("target_root", "Aortic root (cm)"), ("target_asc", "Ascending aorta (cm)")]:
        vals = cohort[col].dropna()
        logger.info(
            "%s — n=%d  mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
            name, len(vals), vals.mean(), vals.std(), vals.min(), vals.max(),
        )

    return cohort


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class AortaDataset(Dataset):
    """
    Multi-modal dataset for aortic diameter regression.

    Each item returns:
        ecg       : (12, 5000) float32 tensor, or zeros if ECG unavailable
        cxr       : (3, 224, 224) float32 tensor, or zeros if CXR unavailable
        target    : (2,) float32 tensor [root_cm, asc_cm]; NaN where label is missing
        modality_mask : dict {'ecg': bool, 'cxr': bool}
        subject_id : int
    """

    def __init__(
        self,
        cohort: pd.DataFrame,
        ecg_transform: Optional[Callable] = None,
        cxr_transform: Optional[Callable] = None,
        ecg_cfg=None,
        cxr_cfg=None,
        is_train: bool = False,
        target_stats: Optional[TargetStats] = None,
        ecg_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        cxr_embeddings: Optional[Dict[int, torch.Tensor]] = None,
    ):
        """
        Parameters
        ----------
        cohort : output of build_cohort() filtered to the desired split
        ecg_transform : callable(path) -> (12, 5000) float32 tensor
        cxr_transform : callable(path, is_train) -> (3, H, W) float32 tensor
        ecg_cfg / cxr_cfg : DataConfig fields (passed through to transforms)
        is_train : controls augmentation in cxr_transform
        target_stats : when provided, targets returned by __getitem__ are
                       z-scored using these stats (computed on the train split).
                       NaN target entries are preserved.
        ecg_embeddings : optional {subject_id: tensor(320,)} dict of precomputed
                         PCLR embeddings. When provided, __getitem__ returns the
                         precomputed embedding as `ecg` instead of loading the raw
                         waveform. ecg_transform is ignored in this mode.
        cxr_embeddings : optional {subject_id: tensor(768,)} dict of precomputed
                         RAD-DINO embeddings. When provided, __getitem__ returns
                         the precomputed embedding as `cxr` instead of loading the
                         raw image. cxr_transform is ignored in this mode.
        """
        self.cohort = cohort.reset_index(drop=True)
        self.ecg_transform = ecg_transform
        self.cxr_transform = cxr_transform
        self.ecg_cfg = ecg_cfg
        self.cxr_cfg = cxr_cfg
        self.is_train = is_train
        self.target_stats = target_stats
        self.ecg_embeddings = ecg_embeddings  # {int subject_id: tensor(320,)} or None
        self.cxr_embeddings = cxr_embeddings  # {int subject_id: tensor(768,)} or None

    def __len__(self) -> int:
        return len(self.cohort)

    def __getitem__(self, idx: int) -> Dict:
        row = self.cohort.iloc[idx]

        # ---- ECG ----
        has_ecg = bool(row["has_ecg"])
        subject_id = int(row["subject_id"])

        if self.ecg_embeddings is not None:
            # PCLR mode: return precomputed 320-dim embedding
            if subject_id in self.ecg_embeddings:
                ecg = self.ecg_embeddings[subject_id].float()
                has_ecg = True
            else:
                ecg = torch.zeros(320, dtype=torch.float32)
                has_ecg = False
        elif has_ecg and self.ecg_transform is not None:
            try:
                ecg = self.ecg_transform(row["ecg_path"], self.ecg_cfg)
            except Exception as e:
                logger.warning("ECG load failed for %s: %s", row["ecg_path"], e)
                ecg = torch.zeros(12, 5000, dtype=torch.float32)
                has_ecg = False
        else:
            ecg = torch.zeros(12, 5000, dtype=torch.float32)
            has_ecg = False

        # ---- CXR ----
        has_cxr = bool(row["has_cxr"])
        if self.cxr_embeddings is not None:
            # RAD-DINO frozen mode: return precomputed 768-dim embedding
            if subject_id in self.cxr_embeddings:
                cxr = self.cxr_embeddings[subject_id].float()
                has_cxr = True
            else:
                cxr = torch.zeros(768, dtype=torch.float32)
                has_cxr = False
        elif has_cxr and self.cxr_transform is not None:
            try:
                cxr = self.cxr_transform(row["cxr_path"], self.cxr_cfg, self.is_train)
            except Exception as e:
                logger.warning("CXR load failed for %s: %s", row["cxr_path"], e)
                cxr = torch.zeros(3, 224, 224, dtype=torch.float32)
                has_cxr = False
        else:
            cxr = torch.zeros(3, 224, 224, dtype=torch.float32)
            has_cxr = False

        # ---- Labels ----
        root_val = row["target_root"]
        asc_val = row["target_asc"]
        target = torch.tensor(
            [
                float(root_val) if not pd.isna(root_val) else float("nan"),
                float(asc_val)  if not pd.isna(asc_val)  else float("nan"),
            ],
            dtype=torch.float32,
        )
        if self.target_stats is not None:
            target = self.target_stats.normalize(target)

        return {
            "ecg": ecg,
            "cxr": cxr,
            "target": target,
            "modality_mask": {"ecg": has_ecg, "cxr": has_cxr},
            "subject_id": int(row["subject_id"]),
        }
