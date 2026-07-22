"""
M0 — standardized out-of-fold (OOF) prediction I/O.

Every model writes its per-patient OOF predictions in ONE long-format schema,
keyed by `subject_id`, so `scripts/make_eval_report.py` can align any pair of
models patient-for-patient (needed for paired bootstrap deltas: multimodal minus
EHR floor, etc.). Each script also keeps its legacy `oof.npz` for backward compat.

Long format: one row per (subject_id, model, site, endpoint).

Columns
-------
subject_id     patient id (int)
fold_id        outer fold that held this patient out
model_name     e.g. "gbdt_concat", "deep_fusion", "late_fusion_meta", "ehr_baseline"
modality_set   e.g. "ecg+cxr+ehr", "ehr", "echo"
site           "root" | "asc"
endpoint       "ge40" | "ge45" | "diam" | ...
target_type    "binary" | "regression"
y_true         ground-truth label (0/1 for binary, cm for regression)
pred_prob      predicted probability (binary; NaN for regression)
pred_value     predicted value (regression; NaN for binary)
has_ecg / has_cxr / has_ehr   modality-presence flags (1/0)
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

OOF_COLUMNS = [
    "subject_id", "fold_id", "model_name", "modality_set", "site", "endpoint",
    "target_type", "y_true", "pred_prob", "pred_value",
    "has_ecg", "has_cxr", "has_ehr",
]


def _as_col(v, n: int):
    """Broadcast a scalar to length n, or pass through an array-like of length n."""
    if v is None or np.isscalar(v):
        return [v] * n
    v = list(v)
    assert len(v) == n, f"length {len(v)} != {n}"
    return v


def build_records(
    *,
    subject_ids,
    fold_ids,
    model_name: str,
    modality_set: str,
    site: str,
    endpoint: str,
    target_type: str,
    y_true,
    pred_prob=None,
    pred_value=None,
    has_ecg=1,
    has_cxr=1,
    has_ehr=1,
) -> List[dict]:
    """
    Build a list of OOF row-dicts for one (model, site, endpoint) block.

    `subject_ids` and every array argument must be aligned to the same order.
    `fold_ids` may be a scalar, an array, or resolved by the caller from a
    subject->fold map. Rows whose y_true is NaN are kept (so coverage is
    auditable); the eval report drops them per metric as needed.
    """
    subject_ids = [int(s) for s in subject_ids]
    n = len(subject_ids)
    fold_ids = _as_col(fold_ids, n)
    y_true = _as_col(y_true, n)
    pred_prob = _as_col(pred_prob, n)
    pred_value = _as_col(pred_value, n)
    has_ecg = _as_col(has_ecg, n)
    has_cxr = _as_col(has_cxr, n)
    has_ehr = _as_col(has_ehr, n)
    records = []
    for i in range(n):
        records.append({
            "subject_id": subject_ids[i],
            "fold_id": fold_ids[i],
            "model_name": model_name,
            "modality_set": modality_set,
            "site": site,
            "endpoint": endpoint,
            "target_type": target_type,
            "y_true": y_true[i],
            "pred_prob": pred_prob[i],
            "pred_value": pred_value[i],
            "has_ecg": has_ecg[i],
            "has_cxr": has_cxr[i],
            "has_ehr": has_ehr[i],
        })
    return records


def save_oof(records: List[dict], path: str) -> pd.DataFrame:
    """Write OOF records to a long-format CSV with the canonical column order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(records)
    for c in OOF_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[OOF_COLUMNS]
    df.to_csv(path, index=False)
    return df


def _git_commit(root: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _file_hash(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


def write_manifest(
    path: str,
    *,
    model_name: str,
    seed: int,
    n_patients: int,
    cohort_csv: Optional[str] = None,
    fold_csv: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Write a small run_manifest.json capturing reproducibility metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    manifest = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "git_commit": _git_commit(root),
        "n_patients": int(n_patients),
        "cohort_csv": cohort_csv,
        "cohort_hash": _file_hash(cohort_csv) if cohort_csv else None,
        "fold_csv": fold_csv,
        "fold_hash": _file_hash(fold_csv) if fold_csv else None,
    }
    if extra:
        manifest.update(extra)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
