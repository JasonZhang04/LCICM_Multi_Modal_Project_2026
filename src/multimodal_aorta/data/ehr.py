"""
EHR (demographics + body-size) feature extraction for the aortic cohort.

Pulls a small, clinically-motivated set of tabular features from MIMIC-IV v3.1
`hosp` module and matches them to each patient's echo date:

    age, sex, height_cm, weight_kg, bmi, bsa, sbp, dbp  (+ *_missing indicators)

Why these: aortic diameter is normally indexed to body size / age / sex, so these
are genuine confounders the downstream model needs — not just "another modality".

Sources (on cluster, no download):
    patients.csv.gz : gender, anchor_age, anchor_year      (one row / subject)
    omr.csv.gz      : Height (Inches), Weight (Lbs), BMI (kg/m2), Blood Pressure
                      (Online Medical Record; one row / subject / chartdate / type)

Matching: for each body-size measurement we take the omr record whose chartdate is
nearest the echo date (across all of a patient's records); body size is slowly
varying so nearest-in-time is a reasonable proxy. The temporal offset is recorded
in `qc_*_offset_days` columns for auditing.

Outputs raw physical units with NaN for missing values. Normalization / imputation
is deliberately left to training time (per CV-fold) so there is no train/test leak.

NOTE on reading omr.csv.gz: the MIMIC-IV file has trailing bytes after the gzip
stream that Python's strict `gzip` rejects ("Not a gzipped file"), even though the
data decompresses fine. We therefore stream it through the system `zcat`.
"""

import logging
import subprocess
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- omr result_name strings (verified against MIMIC-IV v3.1) ---
RN_HEIGHT = "Height (Inches)"
RN_WEIGHT = "Weight (Lbs)"
RN_BMI    = "BMI (kg/m2)"
RN_BP     = "Blood Pressure"
_OMR_WANT = {RN_HEIGHT, RN_WEIGHT, RN_BMI, RN_BP}

# --- unit conversions ---
IN_TO_CM  = 2.54
LB_TO_KG  = 0.45359237

# --- plausibility clip ranges (values outside → treated as missing) ---
RANGE_HEIGHT_CM = (100.0, 220.0)
RANGE_WEIGHT_KG = (25.0, 300.0)
RANGE_BMI       = (10.0, 80.0)
RANGE_SBP       = (60.0, 260.0)
RANGE_DBP       = (30.0, 160.0)

# Feature columns consumed by the model (order matters for the deep encoder).
FEATURE_COLS = [
    "age", "sex", "height_cm", "weight_kg", "bmi", "bsa", "sbp", "dbp",
    "height_missing", "weight_missing", "bsa_missing", "bp_missing",
]


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def _read_omr_for_subjects(omr_path: str, subject_ids: set) -> pd.DataFrame:
    """
    Stream omr.csv.gz via system `zcat`, keep only our subjects + result types,
    parse result_value into a numeric column. Returns long-form DataFrame with
    columns: subject_id, chartdate (datetime), result_name, value (float),
    sbp (float|NaN), dbp (float|NaN).
    """
    logger.info("Reading omr via zcat: %s", omr_path)
    proc = subprocess.Popen(["zcat", omr_path], stdout=subprocess.PIPE)
    parts = []
    try:
        for chunk in pd.read_csv(
            proc.stdout,
            usecols=["subject_id", "chartdate", "result_name", "result_value"],
            chunksize=500_000,
        ):
            chunk = chunk[
                chunk["subject_id"].isin(subject_ids)
                & chunk["result_name"].isin(_OMR_WANT)
            ]
            if len(chunk):
                parts.append(chunk)
    finally:
        proc.stdout.close()
        proc.wait()

    omr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["subject_id", "chartdate", "result_name", "result_value"]
    )
    omr["chartdate"] = pd.to_datetime(omr["chartdate"], errors="coerce")

    # Numeric value (NaN for BP, parsed separately)
    omr["value"] = pd.to_numeric(omr["result_value"], errors="coerce")

    # Blood pressure "S/D" -> two columns
    bp = omr["result_name"] == RN_BP
    sd = omr.loc[bp, "result_value"].astype(str).str.split("/", n=1, expand=True)
    omr["sbp"] = np.nan
    omr["dbp"] = np.nan
    if not sd.empty:
        omr.loc[bp, "sbp"] = pd.to_numeric(sd[0], errors="coerce").values
        if sd.shape[1] > 1:
            omr.loc[bp, "dbp"] = pd.to_numeric(sd[1], errors="coerce").values

    logger.info("omr filtered: %d rows for %d subjects",
                len(omr), omr["subject_id"].nunique())
    return omr


def _nearest_record(
    sub_df: pd.DataFrame, echo_date: pd.Timestamp, value_col: str = "value"
) -> Tuple[float, float]:
    """
    From one subject's records of a single type, return (value, offset_days) for
    the chartdate nearest the echo date. If multiple records share that nearest
    date, the median value is used. Returns (NaN, NaN) when none are valid.
    """
    d = sub_df.dropna(subset=[value_col, "chartdate"])
    if d.empty:
        return float("nan"), float("nan")
    offsets = (d["chartdate"] - echo_date).abs()
    if pd.isna(echo_date):
        # No echo date: use the most recent record.
        idx = d["chartdate"].idxmax()
        nearest_date = d.loc[idx, "chartdate"]
        off = float("nan")
    else:
        nearest_date = d.loc[offsets.idxmin(), "chartdate"]
        off = float(abs((nearest_date - echo_date).days))
    val = float(d.loc[d["chartdate"] == nearest_date, value_col].median())
    return val, off


def _clip(val: float, lo: float, hi: float) -> float:
    """Return val if within [lo, hi] else NaN."""
    if val is None or np.isnan(val):
        return float("nan")
    return val if (lo <= val <= hi) else float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ehr_features(
    echo_dates: pd.DataFrame,   # columns: subject_id, echo_date (datetime)
    patients_path: str,
    omr_path: str,
) -> pd.DataFrame:
    """
    Build one EHR feature row per subject in `echo_dates`.

    Returns DataFrame with FEATURE_COLS plus subject_id and qc_* offset columns.
    Raw physical units; NaN for missing numerics; *_missing are 1/0 indicators.
    """
    subjects = set(int(s) for s in echo_dates["subject_id"].tolist())

    # --- patients: age (at echo), sex ---
    pat = pd.read_csv(
        patients_path, usecols=["subject_id", "gender", "anchor_age", "anchor_year"]
    )
    pat = pat[pat["subject_id"].isin(subjects)].set_index("subject_id")

    # --- omr: height / weight / bmi / bp ---
    omr = _read_omr_for_subjects(omr_path, subjects)
    omr_by_subj = {sid: g for sid, g in omr.groupby("subject_id")}

    rows = []
    for r in echo_dates.itertuples(index=False):
        sid = int(r.subject_id)
        echo_date = r.echo_date if not pd.isna(r.echo_date) else pd.NaT
        echo_year = echo_date.year if not pd.isna(echo_date) else np.nan

        # demographics
        age = sex = np.nan
        if sid in pat.index:
            prow = pat.loc[sid]
            if not np.isnan(echo_year):
                age = float(prow["anchor_age"] + (echo_year - prow["anchor_year"]))
            else:
                age = float(prow["anchor_age"])
            sex = 1.0 if str(prow["gender"]).upper().startswith("M") else 0.0

        # body size from omr
        sub = omr_by_subj.get(sid)
        h_in = w_lb = bmi = sbp = dbp = float("nan")
        h_off = w_off = float("nan")
        if sub is not None:
            h_in, h_off = _nearest_record(sub[sub.result_name == RN_HEIGHT], echo_date)
            w_lb, w_off = _nearest_record(sub[sub.result_name == RN_WEIGHT], echo_date)
            bmi, _      = _nearest_record(sub[sub.result_name == RN_BMI], echo_date)
            sbp, _      = _nearest_record(sub[sub.result_name == RN_BP], echo_date, "sbp")
            dbp, _      = _nearest_record(sub[sub.result_name == RN_BP], echo_date, "dbp")

        height_cm = _clip(h_in * IN_TO_CM if not np.isnan(h_in) else np.nan, *RANGE_HEIGHT_CM)
        weight_kg = _clip(w_lb * LB_TO_KG if not np.isnan(w_lb) else np.nan, *RANGE_WEIGHT_KG)
        bmi       = _clip(bmi, *RANGE_BMI)
        sbp       = _clip(sbp, *RANGE_SBP)
        dbp       = _clip(dbp, *RANGE_DBP)

        # height fallback from BMI + weight: height_m = sqrt(weight_kg / bmi)
        if np.isnan(height_cm) and not np.isnan(weight_kg) and not np.isnan(bmi):
            height_cm = _clip(float(np.sqrt(weight_kg / bmi) * 100.0), *RANGE_HEIGHT_CM)

        # bmi fallback from height + weight
        if np.isnan(bmi) and not np.isnan(weight_kg) and not np.isnan(height_cm):
            bmi = _clip(float(weight_kg / (height_cm / 100.0) ** 2), *RANGE_BMI)

        # BSA (Mosteller) needs height + weight
        if not np.isnan(height_cm) and not np.isnan(weight_kg):
            bsa = float(np.sqrt(height_cm * weight_kg / 3600.0))
        else:
            bsa = float("nan")

        rows.append({
            "subject_id": sid,
            "age": age, "sex": sex,
            "height_cm": height_cm, "weight_kg": weight_kg, "bmi": bmi, "bsa": bsa,
            "sbp": sbp, "dbp": dbp,
            "height_missing": int(np.isnan(height_cm)),
            "weight_missing": int(np.isnan(weight_kg)),
            "bsa_missing":    int(np.isnan(bsa)),
            "bp_missing":     int(np.isnan(sbp)),
            "qc_height_offset_days": h_off,
            "qc_weight_offset_days": w_off,
        })

    out = pd.DataFrame(rows)
    logger.info(
        "EHR features built for %d subjects | age n=%d, height n=%d, weight n=%d, "
        "bsa n=%d, bp n=%d",
        len(out), out["age"].notna().sum(), out["height_cm"].notna().sum(),
        out["weight_kg"].notna().sum(), out["bsa"].notna().sum(), out["sbp"].notna().sum(),
    )
    return out


def load_ehr_features(csv_path: str) -> Dict[int, np.ndarray]:
    """
    Load the cached EHR feature CSV and return {subject_id: np.ndarray(FEATURE_COLS)}.
    Raw units with NaN for missing — imputation/normalization happens at train time.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"EHR cache missing feature columns: {missing}")
    out = {}
    for r in df.itertuples(index=False):
        rd = r._asdict()
        out[int(rd["subject_id"])] = np.array([rd[c] for c in FEATURE_COLS], dtype=np.float32)
    logger.info("Loaded EHR features: %d subjects × %d features", len(out), len(FEATURE_COLS))
    return out
