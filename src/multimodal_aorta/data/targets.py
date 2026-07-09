"""
Aortic-dilation target definitions (v3): ordinal grade + binary/3-class views.

One source per site (root/sinus, ascending) — the echo diameter in cm — yields:
  - the continuous diameter        (regression target, unchanged)
  - an ordinal grade in {0,1,2,3}  (classification target)

Grade cut-points (cm), clinically standard for ascending aortic aneurysm staging:
    0  no AD        diam < 4.0
    1  mild         4.0 <= diam < 4.5
    2  moderate     4.5 <= diam < 5.0
    3  severe       diam >= 5.0

Because the classes are ordered AND the upper classes are very rare at n~522, we
model this ordinally (CORN/CORAL style): K=4 grades -> 3 cumulative binary targets
  t1 = [diam >= 4.0]   (== "any AD", the primary endpoint)
  t2 = [diam >= 4.5]   (== "moderate+", the clinically-actionable 3-class boundary)
  t3 = [diam >= 5.0]   (== "severe")
A single ordinal model therefore yields the binary, 3-class, and 4-grade readouts.

NaN diameters (a site with no measurement) propagate as NaN so losses/metrics can
mask them per-site.
"""

from typing import List

import numpy as np
import pandas as pd

# Cut-points (cm). The k-th cumulative target is [diam >= AD_CUTS[k]].
AD_CUTS: List[float] = [4.0, 4.5, 5.0]
N_GRADES: int = len(AD_CUTS) + 1          # 4 ordinal grades
GRADE_NAMES = ["no_AD", "mild", "moderate", "severe"]


def diam_to_grade(cm: float) -> float:
    """Continuous diameter (cm) -> ordinal grade {0,1,2,3}; NaN preserved."""
    if cm is None or (isinstance(cm, float) and np.isnan(cm)):
        return float("nan")
    g = 0
    for cut in AD_CUTS:
        if cm >= cut:
            g += 1
    return float(g)


def diam_to_cumulative(cm: float) -> np.ndarray:
    """
    Diameter (cm) -> length-3 cumulative binary vector [>=4.0, >=4.5, >=5.0].
    NaN diameter -> [NaN, NaN, NaN].
    """
    if cm is None or (isinstance(cm, float) and np.isnan(cm)):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    return np.array([1.0 if cm >= c else 0.0 for c in AD_CUTS], dtype=np.float32)


def grade_to_3class(grade: float) -> float:
    """Collapse {moderate, severe} -> 2, giving {0:normal, 1:mild, 2:moderate+}."""
    if grade is None or np.isnan(grade):
        return float("nan")
    return float(min(int(grade), 2))


def add_indexed_targets(
    cohort: pd.DataFrame,
    bsa_map: dict,
    height_cm_map: dict,
) -> pd.DataFrame:
    """
    Add body-size-indexed diameter targets (M3). Aortic size scales with body
    size; indexing removes that so a model must capture size-independent variation.

    Adds (per site root/asc), using BSA (m^2) and height (m) from the EHR cache:
        {site}_diam_per_bsa      = target_{site} / bsa      (aortic size index, ASI)
        {site}_diam_per_height   = target_{site} / height_m
    plus the raw `bsa` and `height_m` columns. Missing/zero BSA or height -> NaN
    (those patients are masked in indexed-target training/eval).

    Parameters
    ----------
    cohort : DataFrame with 'subject_id', 'target_root', 'target_asc'
    bsa_map : {subject_id -> bsa_m2}
    height_cm_map : {subject_id -> height_cm}
    """
    df = cohort.copy()
    sid = df["subject_id"].astype(int)
    bsa = sid.map(lambda s: bsa_map.get(s, np.nan)).astype(float)
    height_m = sid.map(lambda s: height_cm_map.get(s, np.nan)).astype(float) / 100.0
    bsa = bsa.where(bsa > 0)               # 0 or negative -> NaN
    height_m = height_m.where(height_m > 0)
    df["bsa"] = bsa.values
    df["height_m"] = height_m.values
    for site in ("root", "asc"):
        col = f"target_{site}"
        df[f"{site}_diam_per_bsa"] = (df[col] / bsa).values
        df[f"{site}_diam_per_height"] = (df[col] / height_m).values
    return df


def add_grade_columns(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-site grade + binary columns to a cohort DataFrame in place-safe manner.
    Expects 'target_root' and 'target_asc' (cm). Returns a copy with new columns:
        grade_root, grade_asc            ordinal {0..3}
        anyAD_root, anyAD_asc            [diam >= 4.0]
        anyAD                            patient-level (either site >= 4.0)
        grade_max                        max(grade_root, grade_asc) for stratification
    """
    df = cohort.copy()
    df["grade_root"] = df["target_root"].apply(diam_to_grade)
    df["grade_asc"]  = df["target_asc"].apply(diam_to_grade)
    df["anyAD_root"] = (df["target_root"] >= AD_CUTS[0]).astype("float")
    df["anyAD_asc"]  = (df["target_asc"]  >= AD_CUTS[0]).astype("float")
    # patient-level any-AD: dilated in either site (NaN treated as not-dilated here,
    # which is the conservative choice for a screening label)
    root_ad = (df["target_root"] >= AD_CUTS[0]).fillna(False)
    asc_ad  = (df["target_asc"]  >= AD_CUTS[0]).fillna(False)
    df["anyAD"] = (root_ad | asc_ad).astype(int)
    df["grade_max"] = df[["grade_root", "grade_asc"]].max(axis=1)
    return df
