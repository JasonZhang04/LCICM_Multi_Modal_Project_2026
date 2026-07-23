"""
Episode-level cohort abstraction for the multimodal aorta project.

An *episode* is one (patient, echo study) = one row of
`pretrained_checkpoints/episodes.csv`, produced by `scripts/build_episode_cohort.py`.
This replaces the patient-level `cohort_triple.csv` used by v1-v7, which collapsed
each patient's whole echo history into a single worst-ever label anchored on the
patient's earliest measurement date (see notes/experiments_2026-07-23*).

The modelling UNIT is the episode; the GROUPING unit for cross-validation and
bootstrap remains the patient (`subject_id`), because a patient's repeated echoes
are correlated and must never straddle a train/test boundary.

Key identifiers
    episode_id   f"{subject_id}_{measurement_id}"  (unique string key)
    subject_id   patient (fold-grouping / cluster-bootstrap unit)
    measurement_id  the specific echo study the label + index date come from

Column convention: `root_cm` / `asc_cm` from episodes.csv are exposed as
`target_root` / `target_asc` so the existing targets.add_grade_columns and all
downstream metric code work unchanged.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from .targets import add_grade_columns

logger = logging.getLogger(__name__)

EPISODES_CSV = "episodes.csv"


def make_episode_id(subject_id, measurement_id) -> str:
    """Stable string key for one (patient, echo study)."""
    return f"{int(subject_id)}_{int(measurement_id)}"


def load_episodes(
    pc_dir: str,
    require_ecg: bool = False,
    filename: str = EPISODES_CSV,
) -> pd.DataFrame:
    """
    Load the episode cohort and attach ids, targets, and grade columns.

    Parameters
    ----------
    pc_dir : path to the pretrained_checkpoints directory containing episodes.csv
    require_ecg : if True, keep only tri-modal episodes (n_ecg > 0). Default False
        keeps every CXR-eligible episode and treats ECG as an optional modality —
        ECG binds only ~10% of episodes (see audit), and a large "ECG adds nothing"
        result is more valuable than a smaller strictly-tri-modal cohort.
    filename : override for the episodes CSV name (e.g. a sensitivity-analysis build)

    Returns
    -------
    DataFrame with, in addition to the raw episodes.csv columns:
        episode_id, target_root, target_asc, and the add_grade_columns outputs
        (grade_root, grade_asc, anyAD_root, anyAD_asc, anyAD, grade_max).
    """
    path = os.path.join(pc_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"episode cohort not found at {path}; run "
            "`python scripts/build_episode_cohort.py` (or its slurm wrapper) first.")
    ep = pd.read_csv(path)
    ep["subject_id"] = ep["subject_id"].astype(int)
    ep["measurement_id"] = ep["measurement_id"].astype(int)
    ep["echo_dt"] = pd.to_datetime(ep["echo_dt"], errors="coerce")

    n0 = len(ep)
    if require_ecg:
        ep = ep[ep["n_ecg"] > 0].reset_index(drop=True)
        logger.info("require_ecg=True: kept %d/%d tri-modal episodes", len(ep), n0)

    # Expose the site diameters under the names the rest of the codebase expects.
    ep = ep.rename(columns={"root_cm": "target_root", "asc_cm": "target_asc"})
    ep["episode_id"] = [make_episode_id(s, m)
                        for s, m in zip(ep["subject_id"], ep["measurement_id"])]

    if not ep["episode_id"].is_unique:
        dup = ep.loc[ep["episode_id"].duplicated(), "episode_id"].head().tolist()
        raise ValueError(f"duplicate episode_id(s) in {path}, e.g. {dup}")

    ep = add_grade_columns(ep)
    logger.info(
        "Loaded %d episodes / %d patients (%.2f per patient) | root>=4.0 %d | asc>=4.0 %d",
        len(ep), ep["subject_id"].nunique(), len(ep) / max(ep["subject_id"].nunique(), 1),
        int(ep["anyAD_root"].sum()), int(ep["anyAD_asc"].sum()))
    return ep


def patient_level_label(ep: pd.DataFrame) -> pd.DataFrame:
    """
    One row per patient with a stratification label = 1 if ANY of the patient's
    episodes is anyAD-positive. Used to stratify the patient-grouped CV folds so
    positives are spread evenly across folds even though grouping is by patient.

    Returns DataFrame: subject_id, strat (int 0/1), n_episodes.
    """
    g = ep.groupby("subject_id")
    out = pd.DataFrame({
        "subject_id": list(g.groups.keys()),
    })
    agg = g.agg(strat=("anyAD", "max"), n_episodes=("episode_id", "size")).reset_index()
    out = out.merge(agg, on="subject_id")
    out["strat"] = out["strat"].fillna(0).astype(int)
    return out
