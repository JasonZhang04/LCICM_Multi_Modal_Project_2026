"""
Patient-level train / val / test splitting with stratification.

Stratification is done on a binned version of the aortic root diameter
(sinus_diam_cm) so that label distribution is balanced across splits.
Patients with NaN root diameter are binned into their own stratum so they
are also distributed proportionally.
"""
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


def load_fold_assignments(
    path: str,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load the immutable fold assignments written by `scripts/build_fold_assignments.py`
    and reconstruct the `(train_ids, test_ids)` list used everywhere else.

    This is a drop-in replacement for `make_cv_folds`: the returned structure is
    identical (a list of subject_id arrays), but the folds come from a frozen file
    rather than being regenerated from a seed each run. That guarantees every
    trainer shares one auditable fold definition even if a patient loses an
    embedding (the trainer's own `idx_of` filtering then simply skips it, without
    reshuffling anyone else).

    Parameters
    ----------
    path : path to fold_assignments.csv (columns: subject_id, fold_id, ...)
    n_splits : expected number of folds (validated against the file)

    Returns
    -------
    folds : list of (train_ids, test_ids) subject_id arrays, ordered by fold_id.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"fold assignments not found at {path}; run "
            "`python scripts/build_fold_assignments.py` first.")
    fa = pd.read_csv(path)
    assert {"subject_id", "fold_id"} <= set(fa.columns), \
        "fold_assignments.csv must have 'subject_id' and 'fold_id'"
    fa["subject_id"] = fa["subject_id"].astype(int)
    assert fa["subject_id"].is_unique, "duplicate subject_id in fold assignments"
    unique_folds = sorted(fa["fold_id"].unique())
    if len(unique_folds) != n_splits:
        logger.warning("fold file has %d folds (expected %d)", len(unique_folds), n_splits)

    all_ids = fa["subject_id"].to_numpy()
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in unique_folds:
        test_ids = fa.loc[fa.fold_id == k, "subject_id"].to_numpy()
        train_ids = np.setdiff1d(all_ids, test_ids)
        folds.append((train_ids, test_ids))
    logger.info("Loaded %d folds for %d subjects from %s", len(folds), len(all_ids), path)
    return folds


def load_fold_id_map(path: str) -> Dict[int, int]:
    """Return {subject_id: fold_id} from the immutable fold assignments file."""
    fa = pd.read_csv(path)
    return {int(s): int(k) for s, k in zip(fa["subject_id"], fa["fold_id"])}


def load_episode_folds(
    path: str,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Episode-level analogue of load_fold_assignments.

    Loads `episode_fold_assignments.csv` (columns: episode_id, subject_id,
    measurement_id, fold_id, ...) and returns `(train_episode_ids, test_episode_ids)`
    per fold. Folds are PATIENT-grouped in the file (every episode of a patient
    shares one fold_id), so returning episode ids here cannot leak a patient
    across the boundary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"episode fold assignments not found at {path}; run "
            "`python scripts/build_episode_folds.py` first.")
    fa = pd.read_csv(path)
    assert {"episode_id", "subject_id", "fold_id"} <= set(fa.columns), \
        "episode_fold_assignments.csv must have episode_id, subject_id, fold_id"
    fa["episode_id"] = fa["episode_id"].astype(str)
    assert fa["episode_id"].is_unique, "duplicate episode_id in fold assignments"
    # A patient must live in exactly one fold.
    per_pt_folds = fa.groupby("subject_id")["fold_id"].nunique()
    assert (per_pt_folds == 1).all(), \
        f"{int((per_pt_folds > 1).sum())} patients span multiple folds — grouping is broken"

    all_eids = fa["episode_id"].to_numpy()
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in sorted(fa["fold_id"].unique()):
        test_eids = fa.loc[fa.fold_id == k, "episode_id"].to_numpy()
        train_eids = np.setdiff1d(all_eids, test_eids)
        folds.append((train_eids, test_eids))
    logger.info("Loaded %d episode folds for %d episodes / %d patients from %s",
                len(folds), len(all_eids), fa["subject_id"].nunique(), path)
    return folds


def load_episode_fold_id_map(path: str) -> Dict[str, int]:
    """Return {episode_id: fold_id} from the episode fold assignments file."""
    fa = pd.read_csv(path)
    return {str(e): int(k) for e, k in zip(fa["episode_id"], fa["fold_id"])}


def make_grouped_cv_folds(
    episodes: pd.DataFrame,
    stratify_col: str = "anyAD",
    group_col: str = "subject_id",
    id_col: str = "episode_id",
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Patient-grouped, stratified k-fold over EPISODES.

    Every episode of a patient goes to the same fold (grouped by `group_col`),
    while folds are balanced on `stratify_col` (episode-level any-AD). Returns
    `(train_episode_ids, test_episode_ids)` per fold. Used by the repeated-CV
    trainers to regenerate folds per seed the same way the immutable file was built.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    for c in (stratify_col, group_col, id_col):
        assert c in episodes.columns, f"episodes must contain '{c}'"

    eids = episodes[id_col].astype(str).to_numpy()
    groups = episodes[group_col].astype(int).to_numpy()
    y = episodes[stratify_col].fillna(0).astype(int).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k, (tr, te) in enumerate(sgkf.split(eids, y, groups)):
        # Defensive: no patient may appear in both sides.
        assert not (set(groups[tr]) & set(groups[te])), \
            f"patient leak in grouped fold {k}"
        folds.append((eids[tr], eids[te]))
        logger.info("grouped fold %d/%d — train ep: %d  test ep: %d  (test AD+ ep: %d)",
                    k + 1, n_splits, len(tr), len(te), int(y[te].sum()))
    return folds


def make_cv_folds(
    cohort: pd.DataFrame,
    stratify_col: str = "anyAD",
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified k-fold split by subject_id for the small triple-modality cohort.

    Stratifies on a binary/low-cardinality column (default the patient-level
    `anyAD` label from targets.add_grade_columns) so every fold contains AD
    positives — essential at n~522 where positives are ~14%.

    Parameters
    ----------
    cohort : DataFrame with 'subject_id' and `stratify_col`
    stratify_col : column to stratify on (binary any-AD by default)
    n_splits : number of CV folds
    seed : random seed

    Returns
    -------
    folds : list of (train_ids, test_ids) subject_id arrays, one per fold.
    """
    assert "subject_id" in cohort.columns, "cohort must contain 'subject_id'"
    assert stratify_col in cohort.columns, f"cohort must contain '{stratify_col}'"

    sids = cohort["subject_id"].to_numpy()
    y = cohort[stratify_col].fillna(0).astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k, (tr, te) in enumerate(skf.split(sids, y)):
        train_ids, test_ids = sids[tr], sids[te]
        n_pos = int(y[te].sum())
        logger.info(
            "CV fold %d/%d — train: %d  test: %d  (test AD+: %d)",
            k + 1, n_splits, len(train_ids), len(test_ids), n_pos,
        )
        folds.append((train_ids, test_ids))
    return folds


def make_splits(
    cohort: pd.DataFrame,
    col_root: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    n_bins: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split cohort by subject_id into train / val / test.

    Parameters
    ----------
    cohort : DataFrame with at least ['subject_id', col_root]
    col_root : column name for the aortic root diameter (used for stratification)
    train_frac, val_frac : fractions (test = remainder)
    n_bins : number of equal-frequency bins for stratification
    seed : random seed

    Returns
    -------
    train_ids, val_ids, test_ids : arrays of subject_id values
    """
    assert "subject_id" in cohort.columns, "cohort must contain 'subject_id'"

    subject_ids = cohort["subject_id"].values
    root_vals = cohort[col_root].values

    # Build stratification labels: qcut into equal-frequency bins;
    # NaN values get their own "nan" bin.
    valid_mask = ~np.isnan(root_vals.astype(float))
    strat = np.full(len(subject_ids), "nan", dtype=object)
    if valid_mask.sum() > 0:
        bins = pd.qcut(
            root_vals[valid_mask],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        strat[valid_mask] = bins.astype(str)

    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 0, "train_frac + val_frac must be < 1.0"

    # First split: train vs. (val + test)
    train_ids, temp_ids, train_strat, temp_strat = train_test_split(
        subject_ids,
        strat,
        test_size=(1.0 - train_frac),
        stratify=strat,
        random_state=seed,
    )

    # Second split: val vs. test from the remaining fraction.
    # Fall back to un-stratified split if any stratum has fewer than 2 members
    # (can happen with small cohorts or unusual label distributions).
    val_share = val_frac / (val_frac + test_frac)
    try:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1.0 - val_share),
            stratify=temp_strat,
            random_state=seed,
        )
    except ValueError as e:
        logger.warning(
            "Stratified val/test split failed (%s); falling back to random split.", e
        )
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1.0 - val_share),
            stratify=None,
            random_state=seed,
        )

    # Sanity: no patient leakage
    assert len(set(train_ids) & set(val_ids)) == 0, "Train/val overlap detected"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/test overlap detected"
    assert len(set(val_ids) & set(test_ids)) == 0, "Val/test overlap detected"

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d  (total: %d)",
        len(train_ids),
        len(val_ids),
        len(test_ids),
        len(subject_ids),
    )
    return train_ids, val_ids, test_ids
