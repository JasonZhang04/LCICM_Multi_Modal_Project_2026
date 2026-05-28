"""
Patient-level train / val / test splitting with stratification.

Stratification is done on a binned version of the aortic root diameter
(sinus_diam_cm) so that label distribution is balanced across splits.
Patients with NaN root diameter are binned into their own stratum so they
are also distributed proportionally.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


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
