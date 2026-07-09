"""
M0 — build the immutable patient-level fold assignments.

The three fusion scripts (GBDT / deep / late) used to each call `make_cv_folds`
at run time. That is fragile: the folds are only identical across scripts as long
as every script sees the exact same cohort in the exact same row order. If any
patient ever loses an embedding, one script's folds silently diverge from the
others' and the OOF predictions stop being comparable patient-for-patient.

This script generates the 5-fold stratified split ONCE from the frozen triple
cohort and writes it to `pretrained_checkpoints/fold_assignments.csv`. Every
trainer then loads this file (see `data.splits.load_fold_assignments`) so all
experiments share one immutable, auditable fold definition.

Run: python scripts/build_fold_assignments.py
Output: pretrained_checkpoints/fold_assignments.csv
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Endpoint / stratification columns carried alongside the fold id so the file is
# self-describing for downstream event-count and subgroup reporting.
CARRY_COLS = ["anyAD", "anyAD_root", "anyAD_asc", "grade_root", "grade_asc", "grade_max"]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.splits import make_cv_folds

    cfg = Config()
    pc = os.path.join(root, "pretrained_checkpoints")
    cohort_path = os.path.join(pc, "cohort_triple.csv")
    out_path = os.path.join(pc, "fold_assignments.csv")

    cohort = pd.read_csv(cohort_path)
    cohort["subject_id"] = cohort["subject_id"].astype(int)
    n = len(cohort)
    log.info("Loaded triple cohort: %d patients from %s", n, cohort_path)

    n_splits = 5
    seed = cfg.data.split_seed
    folds = make_cv_folds(cohort, stratify_col="anyAD", n_splits=n_splits, seed=seed)

    # Each subject appears in exactly one test fold -> that is its fold_id.
    fold_of = {}
    for k, (_, test_ids) in enumerate(folds):
        for s in test_ids:
            fold_of[int(s)] = k
    assert len(fold_of) == n, (
        f"fold assignment covers {len(fold_of)} subjects but cohort has {n}")

    out = pd.DataFrame({"subject_id": cohort["subject_id"]})
    out["fold_id"] = out["subject_id"].map(fold_of).astype(int)
    for c in CARRY_COLS:
        if c in cohort.columns:
            out[c] = cohort[c].values
        else:
            log.warning("cohort missing carry column '%s' — skipping", c)

    out.to_csv(out_path, index=False)
    log.info("Wrote %s (n=%d, n_splits=%d, seed=%d)", out_path, n, n_splits, seed)

    # --- verification / event-count report per fold ---
    log.info("Per-fold event counts:")
    for k in range(n_splits):
        te = out[out.fold_id == k]
        log.info("  fold %d: n=%3d  anyAD+=%2d  root>=4.0+=%2d  asc>=4.0+=%2d",
                 k, len(te), int(te["anyAD"].sum()),
                 int((te.get("anyAD_root", pd.Series(dtype=float)).fillna(0) == 1).sum()),
                 int((te.get("anyAD_asc", pd.Series(dtype=float)).fillna(0) == 1).sum()))
    # sanity: every subject exactly once, folds partition the cohort
    assert out["subject_id"].is_unique, "duplicate subject_id in fold assignments"
    assert set(out["fold_id"].unique()) == set(range(n_splits)), "fold ids not contiguous"
    log.info("OK — %d subjects partitioned into %d folds.", n, n_splits)


if __name__ == "__main__":
    main()
