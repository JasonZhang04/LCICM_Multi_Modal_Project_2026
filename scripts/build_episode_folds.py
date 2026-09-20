"""
Build the immutable, patient-grouped, episode-level CV fold assignments.

Episode analogue of scripts/build_fold_assignments.py. Two invariants matter:
  - GROUPING: every episode of a patient goes to the same fold, so a patient's
    correlated repeated echoes never straddle a train/test boundary.
  - STRATIFICATION: folds are balanced on episode-level any-AD so each fold holds
    a comparable number of positives.

Writes `pretrained_checkpoints/episode_fold_assignments.csv` once so every trainer
loads one auditable fold definition (see data.splits.load_episode_folds).

Run: python scripts/build_episode_folds.py
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

N_SPLITS = 5
SEED = 42
CARRY = ["anyAD", "anyAD_root", "anyAD_asc", "grade_root", "grade_asc", "grade_max",
         "target_root", "target_asc", "n_cxr", "n_ecg"]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.data.episodes import load_episodes, patient_level_label
    from sklearn.model_selection import StratifiedGroupKFold

    pc = os.path.join(root, "pretrained_checkpoints")
    out_path = os.path.join(pc, "episode_fold_assignments.csv")

    ep = load_episodes(pc, require_ecg=False)
    log.info("Episodes: %d / patients %d", len(ep), ep.subject_id.nunique())

    # Patient-grouped stratified split. Stratify each episode on its own any-AD
    # but keep patients intact via the group array.
    eids = ep["episode_id"].to_numpy()
    groups = ep["subject_id"].to_numpy()
    y = ep["anyAD"].fillna(0).astype(int).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_of = {}
    for k, (_, te) in enumerate(sgkf.split(eids, y, groups)):
        for e in eids[te]:
            fold_of[e] = k
    assert len(fold_of) == len(ep), "not every episode was assigned a fold"

    out = ep[["episode_id", "subject_id", "measurement_id"]].copy()
    out["fold_id"] = out["episode_id"].map(fold_of).astype(int)
    for c in CARRY:
        if c in ep.columns:
            out[c] = ep[c].values

    # --- invariant checks ---
    per_pt = out.groupby("subject_id")["fold_id"].nunique()
    n_span = int((per_pt > 1).sum())
    assert n_span == 0, f"{n_span} patients span multiple folds — grouping broken"
    assert out["episode_id"].is_unique, "duplicate episode_id"
    assert set(out["fold_id"].unique()) == set(range(N_SPLITS)), "fold ids not contiguous"

    # GUARD: this file is the project's "immutable" fold definition. Every saved OOF
    # prediction, embedding and metric was generated against it, and they are joined by
    # episode_id with no fold provenance recorded. Silently overwriting it would make
    # existing results non-comparable in a way that is very hard to detect afterwards.
    # Overwrite only deliberately: FORCE=1, and the old file is archived beside it.
    if os.path.exists(out_path) and os.environ.get("FORCE", "0") != "1":
        raise SystemExit(
            f"{out_path} already exists.\n"
            "Existing results were generated against it; regenerating changes fold\n"
            "membership and invalidates them. See notes/cohort_v2_promotion_runbook.md.\n"
            "To proceed deliberately: FORCE=1 python scripts/build_episode_folds.py")
    if os.path.exists(out_path):
        import shutil, time
        bak = out_path.replace(".csv", f".bak_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy2(out_path, bak)
        log.warning("FORCE=1: archived previous fold assignments -> %s", bak)
    out.to_csv(out_path, index=False)
    log.info("Wrote %s (episodes=%d, patients=%d, folds=%d, seed=%d)",
             out_path, len(out), out.subject_id.nunique(), N_SPLITS, SEED)

    # --- per-fold event-count report ---
    log.info("Per-fold counts (episodes | patients | root>=4.0 ep | asc>=4.0 ep):")
    for k in range(N_SPLITS):
        te = out[out.fold_id == k]
        log.info("  fold %d: ep=%5d  pt=%5d  root+=%4d  asc+=%4d",
                 k, len(te), te.subject_id.nunique(),
                 int((te.get("anyAD_root", pd.Series(dtype=float)).fillna(0) == 1).sum()),
                 int((te.get("anyAD_asc", pd.Series(dtype=float)).fillna(0) == 1).sum()))
    # positives-per-fold balance sanity
    pos = [int((out[out.fold_id == k]["anyAD"] == 1).sum()) for k in range(N_SPLITS)]
    log.info("anyAD+ episodes per fold: %s (min %d, max %d)", pos, min(pos), max(pos))
    log.info("OK.")


if __name__ == "__main__":
    main()
