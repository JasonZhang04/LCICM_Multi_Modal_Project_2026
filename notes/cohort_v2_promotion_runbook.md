# Promoting the timestamp-corrected cohort (cohort_v2_timefix) — runbook

**Status: NOT YET PROMOTED.** The corrected cohort is built and verified in
`pretrained_checkpoints/cohort_v2_timefix/`. Promoting it is a whole-pipeline operation,
not a file copy. This document exists because several steps overwrite artifacts in place
and would silently invalidate saved results if run in the wrong order.

## Why promote

`notes/repairs_2026-09-14.md` (A5): the live cohort contains 1,931 image–episode rows
across 1,572 episodes where the CXR was taken **after** the echo it is meant to predict,
by up to 22.7 h. The rebuilt cohort has **zero**. Cost: 20,682 → 20,211 episodes (−2.3%),
64,516 → 62,539 instance rows (−3.1%).

## The trap

`scripts/build_episode_folds.py` hardcodes its output to
`pretrained_checkpoints/episode_fold_assignments.csv` and **overwrites it in place**.
Every saved result — the nested runs, the A7 comparison, the fine-tuned CXR OOF, the ECG
embeddings — was generated against the current file. Overwriting it without archiving
first makes all of them unreproducible and, worse, quietly non-comparable: a later run
would join predictions to different folds under the same filename.

Likewise `build_episode_cohort.py` defaults `--out-dir` to `pretrained_checkpoints`.

## Ordered procedure

1. **Archive the current state.** Copy `episodes.csv`, `episode_cxr_instances.csv`,
   `episode_fold_assignments.csv`, `ehr_features_episode.csv`, the image cache indexes
   and the geometry/patchpool artifacts to `pretrained_checkpoints/cohort_v1_archive/`.
   Archive `outputs/` runs that must stay citable, or record their commit.

2. **Swap in the corrected cohort** (`episodes.csv`, `episode_cxr_instances.csv`).

3. **Rebuild the folds** (`build_episode_folds.py`). Folds change because the episode
   set changed. Every downstream artifact must then be regenerated on the new folds —
   no mixing.

4. **Re-extract per-image features** for the new instance set: patchpool embeddings,
   geometry features, image cache (`CXR_PREPROC=legacy` and `ckpt_norm`). Most images
   are unchanged, so this is mostly a re-index rather than fresh GPU work — but the
   instance→episode mapping and the per-fold PCA inputs both change.

5. **Rebuild EHR features and the ECG binding** on the new episodes.

6. **Re-run the models** in this order: EHR floor → fine-tuned CXR → ECG waveform →
   nested final model → A7 comparison.

7. **Rebuild the split** (`build_temporal_holdout_v2.py`). The quarantine patient set
   will shift slightly; re-lock it and re-record the counts.

8. **Re-run the test suite.** `test_preprocessing_and_timing.py` should pass with
   `cohort_dir="."`-equivalent paths once promoted; update the parametrization.

## Recommendation

Do this as one deliberate batch **after** the A9 preprocessing question is settled, so
the pipeline is re-run once against final preprocessing rather than twice. Until then,
report current results against the v1 cohort and disclose the timing defect explicitly —
the A5 evidence above is the honest way to state it, and the 2.3% attrition is small
enough that the qualitative conclusions are unlikely to change.

Do **not** promote piecemeal. A half-promoted state (new episodes, old folds) is worse
than either consistent state and is hard to detect after the fact.
