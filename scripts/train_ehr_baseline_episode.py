"""
Episode-level EHR-only baseline (the lower-bound "floor") for the rebuild.

Two things change vs the patient-level train_ehr_baseline.py, both structural:

  1. UNIT = episode. Features come from ehr_features_episode.csv (one causal,
     pre-index row per (patient, echo study)); the label is that episode's own
     diameter. Keyed by episode_id, grouped by subject_id.

  2. PROPERLY NESTED OOF (fixes audit Problem 4). The old floor produced OOF with
     an internal StratifiedKFold that was NOT aligned to the outer evaluation
     folds, so a held-out patient could have been in the floor's own training
     split — a documented second-order leak. Here the OOF is generated with the
     SINGLE immutable patient-grouped fold file: an episode in fold k is predicted
     only by a model trained on folds != k, and because folds are patient-grouped,
     none of that patient's other episodes leak in. Nested by construction.

Because every episode in this cohort already has imaging, the floor and the
multimodal models now train on the SAME episodes with the SAME folds — the
"equal-n" comparison the audit asked for, with no full-cohort-vs-522 confound.

Models: HGB (primary, NaN-native) + LogisticRegression/Ridge (linear, imputed).
Metrics use the patient-CLUSTERED bootstrap.

Run: sbatch scripts/slurm_train_ehr_baseline_episode.sh
Outputs: outputs/ehr_baseline_episode/{results.json, oof_predictions.csv, run_manifest.json}
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SEED = 42
FEATURE_COLS = [
    "age", "sex", "height_cm", "weight_kg", "bmi", "bsa", "sbp", "dbp",
    "height_missing", "weight_missing", "bsa_missing", "bp_missing",
]


def _hgb_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, class_weight="balanced", random_state=0)


def _hgb_reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def _logreg_pipe():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                                solver="lbfgs", max_iter=5000))])


def _ridge_pipe():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("reg", Ridge(alpha=10.0))])


def nested_oof(model_factory, X, y, fold_of_row, is_clf):
    """
    OOF prediction using the predefined outer folds. For each fold k, fit on all
    rows with a non-NaN label whose fold != k, predict rows whose fold == k. Rows
    with NaN y are never predicted (stay NaN). Patient grouping lives in the fold
    assignment, so training on folds != k excludes every episode of fold-k patients.
    """
    from sklearn.base import clone
    n = len(y)
    out = np.full(n, np.nan)
    folds = sorted(set(int(f) for f in fold_of_row if f >= 0))
    labelled = ~np.isnan(y)
    for k in folds:
        tr = np.where(labelled & (fold_of_row != k))[0]
        te = np.where(fold_of_row == k)[0]
        if len(tr) < 20 or len(te) == 0:
            continue
        if is_clf and len(np.unique(y[tr])) < 2:
            continue
        m = clone(model_factory()).fit(X[tr], y[tr].astype(int) if is_clf else y[tr])
        if is_clf:
            out[te] = m.predict_proba(X[te])[:, 1]
        else:
            out[te] = m.predict(X[te])
    return out


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.data.episodes import load_episodes
    from multimodal_aorta.data.splits import load_episode_fold_id_map
    from multimodal_aorta.training.bootstrap import (
        cluster_bootstrap_ci, auroc, auprc, mae, r2, fmt)
    from multimodal_aorta.training.oof_io import build_records, save_oof, write_manifest

    pc = os.path.join(root, "pretrained_checkpoints")
    out_dir = os.path.join(root, "outputs", "ehr_baseline_episode"); os.makedirs(out_dir, exist_ok=True)
    fold_path = os.path.join(pc, "episode_fold_assignments.csv")

    ep = load_episodes(pc, require_ecg=False)
    ehr = pd.read_csv(os.path.join(pc, "ehr_features_episode.csv"))
    ehr["episode_id"] = ehr["episode_id"].astype(str)
    ep["episode_id"] = ep["episode_id"].astype(str)
    # align EHR features to the episode order
    df = ep[["episode_id", "subject_id", "target_root", "target_asc"]].merge(
        ehr[["episode_id"] + FEATURE_COLS], on="episode_id", how="left")
    assert len(df) == len(ep), "EHR/episode row mismatch"

    fold_map = load_episode_fold_id_map(fold_path)
    fold_of_row = np.array([fold_map.get(e, -1) for e in df["episode_id"]], dtype=int)
    n_missing_fold = int((fold_of_row < 0).sum())
    if n_missing_fold:
        log.warning("%d episodes have no fold id", n_missing_fold)
    X = df[FEATURE_COLS].to_numpy(np.float32)
    groups = df["subject_id"].to_numpy(int)
    eids = df["episode_id"].tolist()
    log.info("Episodes: %d / patients %d | features %d", len(df), df.subject_id.nunique(), X.shape[1])

    results = {"n_episodes": len(df), "n_patients": int(df.subject_id.nunique()), "sites": {}}
    records = []
    for site in ("root", "asc"):
        d = df[f"target_{site}"].to_numpy(float)
        site_res = {}
        for ep_name, thr in [("ge40", 4.0), ("ge45", 4.5)]:
            y = np.where(np.isnan(d), np.nan, (d >= thr).astype(float))
            p_hgb = nested_oof(_hgb_clf, X, y, fold_of_row, is_clf=True)
            p_lr = nested_oof(_logreg_pipe, X, y, fold_of_row, is_clf=True)
            site_res[ep_name] = {
                "pos": int(np.nansum(y)),
                "hgb_auroc": fmt(cluster_bootstrap_ci(y, p_hgb, groups, auroc)),
                "hgb_auprc": fmt(cluster_bootstrap_ci(y, p_hgb, groups, auprc)),
                "logreg_auroc": fmt(cluster_bootstrap_ci(y, p_lr, groups, auroc)),
            }
            log.info("[%s %s] HGB AUROC=%s (pos ep=%d)", site, ep_name,
                     site_res[ep_name]["hgb_auroc"], site_res[ep_name]["pos"])
            records += build_records(
                episode_ids=eids, subject_ids=groups,
                fold_ids=fold_of_row.tolist(), model_name="ehr_baseline_episode",
                modality_set="ehr", site=site, endpoint=ep_name, target_type="binary",
                y_true=y, pred_prob=p_hgb, has_ecg=0, has_cxr=0, has_ehr=1)
        # diameter regression
        p_hgb_r = nested_oof(_hgb_reg, X, d, fold_of_row, is_clf=False)
        p_rid_r = nested_oof(_ridge_pipe, X, d, fold_of_row, is_clf=False)
        site_res["diam"] = {
            "hgb_mae": fmt(cluster_bootstrap_ci(d, p_hgb_r, groups, mae, need_both_classes=False)),
            "hgb_r2": fmt(cluster_bootstrap_ci(d, p_hgb_r, groups, r2, need_both_classes=False)),
            "ridge_r2": fmt(cluster_bootstrap_ci(d, p_rid_r, groups, r2, need_both_classes=False)),
        }
        log.info("[%s diam] HGB R2=%s MAE=%s | Ridge R2=%s", site,
                 site_res["diam"]["hgb_r2"], site_res["diam"]["hgb_mae"], site_res["diam"]["ridge_r2"])
        records += build_records(
            episode_ids=eids, subject_ids=groups,
            fold_ids=fold_of_row.tolist(), model_name="ehr_baseline_episode",
            modality_set="ehr", site=site, endpoint="diam", target_type="regression",
            y_true=d, pred_value=p_hgb_r, has_ecg=0, has_cxr=0, has_ehr=1)
        results["sites"][site] = site_res

    save_oof(records, os.path.join(out_dir, "oof_predictions.csv"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_manifest(os.path.join(out_dir, "run_manifest.json"),
                   model_name="ehr_baseline_episode", seed=SEED, n_patients=len(df),
                   cohort_csv=os.path.join(pc, "episodes.csv"), fold_csv=fold_path)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
