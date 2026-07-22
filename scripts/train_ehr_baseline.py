"""
M1 — EHR-only clinical baseline (the lower-bound "floor").

This is the reference every multimodal model is measured against. It is deliberately
NOT the headline model — it exists to answer "does ECG/CXR add value beyond age, sex,
body size, and blood pressure?" (see scripts/make_eval_report.py paired deltas).

Design (mirrors the late-fusion Level-0 pattern in train_late_fusion.py):
  - Train EHR-only models on the FULL EHR cohort (~4525 labelled patients) via
    leakage-safe `cross_val_predict` -> out-of-fold probability for every patient.
    Using the full cohort makes this the strongest (most conservative) floor.
  - Report out-of-fold performance on the n=522 triple cohort, tagged with the
    immutable outer-fold ids so make_eval_report can compare it patient-for-patient
    against the fusion models on the same patients.

Models: HistGradientBoosting (primary; NaN-native, matches the late-fusion base
learner and reproduces its ~0.78/0.85 standalone EHR AUROC), plus LogisticRegression
/ Ridge (linear, with in-fold median-impute + standardize). Minimal 12-feature set.

Leakage note: like train_late_fusion.py, the full-cohort internal CV is not nested
inside the outer folds (a documented second-order optimism); M4 addresses nesting.

Run: python scripts/train_ehr_baseline.py
Outputs: outputs/ehr_baseline/{results.json, oof_predictions.csv, run_manifest.json}
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

ROOT_RANGE = (1.5, 6.0)
ASC_RANGE = (1.5, 7.0)
SEED = 42


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
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                                   solver="lbfgs", max_iter=5000)),
    ])


def _ridge_pipe():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=10.0)),
    ])


def feature_matrix(ehr, subjects):
    """Stack EHR feature vectors for subjects present in the EHR dict."""
    X, keep = [], []
    for s in subjects:
        if s in ehr:
            X.append(np.asarray(ehr[s], dtype=np.float32))
            keep.append(s)
    return np.vstack(X), keep


def oof_clf(model_factory, ehr, label_map, threshold, subjects, seed=SEED):
    """Leakage-safe OOF probability for a binary threshold, over the full cohort. -> {sid: prob}."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    valid = [s for s in subjects if s in ehr and not np.isnan(label_map.get(s, np.nan))]
    X, keep = feature_matrix(ehr, valid)
    y = np.array([1 if label_map[s] >= threshold else 0 for s in keep])
    if len(np.unique(y)) < 2:
        return {}
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(model_factory(), X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    return dict(zip(keep, p))


def oof_reg(model_factory, ehr, label_map, subjects, seed=SEED):
    from sklearn.model_selection import KFold, cross_val_predict
    valid = [s for s in subjects if s in ehr and not np.isnan(label_map.get(s, np.nan))]
    X, keep = feature_matrix(ehr, valid)
    y = np.array([label_map[s] for s in keep], dtype=float)
    kf = KFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(model_factory(), X, y, cv=kf, n_jobs=1)
    return dict(zip(keep, p))


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.ehr import load_ehr_features
    from multimodal_aorta.data.splits import load_fold_id_map
    from multimodal_aorta.training.bootstrap import bootstrap_ci, auroc, auprc, mae, r2, fmt
    from multimodal_aorta.training.oof_io import build_records, save_oof, write_manifest

    cfg = Config()
    pc = os.path.join(root, "pretrained_checkpoints")
    out_dir = os.path.join(root, "outputs", "ehr_baseline"); os.makedirs(out_dir, exist_ok=True)
    fold_path = os.path.join(pc, "fold_assignments.csv")

    # labels for ALL subjects, QC-clipped identically to the other scripts
    lab = pd.read_csv(cfg.data.echo_labels_path)

    def clip(col, lo, hi):
        v = lab[col].copy()
        v[(v < lo) | (v > hi)] = np.nan
        return v

    root_diam = dict(zip(lab.subject_id.astype(int), clip("sinus_diam_cm", *ROOT_RANGE)))
    asc_diam = dict(zip(lab.subject_id.astype(int), clip("ascending_diam_cm", *ASC_RANGE)))
    ehr = {int(k): v for k, v in load_ehr_features(os.path.join(pc, "ehr_features.csv")).items()}
    all_subjects = sorted(int(s) for s in lab.subject_id)
    log.info("EHR cohort: %d subjects with features; %d labelled", len(ehr), len(all_subjects))

    # n=522 triple cohort + immutable outer folds
    cohort = pd.read_csv(os.path.join(pc, "cohort_triple.csv"))
    s522 = cohort["subject_id"].astype(int).tolist()
    fold_map = load_fold_id_map(fold_path)

    results = {"n_full_cohort": len(all_subjects), "n_eval": len(s522), "sites": {}}
    records = []   # standardized OOF for the PRIMARY (HGB) model on the 522

    for site, dmap, rng_ in [("root", root_diam, ROOT_RANGE), ("asc", asc_diam, ASC_RANGE)]:
        diam522 = np.array([dmap.get(s, np.nan) for s in s522])
        site_res = {}
        # ---------- binary endpoints ----------
        for ep, thr in [("ge40", 4.0), ("ge45", 4.5)]:
            y522 = np.where(np.isnan(diam522), np.nan, (diam522 >= thr).astype(float))
            hgb = oof_clf(_hgb_clf, ehr, dmap, thr, all_subjects)
            lr = oof_clf(_logreg_pipe, ehr, dmap, thr, all_subjects)
            p_hgb = np.array([hgb.get(s, np.nan) for s in s522])
            p_lr = np.array([lr.get(s, np.nan) for s in s522])
            site_res[ep] = {
                "pos": int(np.nansum(y522)),
                "hgb_auroc": fmt(bootstrap_ci(y522, p_hgb, auroc)),
                "hgb_auprc": fmt(bootstrap_ci(y522, p_hgb, auprc)),
                "logreg_auroc": fmt(bootstrap_ci(y522, p_lr, auroc)),
                "logreg_auprc": fmt(bootstrap_ci(y522, p_lr, auprc)),
            }
            log.info("[%s %s] HGB AUROC=%s | LogReg AUROC=%s (pos=%d)", site, ep,
                     site_res[ep]["hgb_auroc"], site_res[ep]["logreg_auroc"], site_res[ep]["pos"])
            # PRIMARY (HGB) -> standardized OOF
            records += build_records(
                subject_ids=s522, fold_ids=[fold_map.get(s, -1) for s in s522],
                model_name="ehr_baseline", modality_set="ehr", site=site, endpoint=ep,
                target_type="binary", y_true=y522, pred_prob=p_hgb, has_ehr=1)
        # ---------- diameter regression ----------
        hgb_r = oof_reg(_hgb_reg, ehr, dmap, all_subjects)
        rid_r = oof_reg(_ridge_pipe, ehr, dmap, all_subjects)
        p_hgb_r = np.array([hgb_r.get(s, np.nan) for s in s522])
        p_rid_r = np.array([rid_r.get(s, np.nan) for s in s522])
        site_res["diam"] = {
            "hgb_mae": fmt(bootstrap_ci(diam522, p_hgb_r, mae, need_both_classes=False)),
            "hgb_r2": fmt(bootstrap_ci(diam522, p_hgb_r, r2, need_both_classes=False)),
            "ridge_mae": fmt(bootstrap_ci(diam522, p_rid_r, mae, need_both_classes=False)),
            "ridge_r2": fmt(bootstrap_ci(diam522, p_rid_r, r2, need_both_classes=False)),
        }
        log.info("[%s diam] HGB MAE=%s R2=%s | Ridge MAE=%s R2=%s", site,
                 site_res["diam"]["hgb_mae"], site_res["diam"]["hgb_r2"],
                 site_res["diam"]["ridge_mae"], site_res["diam"]["ridge_r2"])
        records += build_records(
            subject_ids=s522, fold_ids=[fold_map.get(s, -1) for s in s522],
            model_name="ehr_baseline", modality_set="ehr", site=site, endpoint="diam",
            target_type="regression", y_true=diam522, pred_value=p_hgb_r, has_ehr=1)
        results["sites"][site] = site_res

    save_oof(records, os.path.join(out_dir, "oof_predictions.csv"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_manifest(
        os.path.join(out_dir, "run_manifest.json"), model_name="ehr_baseline",
        seed=SEED, n_patients=len(s522),
        cohort_csv=os.path.join(pc, "cohort_triple.csv"), fold_csv=fold_path)
    log.info("Saved -> %s (+ oof_predictions.csv, run_manifest.json)", out_dir)


if __name__ == "__main__":
    main()
