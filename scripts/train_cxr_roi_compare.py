"""
M4c-2 — does an aorta/mediastinum ROI CXR embedding beat the whole-image one?

Compares RAD-DINO whole-image (raddino_embeddings.pt) vs aorta-ROI
(raddino_roi_embeddings.pt) on the n=522 triple cohort, same immutable folds. Two
framings, both with PAIRED bootstrap CIs (ROI minus whole on the same patients):

  A) CXR-only: HGB(CXR) -> diameter (MAE/R2) and -> ge40/ge45 (AUROC).
  B) Residual: EHR-clinical + HGB(CXR) on the RESIDUAL (leakage-safe) -> does ROI
     add size-independent signal the whole-image token doesn't?

Emits standardized OOF (cxr_whole_only, cxr_roi_only) so the ROI model also lands
in make_eval_report's tiered comparison vs the EHR floor.

Run via SLURM: sbatch scripts/slurm_cxr_roi_compare.sh
Outputs: outputs/cxr_roi_compare/{results.json, oof_predictions.csv, run_manifest.json}
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


def oof_clf_plain(X, y, folds, idx_of):
    oof = np.full(len(y), np.nan)
    for tr_ids, te_ids in folds:
        tr = [idx_of[s] for s in tr_ids if s in idx_of and not np.isnan(y[idx_of[s]])]
        te = [idx_of[s] for s in te_ids if s in idx_of and not np.isnan(y[idx_of[s]])]
        if len(np.unique(y[np.asarray(tr)])) < 2:
            continue
        m = _hgb_clf().fit(X[np.asarray(tr)], y[np.asarray(tr)].astype(int))
        oof[np.asarray(te)] = m.predict_proba(X[np.asarray(te)])[:, 1]
    return oof


def oof_reg_plain(X, y, folds, idx_of, min_train=10):
    oof = np.full(len(y), np.nan)
    for tr_ids, te_ids in folds:
        tr = [idx_of[s] for s in tr_ids if s in idx_of and not np.isnan(y[idx_of[s]])]
        te = [idx_of[s] for s in te_ids if s in idx_of and not np.isnan(y[idx_of[s]])]
        if len(tr) < min_train:
            continue
        m = _hgb_reg().fit(X[np.asarray(tr)], y[np.asarray(tr)])
        oof[np.asarray(te)] = m.predict(X[np.asarray(te)])
    return oof


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.data.splits import load_fold_assignments, load_fold_id_map
    from multimodal_aorta.data.ehr import load_ehr_features, FEATURE_COLS
    from multimodal_aorta.data.residualization import residual_oof_regression
    from multimodal_aorta.models.cxr_encoder import CXREmbeddingEncoder
    from multimodal_aorta.training.bootstrap import (
        bootstrap_ci, paired_bootstrap_diff, auroc, mae, r2, fmt)
    from multimodal_aorta.training.oof_io import build_records, save_oof, write_manifest
    import torch

    pc = os.path.join(root, "pretrained_checkpoints")
    out_dir = os.path.join(root, "outputs", "cxr_roi_compare"); os.makedirs(out_dir, exist_ok=True)
    fold_path = os.path.join(pc, "fold_assignments.csv")

    whole = {int(k): v for k, v in CXREmbeddingEncoder.load_embeddings(
        os.path.join(pc, "raddino_embeddings.pt")).items()}
    roi_path = os.path.join(pc, "raddino_roi_embeddings.pt")
    if not os.path.exists(roi_path):
        raise SystemExit(f"missing {roi_path}; run sbatch scripts/slurm_extract_raddino_roi.sh first")
    roi = {int(k): v for k, v in CXREmbeddingEncoder.load_embeddings(roi_path).items()}
    ehr = {int(k): v for k, v in load_ehr_features(os.path.join(pc, "ehr_features.csv")).items()}

    cohort = pd.read_csv(os.path.join(pc, "cohort_triple.csv"))
    cohort["subject_id"] = cohort["subject_id"].astype(int)

    def vec(d, s):
        v = d[s]
        return (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)).astype(np.float32)

    rows, whole_X, roi_X, ehr_X = [], [], [], []
    ehr_dim = len(FEATURE_COLS)
    n_drop = 0
    for r in cohort.itertuples(index=False):
        s = int(r.subject_id)
        if s not in whole or s not in roi:
            n_drop += 1
            continue
        whole_X.append(vec(whole, s)); roi_X.append(vec(roi, s))
        ehr_X.append(np.asarray(ehr.get(s, np.full(ehr_dim, np.nan)), dtype=np.float32))
        rows.append(r)
    df = pd.DataFrame(rows).reset_index(drop=True)
    whole_X = np.vstack(whole_X); roi_X = np.vstack(roi_X); ehr_X = np.vstack(ehr_X)
    idx_of = {int(s): i for i, s in enumerate(df["subject_id"])}
    folds = load_fold_assignments(fold_path, n_splits=5)
    fold_map = load_fold_id_map(fold_path)
    subj = [int(s) for s in df["subject_id"]]
    fold_ids = [fold_map.get(s, -1) for s in subj]
    log.info("Assembled n=%d (dropped %d missing whole/ROI); whole%s roi%s",
             len(df), n_drop, whole_X.shape, roi_X.shape)

    results = {"n_patients": int(len(df)), "roi": "upper_mediastinum(0.25,0.08,0.75,0.60)", "sites": {}}
    records = []

    for site in ("root", "asc"):
        diam = df[f"target_{site}"].to_numpy(float)
        site_res = {"cxr_only": {}, "residual": {}}

        # ---- A) CXR-only ----
        for name, X in [("whole", whole_X), ("roi", roi_X)]:
            dpred = oof_reg_plain(X, diam, folds, idx_of)
            entry = {"diam_mae": fmt(bootstrap_ci(diam, dpred, mae, need_both_classes=False)),
                     "diam_r2": fmt(bootstrap_ci(diam, dpred, r2, need_both_classes=False))}
            for ep, thr in [("ge40", 4.0), ("ge45", 4.5)]:
                y = np.where(np.isnan(diam), np.nan, (diam >= thr).astype(float))
                p = oof_clf_plain(X, y, folds, idx_of)
                entry[f"{ep}_auroc"] = fmt(bootstrap_ci(y, p, auroc))
                if name == "roi":
                    records += build_records(
                        subject_ids=subj, fold_ids=fold_ids, model_name="cxr_roi_only",
                        modality_set="cxr", site=site, endpoint=ep, target_type="binary",
                        y_true=y, pred_prob=p, has_ecg=0, has_ehr=0)
            site_res["cxr_only"][name] = entry
        # paired ROI - whole (diameter)
        dpred_w = oof_reg_plain(whole_X, diam, folds, idx_of)
        dpred_r = oof_reg_plain(roi_X, diam, folds, idx_of)
        site_res["cxr_only"]["delta_roi_minus_whole"] = {
            "mae": fmt(paired_bootstrap_diff(diam, dpred_r, dpred_w, mae, need_both_classes=False)),
            "r2": fmt(paired_bootstrap_diff(diam, dpred_r, dpred_w, r2, need_both_classes=False))}
        for ep, thr in [("ge40", 4.0), ("ge45", 4.5)]:
            y = np.where(np.isnan(diam), np.nan, (diam >= thr).astype(float))
            pw = oof_clf_plain(whole_X, y, folds, idx_of)
            pr = oof_clf_plain(roi_X, y, folds, idx_of)
            d = paired_bootstrap_diff(y, pr, pw, auroc)
            site_res["cxr_only"]["delta_roi_minus_whole"][f"{ep}_auroc"] = fmt(d)
            site_res["cxr_only"]["delta_roi_minus_whole"][f"{ep}_excl0"] = bool(d[1] > 0 or d[2] < 0)

        # ---- B) Residual (EHR clinical + CXR residual) ----
        for name, X in [("whole", whole_X), ("roi", roi_X)]:
            fin, clin = residual_oof_regression(diam, ehr_X, X, folds, idx_of, _hgb_reg, _hgb_reg)
            site_res["residual"][name] = {
                "clinical_mae": fmt(bootstrap_ci(diam, clin, mae, need_both_classes=False)),
                "recon_mae": fmt(bootstrap_ci(diam, fin, mae, need_both_classes=False)),
                "recon_r2": fmt(bootstrap_ci(diam, fin, r2, need_both_classes=False)),
                "delta_recon_minus_clinical_mae":
                    fmt(paired_bootstrap_diff(diam, fin, clin, mae, need_both_classes=False))}
            if name == "roi":
                records += build_records(
                    subject_ids=subj, fold_ids=fold_ids, model_name="cxr_roi_residual",
                    modality_set="cxr+ehr", site=site, endpoint="diam", target_type="regression",
                    y_true=diam, pred_value=fin, has_ecg=0)
        # paired residual ROI - whole
        fin_w, _ = residual_oof_regression(diam, ehr_X, whole_X, folds, idx_of, _hgb_reg, _hgb_reg)
        fin_r, _ = residual_oof_regression(diam, ehr_X, roi_X, folds, idx_of, _hgb_reg, _hgb_reg)
        site_res["residual"]["delta_roi_minus_whole_mae"] = fmt(
            paired_bootstrap_diff(diam, fin_r, fin_w, mae, need_both_classes=False))

        results["sites"][site] = site_res
        log.info("[%s] CXR-only diam MAE whole=%s roi=%s | delta(roi-whole)=%s",
                 site, site_res["cxr_only"]["whole"]["diam_mae"],
                 site_res["cxr_only"]["roi"]["diam_mae"],
                 site_res["cxr_only"]["delta_roi_minus_whole"]["mae"])
        log.info("[%s] CXR-only ge40 AUROC whole=%s roi=%s (delta %s excl0=%s)", site,
                 site_res["cxr_only"]["whole"]["ge40_auroc"], site_res["cxr_only"]["roi"]["ge40_auroc"],
                 site_res["cxr_only"]["delta_roi_minus_whole"]["ge40_auroc"],
                 site_res["cxr_only"]["delta_roi_minus_whole"]["ge40_excl0"])

    save_oof(records, os.path.join(out_dir, "oof_predictions.csv"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_manifest(os.path.join(out_dir, "run_manifest.json"), model_name="cxr_roi_compare",
                   seed=SEED, n_patients=len(df),
                   cohort_csv=os.path.join(pc, "cohort_triple.csv"), fold_csv=fold_path)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
