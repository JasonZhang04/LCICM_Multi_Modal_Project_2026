"""
M3 — residualized & body-size-indexed diameter modeling.

Answers the project's core question head-on: do ECG+CXR add aortic-size signal
beyond age/sex/body-size? Three target modes on the n=522 triple cohort, same
immutable folds as every other model:

  clinical_only : HGB(EHR-12) -> diameter           (within-protocol EHR floor)
  residual      : clinical_only + HGB(ECG+CXR) on the RESIDUAL (leakage-safe,
                  clinical model refit inside each fold) -> reconstructed diameter
  indexed       : HGB(ECG+CXR+EHR) -> diameter / BSA  (size-indexed regression)

Headline: the PAIRED bootstrap delta of `residual` minus `clinical_only` on the
SAME held-out patients. If its CI excludes 0 (MAE down / R2 up), imaging adds
size-independent signal; if not, it doesn't — a clean, defensible answer either way.

Run via SLURM: sbatch scripts/slurm_residual_fusion.sh
Outputs: outputs/residual_fusion/{results.json, oof_predictions.csv, run_manifest.json}
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


def _hgb_reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def oof_reg_plain(X, y, folds, idx_of, min_train=10):
    """Plain OOF regression (used for the indexed target)."""
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
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.splits import load_fold_assignments, load_fold_id_map
    from multimodal_aorta.data.ehr import load_ehr_features, FEATURE_COLS
    from multimodal_aorta.data.targets import add_indexed_targets
    from multimodal_aorta.data.residualization import residual_oof_regression
    from multimodal_aorta.models.ecg_encoder import PCLREmbeddingEncoder
    from multimodal_aorta.models.cxr_encoder import CXREmbeddingEncoder
    from multimodal_aorta.training.bootstrap import (
        bootstrap_ci, paired_bootstrap_diff, auroc, mae, r2, fmt)
    from multimodal_aorta.training.oof_io import build_records, save_oof, write_manifest
    import torch

    cfg = Config()
    pc = os.path.join(root, "pretrained_checkpoints")
    out_dir = os.path.join(root, "outputs", "residual_fusion"); os.makedirs(out_dir, exist_ok=True)
    fold_path = os.path.join(pc, "fold_assignments.csv")

    cohort = pd.read_csv(os.path.join(pc, "cohort_triple.csv"))
    cohort["subject_id"] = cohort["subject_id"].astype(int)
    pclr = {int(k): v for k, v in PCLREmbeddingEncoder.load_embeddings(
        os.path.join(pc, "pclr_embeddings.pt")).items()}
    raddino = {int(k): v for k, v in CXREmbeddingEncoder.load_embeddings(
        os.path.join(pc, "raddino_embeddings.pt")).items()}
    ehr = {int(k): v for k, v in load_ehr_features(os.path.join(pc, "ehr_features.csv")).items()}

    # BSA / height for indexed targets (named columns in the EHR cache)
    ecsv = pd.read_csv(os.path.join(pc, "ehr_features.csv"))
    bsa_map = dict(zip(ecsv.subject_id.astype(int), ecsv["bsa"]))
    height_map = dict(zip(ecsv.subject_id.astype(int), ecsv["height_cm"]))
    cohort = add_indexed_targets(cohort, bsa_map, height_map)

    # assemble aligned matrices over subjects with both embeddings
    def vec(d, s):
        v = d[s]
        return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    rows, ehr_X, img_X = [], [], []
    ehr_dim = len(FEATURE_COLS)
    for r in cohort.itertuples(index=False):
        s = int(r.subject_id)
        if s not in pclr or s not in raddino:
            continue
        ehr_X.append(np.asarray(ehr.get(s, np.full(ehr_dim, np.nan)), dtype=np.float32))
        img_X.append(np.concatenate([vec(pclr, s).astype(np.float32),
                                      vec(raddino, s).astype(np.float32)]))
        rows.append(r)
    df = pd.DataFrame(rows).reset_index(drop=True)
    ehr_X = np.vstack(ehr_X); img_X = np.vstack(img_X)
    concat_X = np.concatenate([img_X, ehr_X], axis=1)
    idx_of = {int(s): i for i, s in enumerate(df["subject_id"])}
    folds = load_fold_assignments(fold_path, n_splits=5)
    fold_map = load_fold_id_map(fold_path)
    log.info("Assembled n=%d (ehr%s img%s)", len(df), ehr_X.shape, img_X.shape)

    results = {"n_patients": int(len(df)), "sites": {}}
    records = []
    subj = [int(s) for s in df["subject_id"]]
    fold_ids = [fold_map.get(s, -1) for s in subj]

    for site in ("root", "asc"):
        diam = df[f"target_{site}"].to_numpy(float)
        idx = df[f"{site}_diam_per_bsa"].to_numpy(float)

        # --- residual vs clinical-only (leakage-safe) ---
        oof_final, oof_clin = residual_oof_regression(
            diam, ehr_X, img_X, folds, idx_of, _hgb_reg, _hgb_reg)

        clin_mae = bootstrap_ci(diam, oof_clin, mae, need_both_classes=False)
        fin_mae = bootstrap_ci(diam, oof_final, mae, need_both_classes=False)
        clin_r2 = bootstrap_ci(diam, oof_clin, r2, need_both_classes=False)
        fin_r2 = bootstrap_ci(diam, oof_final, r2, need_both_classes=False)
        d_mae = paired_bootstrap_diff(diam, oof_final, oof_clin, mae, need_both_classes=False)
        d_r2 = paired_bootstrap_diff(diam, oof_final, oof_clin, r2, need_both_classes=False)

        # --- derived binary AUROC by ranking on predicted diameter ---
        derived = {}
        for ep, thr in [("ge40", 4.0), ("ge45", 4.5)]:
            y = np.where(np.isnan(diam), np.nan, (diam >= thr).astype(float))
            a_c = bootstrap_ci(y, oof_clin, auroc)
            a_f = bootstrap_ci(y, oof_final, auroc)
            d_a = paired_bootstrap_diff(y, oof_final, oof_clin, auroc)
            derived[ep] = {"pos": int(np.nansum(y)), "clinical_auroc": fmt(a_c),
                           "residual_auroc": fmt(a_f), "delta_auroc": fmt(d_a),
                           "delta_excludes_0": bool(d_a[1] > 0 or d_a[2] < 0)}

        # --- indexed-diameter regression (secondary) ---
        oof_idx = oof_reg_plain(concat_X, idx, folds, idx_of)
        idx_mae = bootstrap_ci(idx, oof_idx, mae, need_both_classes=False)
        idx_r2 = bootstrap_ci(idx, oof_idx, r2, need_both_classes=False)

        results["sites"][site] = {
            "clinical_only": {"mae": fmt(clin_mae), "r2": fmt(clin_r2)},
            "residual_fusion": {"mae": fmt(fin_mae), "r2": fmt(fin_r2)},
            "delta_residual_minus_clinical": {
                "mae": fmt(d_mae), "r2": fmt(d_r2),
                "mae_excludes_0": bool(d_mae[1] > 0 or d_mae[2] < 0),
                "r2_excludes_0": bool(d_r2[1] > 0 or d_r2[2] < 0)},
            "derived_binary": derived,
            "indexed_per_bsa": {"mae": fmt(idx_mae), "r2": fmt(idx_r2),
                                "n": int(np.sum(~np.isnan(idx)))},
        }
        log.info("[%s] clinical MAE=%s R2=%s | residual MAE=%s R2=%s",
                 site, fmt(clin_mae), fmt(clin_r2), fmt(fin_mae), fmt(fin_r2))
        log.info("[%s] DELTA residual-clinical: MAE=%s (excl0=%s) R2=%s (excl0=%s)",
                 site, fmt(d_mae), results["sites"][site]["delta_residual_minus_clinical"]["mae_excludes_0"],
                 fmt(d_r2), results["sites"][site]["delta_residual_minus_clinical"]["r2_excludes_0"])

        # standardized OOF (regression) for the global report
        records += build_records(
            subject_ids=subj, fold_ids=fold_ids, model_name="residual_fusion",
            modality_set="ecg+cxr+ehr", site=site, endpoint="diam", target_type="regression",
            y_true=diam, pred_value=oof_final, has_ehr=[int(s in ehr) for s in subj])
        records += build_records(
            subject_ids=subj, fold_ids=fold_ids, model_name="clinical_only_diam",
            modality_set="ehr", site=site, endpoint="diam", target_type="regression",
            y_true=diam, pred_value=oof_clin, has_ehr=[int(s in ehr) for s in subj])

    save_oof(records, os.path.join(out_dir, "oof_predictions.csv"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_manifest(
        os.path.join(out_dir, "run_manifest.json"), model_name="residual_fusion",
        seed=SEED, n_patients=len(df),
        cohort_csv=os.path.join(pc, "cohort_triple.csv"), fold_csv=fold_path)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
