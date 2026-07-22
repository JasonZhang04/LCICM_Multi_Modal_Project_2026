"""
M4a/b — stronger, leakage-safe late fusion (v2).

Late fusion is the only model that exploits the full per-modality cohorts (the
biggest lever we found: full-cohort EHR root-diam R2 0.31 vs 0.11 on the 522) and
the only one that trends above the EHR floor (ascending-diameter regression). But
v1 still loses to EHR-alone on root because its LogisticRegression meta-learner
over [p_ecg, p_cxr, p_ehr] overfits/dilutes the strong EHR signal at ~32 positives.

v2 targets exactly those failure modes:
  1. Inject EHR's CONTINUOUS diameter prediction (d_ehr) as a meta-feature — the
     binary p_ehr discards the well-calibrated size signal.
  2. Add the aorta-ROI CXR base learner alongside whole-image (ROI helped root).
  3. Nested inner-CV selection of the meta-learner's regularization + feature
     standardization, so the blend can't overfit and drop EHR.
  4. Report the PAIRED delta vs EHR-only (same full-cohort protocol) — the clean
     "does the blend finally beat the floor?" test — plus regression + binary.

All base learners are full-cohort OOF (cross_val_predict), so a patient's meta
features never used its own label. Emits standardized OOF as `late_fusion_v2`.

Run via SLURM: sbatch scripts/slurm_late_fusion_v2.sh
Outputs: outputs/late_fusion_v2/{results.json, oof_predictions.csv, run_manifest.json}
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
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
ALPHA_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]


def _clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, class_weight="balanced", random_state=0)


def _reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def feature_matrix(mod_dict, subjects):
    import torch
    X, keep = [], []
    for s in subjects:
        if s in mod_dict:
            v = mod_dict[s]
            v = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            X.append(v.astype(np.float32)); keep.append(s)
    return np.vstack(X), keep


def unimodal_oof_clf(mod_dict, label_map, threshold, subjects, seed=SEED):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    valid = [s for s in subjects if s in mod_dict and not np.isnan(label_map.get(s, np.nan))]
    X, keep = feature_matrix(mod_dict, valid)
    y = np.array([1 if label_map[s] >= threshold else 0 for s in keep])
    if len(np.unique(y)) < 2:
        return {}
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(_clf(), X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    return dict(zip(keep, p))


def unimodal_oof_reg(mod_dict, label_map, subjects, seed=SEED):
    from sklearn.model_selection import KFold, cross_val_predict
    valid = [s for s in subjects if s in mod_dict and not np.isnan(label_map.get(s, np.nan))]
    X, keep = feature_matrix(mod_dict, valid)
    y = np.array([label_map[s] for s in keep], dtype=float)
    p = cross_val_predict(_reg(), X, y, cv=KFold(5, shuffle=True, random_state=seed), n_jobs=1)
    return dict(zip(keep, p))


def _meta_clf(C):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000))])


def select_C(X, y, seed=SEED):
    """Inner 3-fold AUROC selection of the meta-LR regularization C."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    if len(np.unique(y)) < 2:
        return 1.0
    best_C, best = 1.0, -1
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for C in C_GRID:
        try:
            s = cross_val_score(_meta_clf(C), X, y, cv=skf, scoring="roc_auc").mean()
        except Exception:  # noqa: BLE001
            continue
        if s > best:
            best, best_C = s, C
    return best_C


def select_alpha(X, y, seed=SEED):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_val_score
    best_a, best = 1.0, -1e9
    kf = KFold(3, shuffle=True, random_state=seed)
    for a in ALPHA_GRID:
        pipe = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=a))])
        try:
            s = cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
        except Exception:  # noqa: BLE001
            continue
        if s > best:
            best, best_a = s, a
    return best_a


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.splits import load_fold_assignments, load_fold_id_map
    from multimodal_aorta.data.ehr import load_ehr_features
    from multimodal_aorta.models.ecg_encoder import PCLREmbeddingEncoder
    from multimodal_aorta.models.cxr_encoder import CXREmbeddingEncoder
    from multimodal_aorta.training.bootstrap import (
        bootstrap_ci, paired_bootstrap_diff, auroc, auprc, mae, r2, fmt)
    from multimodal_aorta.training.oof_io import build_records, save_oof, write_manifest
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = Config()
    pc = os.path.join(root, "pretrained_checkpoints")
    out_dir = os.path.join(root, "outputs", "late_fusion_v2"); os.makedirs(out_dir, exist_ok=True)
    fold_path = os.path.join(pc, "fold_assignments.csv")

    lab = pd.read_csv(cfg.data.echo_labels_path)

    def clip(col, lo, hi):
        v = lab[col].copy(); v[(v < lo) | (v > hi)] = np.nan; return v

    root_diam = dict(zip(lab.subject_id.astype(int), clip("sinus_diam_cm", *ROOT_RANGE)))
    asc_diam = dict(zip(lab.subject_id.astype(int), clip("ascending_diam_cm", *ASC_RANGE)))
    pclr = {int(k): v for k, v in PCLREmbeddingEncoder.load_embeddings(os.path.join(pc, "pclr_embeddings.pt")).items()}
    raddino = {int(k): v for k, v in CXREmbeddingEncoder.load_embeddings(os.path.join(pc, "raddino_embeddings.pt")).items()}
    raddino_roi = {int(k): v for k, v in CXREmbeddingEncoder.load_embeddings(os.path.join(pc, "raddino_roi_embeddings.pt")).items()}
    ehr = {int(k): v for k, v in load_ehr_features(os.path.join(pc, "ehr_features.csv")).items()}
    MOD = {"ecg": pclr, "cxr": raddino, "cxr_roi": raddino_roi, "ehr": ehr}
    all_subjects = sorted(int(s) for s in lab.subject_id)

    cohort = pd.read_csv(os.path.join(pc, "cohort_triple.csv"))
    s522 = cohort["subject_id"].astype(int).tolist()
    folds = load_fold_assignments(fold_path, n_splits=5)
    fold_map = load_fold_id_map(fold_path)
    row_of = {s: i for i, s in enumerate(s522)}
    fold_ids = [fold_map.get(s, -1) for s in s522]

    results = {"n_patients": len(s522), "sites": {}}
    records = []

    for site, dmap in [("root", root_diam), ("asc", asc_diam)]:
        diam522 = np.array([dmap.get(s, np.nan) for s in s522])
        site_res = {}
        # EHR continuous diameter base (shared meta-feature for binary + the reg floor)
        d_ehr_full = unimodal_oof_reg(MOD["ehr"], dmap, all_subjects)
        d_ehr = np.array([d_ehr_full.get(s, np.nan) for s in s522])

        # ---------- binary endpoints ----------
        for ep, thr in [("ge40", 4.0), ("ge45", 4.5)]:
            y522 = np.where(np.isnan(diam522), np.nan, (diam522 >= thr).astype(float))
            base = {m: unimodal_oof_clf(MOD[m], dmap, thr, all_subjects) for m in MOD}
            p = {m: np.array([base[m].get(s, np.nan) for s in s522]) for m in MOD}
            # meta features: modality probs + EHR continuous diameter
            feats = np.column_stack([p["ecg"], p["cxr"], p["cxr_roi"], p["ehr"], d_ehr])
            meta = np.full(len(s522), np.nan)
            for tr_ids, te_ids in folds:
                tr = [row_of[s] for s in tr_ids if s in row_of]
                te = [row_of[s] for s in te_ids if s in row_of]
                trm = [i for i in tr if not np.isnan(y522[i]) and not np.isnan(feats[i]).any()]
                tem = [i for i in te if not np.isnan(feats[i]).any()]
                if len(np.unique(y522[trm])) < 2:
                    continue
                C = select_C(feats[trm], y522[trm].astype(int))
                m = _meta_clf(C).fit(feats[trm], y522[trm].astype(int))
                meta[tem] = m.predict_proba(feats[tem])[:, 1]
            # EHR-only reference (same full-cohort protocol)
            ehr_ref = p["ehr"]
            d_auroc = paired_bootstrap_diff(y522, meta, ehr_ref, auroc)
            site_res[ep] = {
                "pos": int(np.nansum(y522)),
                "v2_auroc": fmt(bootstrap_ci(y522, meta, auroc)),
                "v2_auprc": fmt(bootstrap_ci(y522, meta, auprc)),
                "ehr_floor_auroc": fmt(bootstrap_ci(y522, ehr_ref, auroc)),
                "delta_v2_minus_ehr_auroc": fmt(d_auroc),
                "delta_excludes_0": bool(d_auroc[1] > 0 or d_auroc[2] < 0),
            }
            log.info("[%s %s] v2 AUROC=%s | EHR floor=%s | delta=%s (excl0=%s)",
                     site, ep, site_res[ep]["v2_auroc"], site_res[ep]["ehr_floor_auroc"],
                     site_res[ep]["delta_v2_minus_ehr_auroc"], site_res[ep]["delta_excludes_0"])
            records += build_records(
                subject_ids=s522, fold_ids=fold_ids, model_name="late_fusion_v2",
                modality_set="ecg+cxr+ehr", site=site, endpoint=ep, target_type="binary",
                y_true=y522, pred_prob=meta)

        # ---------- regression (diameter) ----------
        dbase = {m: unimodal_oof_reg(MOD[m], dmap, all_subjects) for m in MOD}
        dfeat = np.column_stack([[dbase[m].get(s, np.nan) for s in s522] for m in MOD])  # ecg,cxr,cxr_roi,ehr
        meta_r = np.full(len(s522), np.nan)
        for tr_ids, te_ids in folds:
            tr = [row_of[s] for s in tr_ids if s in row_of]
            te = [row_of[s] for s in te_ids if s in row_of]
            trm = [i for i in tr if not np.isnan(diam522[i]) and not np.isnan(dfeat[i]).any()]
            tem = [i for i in te if not np.isnan(dfeat[i]).any()]
            if len(trm) < 10:
                continue
            a = select_alpha(dfeat[trm], diam522[trm])
            rg = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=a))]).fit(dfeat[trm], diam522[trm])
            meta_r[tem] = rg.predict(dfeat[tem])
        d_mae = paired_bootstrap_diff(diam522, meta_r, d_ehr, mae, need_both_classes=False)
        d_r2 = paired_bootstrap_diff(diam522, meta_r, d_ehr, r2, need_both_classes=False)
        site_res["diam"] = {
            "v2_mae": fmt(bootstrap_ci(diam522, meta_r, mae, need_both_classes=False)),
            "v2_r2": fmt(bootstrap_ci(diam522, meta_r, r2, need_both_classes=False)),
            "ehr_floor_mae": fmt(bootstrap_ci(diam522, d_ehr, mae, need_both_classes=False)),
            "ehr_floor_r2": fmt(bootstrap_ci(diam522, d_ehr, r2, need_both_classes=False)),
            "delta_v2_minus_ehr_mae": fmt(d_mae),
            "delta_v2_minus_ehr_r2": fmt(d_r2),
            "r2_delta_excludes_0": bool(d_r2[1] > 0 or d_r2[2] < 0),
        }
        log.info("[%s diam] v2 R2=%s | EHR floor R2=%s | deltaR2=%s (excl0=%s)",
                 site, site_res["diam"]["v2_r2"], site_res["diam"]["ehr_floor_r2"],
                 site_res["diam"]["delta_v2_minus_ehr_r2"], site_res["diam"]["r2_delta_excludes_0"])
        records += build_records(
            subject_ids=s522, fold_ids=fold_ids, model_name="late_fusion_v2",
            modality_set="ecg+cxr+ehr", site=site, endpoint="diam", target_type="regression",
            y_true=diam522, pred_value=meta_r)
        results["sites"][site] = site_res

    save_oof(records, os.path.join(out_dir, "oof_predictions.csv"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_manifest(os.path.join(out_dir, "run_manifest.json"), model_name="late_fusion_v2",
                   seed=SEED, n_patients=len(s522),
                   cohort_csv=os.path.join(pc, "cohort_triple.csv"), fold_csv=fold_path)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
