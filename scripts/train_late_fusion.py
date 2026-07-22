"""
v3 Milestone 6 — late fusion / stacking.

Motivation: the 522-only fusion wastes the much larger per-modality cohorts.
The ECG cohort has ~195/291 root/asc AD+ (vs 32/48 in the 522) and EHR ~288/432;
≥4.5 positives jump from ~8/10 to ~54-96. Stacking exploits that:

  Level 0 (per modality, trained on its FULL cohort via K-fold cross_val_predict):
     ECG  : PCLR(320)     over ~2874 patients
     CXR  : RAD-DINO(768) over ~550   patients
     EHR  : tabular(12)   over ~4525  patients
   -> leakage-free out-of-fold probability for every patient (incl. the 522).

  Level 1 (meta-learner on the 522, SAME 5 outer folds as GBDT/deep):
     LogisticRegression over [p_ecg, p_cxr, p_ehr] -> final OOF prediction.

The meta-test patient's level-0 features are OOF (never trained on its own label),
so stacking is leakage-free. Reports per-modality (full-cohort) standalone on the
522 too, plus bootstrap CIs. Entirely new file — does not touch existing scripts.

Run via SLURM: sbatch scripts/slurm_late_fusion.sh
Outputs: console log + outputs/late_fusion/{results.json, oof.npz}
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


def unimodal_oof_clf(mod_dict, label_map, threshold, subjects, seed=42):
    """5-fold cross_val_predict prob over a modality's full cohort. -> {sid: prob}."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    valid = [s for s in subjects if s in mod_dict and not np.isnan(label_map.get(s, np.nan))]
    X, keep = feature_matrix(mod_dict, valid)
    y = np.array([1 if label_map[s] >= threshold else 0 for s in keep])
    if len(np.unique(y)) < 2:
        return {}
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(_clf(), X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    return dict(zip(keep, p))


def unimodal_oof_reg(mod_dict, label_map, subjects, seed=42):
    from sklearn.model_selection import KFold, cross_val_predict
    valid = [s for s in subjects if s in mod_dict and not np.isnan(label_map.get(s, np.nan))]
    X, keep = feature_matrix(mod_dict, valid)
    y = np.array([label_map[s] for s in keep], dtype=float)
    kf = KFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(_reg(), X, y, cv=kf, n_jobs=1)
    return dict(zip(keep, p))


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.splits import load_fold_assignments, load_fold_id_map
    from multimodal_aorta.data.ehr import load_ehr_features, FEATURE_COLS
    from multimodal_aorta.models.ecg_encoder import PCLREmbeddingEncoder
    from multimodal_aorta.models.cxr_encoder import CXREmbeddingEncoder
    from multimodal_aorta.training.bootstrap import bootstrap_ci, auroc, auprc, mae, r2, fmt
    from multimodal_aorta.training.oof_io import build_records, save_oof, write_manifest
    from sklearn.linear_model import LogisticRegression, Ridge

    cfg = Config()
    pc = os.path.join(root, "pretrained_checkpoints")
    out_dir = os.path.join(root, "outputs", "late_fusion"); os.makedirs(out_dir, exist_ok=True)
    fold_path = os.path.join(pc, "fold_assignments.csv")

    # --- labels for ALL subjects (full per-modality cohorts), QC-clipped ---
    lab = pd.read_csv(cfg.data.echo_labels_path)
    def clip(col, lo, hi):
        v = lab[col].copy()
        v[(v < lo) | (v > hi)] = np.nan
        return v
    root_diam = dict(zip(lab.subject_id, clip("sinus_diam_cm", *ROOT_RANGE)))
    asc_diam = dict(zip(lab.subject_id, clip("ascending_diam_cm", *ASC_RANGE)))

    pclr = PCLREmbeddingEncoder.load_embeddings(os.path.join(pc, "pclr_embeddings.pt"))
    raddino = CXREmbeddingEncoder.load_embeddings(os.path.join(pc, "raddino_embeddings.pt"))
    ehr = load_ehr_features(os.path.join(pc, "ehr_features.csv"))
    pclr = {int(k): v for k, v in pclr.items()}
    raddino = {int(k): v for k, v in raddino.items()}
    MOD = {"ecg": pclr, "cxr": raddino, "ehr": ehr}
    all_subjects = set(lab.subject_id.astype(int))

    # --- 522 triple cohort + same outer folds as GBDT/deep ---
    cohort = pd.read_csv(os.path.join(pc, "cohort_triple.csv"))
    s522 = cohort["subject_id"].astype(int).tolist()
    folds = load_fold_assignments(fold_path, n_splits=5)
    row_of = {s: i for i, s in enumerate(s522)}

    results, oof_store = {"n_patients": len(s522), "sites": {}}, {}

    for site, dmap, rng_ in [("root", root_diam, ROOT_RANGE), ("asc", asc_diam, ASC_RANGE)]:
        diam522 = np.array([dmap.get(s, np.nan) for s in s522])
        site_res = {}
        # ---------- classification targets ----------
        for tname, thr in [("ge40", 4.0), ("ge45", 4.5)]:
            y522 = np.where(np.isnan(diam522), np.nan, (diam522 >= thr).astype(float))
            # level-0 OOF per modality (trained on full cohorts)
            uni = {m: unimodal_oof_clf(MOD[m], dmap, thr, sorted(all_subjects)) for m in MOD}
            # standalone (full-cohort unimodal) on the 522
            stand = {}
            for m in MOD:
                p = np.array([uni[m].get(s, np.nan) for s in s522])
                stand[m] = bootstrap_ci(y522, p, auroc)
            # meta features (all 522 have all 3 modalities)
            MX = np.column_stack([[uni[m].get(s, np.nan) for s in s522] for m in MOD])
            meta_oof = np.full(len(s522), np.nan)
            for tr_ids, te_ids in folds:
                tr = [row_of[s] for s in tr_ids if s in row_of]
                te = [row_of[s] for s in te_ids if s in row_of]
                trm = [i for i in tr if not np.isnan(y522[i]) and not np.isnan(MX[i]).any()]
                tem = [i for i in te if not np.isnan(MX[i]).any()]
                if len(np.unique(y522[trm])) < 2:
                    continue
                lr = LogisticRegression(class_weight="balanced", max_iter=1000)
                lr.fit(MX[trm], y522[trm].astype(int))
                meta_oof[tem] = lr.predict_proba(MX[tem])[:, 1]
            site_res[tname] = {
                "meta_auroc": fmt(bootstrap_ci(y522, meta_oof, auroc)),
                "meta_auprc": fmt(bootstrap_ci(y522, meta_oof, auprc)),
                "standalone_auroc": {m: fmt(stand[m]) for m in MOD},
                "pos": int(np.nansum(y522)),
            }
            oof_store[f"{site}_{tname}_y"] = y522
            oof_store[f"{site}_{tname}_p"] = meta_oof
            log.info("[%s %s] meta AUROC=%s AUPRC=%s (pos=%d)", site, tname,
                     site_res[tname]["meta_auroc"], site_res[tname]["meta_auprc"], site_res[tname]["pos"])
            log.info("    standalone(full-cohort) AUROC: %s",
                     {m: fmt(stand[m]) for m in MOD})
        # ---------- regression (diameter) ----------
        uni_r = {m: unimodal_oof_reg(MOD[m], dmap, sorted(all_subjects)) for m in MOD}
        MXr = np.column_stack([[uni_r[m].get(s, np.nan) for s in s522] for m in MOD])
        meta_r = np.full(len(s522), np.nan)
        for tr_ids, te_ids in folds:
            tr = [row_of[s] for s in tr_ids if s in row_of]
            te = [row_of[s] for s in te_ids if s in row_of]
            trm = [i for i in tr if not np.isnan(diam522[i]) and not np.isnan(MXr[i]).any()]
            tem = [i for i in te if not np.isnan(MXr[i]).any()]
            if len(trm) < 10:
                continue
            rg = Ridge(alpha=1.0).fit(MXr[trm], diam522[trm])
            meta_r[tem] = rg.predict(MXr[tem])
        site_res["diam"] = {"mae": fmt(bootstrap_ci(diam522, meta_r, mae, need_both_classes=False)),
                            "r2": fmt(bootstrap_ci(diam522, meta_r, r2, need_both_classes=False))}
        oof_store[f"{site}_diam_y"] = diam522
        oof_store[f"{site}_diam_p"] = meta_r
        log.info("[%s diam] meta MAE=%s R2=%s", site, site_res["diam"]["mae"], site_res["diam"]["r2"])
        results["sites"][site] = site_res

    np.savez(os.path.join(out_dir, "oof.npz"), **oof_store)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- standardized per-patient OOF (M0) for cross-model paired comparison ---
    fold_map = load_fold_id_map(fold_path)
    fold_ids = [fold_map.get(int(s), -1) for s in s522]
    has_ehr = [int(int(s) in ehr) for s in s522]
    records = []
    for site in ("root", "asc"):
        for ep in ("ge40", "ge45"):
            records += build_records(
                subject_ids=s522, fold_ids=fold_ids, model_name="late_fusion_meta",
                modality_set="ecg+cxr+ehr", site=site, endpoint=ep, target_type="binary",
                y_true=oof_store[f"{site}_{ep}_y"], pred_prob=oof_store[f"{site}_{ep}_p"],
                has_ehr=has_ehr)
        records += build_records(
            subject_ids=s522, fold_ids=fold_ids, model_name="late_fusion_meta",
            modality_set="ecg+cxr+ehr", site=site, endpoint="diam", target_type="regression",
            y_true=oof_store[f"{site}_diam_y"], pred_value=oof_store[f"{site}_diam_p"],
            has_ehr=has_ehr)
    save_oof(records, os.path.join(out_dir, "oof_predictions.csv"))
    write_manifest(
        os.path.join(out_dir, "run_manifest.json"), model_name="late_fusion_meta",
        seed=cfg.data.split_seed, n_patients=len(s522),
        cohort_csv=os.path.join(pc, "cohort_triple.csv"), fold_csv=fold_path)
    log.info("Saved -> %s (+ oof_predictions.csv, run_manifest.json)", out_dir)


if __name__ == "__main__":
    main()
