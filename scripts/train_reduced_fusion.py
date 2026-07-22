"""
M5 — PCA-reduced early fusion (the "denoise the embeddings" win).

Motivation (validated 2026-07-16): concatenating the raw frozen features
[PCLR(320) | RAD-DINO(768) | EHR(12)] = 1100 dims on ~420 train rows badly
overfits the tree/MLP. Reducing each embedding with a FOLD-SAFE PCA before the
head lifts every endpoint, and for the ASCENDING aorta the paired delta vs the
equal-n EHR floor EXCLUDES 0 (a genuine, significant multimodal signal — the
ascending aorta is visible on the frontal CXR silhouette).

Design:
  - PCA fit on the TRAIN fold only (no leakage), applied to ECG and CXR blocks.
    EHR kept raw (HGB is NaN-native; only 12 dims, no reduction needed).
  - HistGradientBoosting head (matches the other baselines for comparability).
  - Emits the standardized OOF schema so make_eval_report.py aligns it patient-
    for-patient against the EHR floor and the other fusion models.

Config via env vars (all optional):
  CXR_EMB   embedding file under pretrained_checkpoints (default raddino_embeddings.pt;
            use raddino_anatomy_roi_union_embeddings.pt for the anatomy-ROI variant)
  K_ECG     PCA comps for PCLR   (default 16)
  K_CXR     PCA comps for RAD-DINO (default 32)
  USE_ECG   1/0 include ECG block (default 1; ECG adds little — 0 is often as good)
  MODEL_NAME  name written into oof_predictions.csv (default reduced_fusion)

Run: python scripts/train_reduced_fusion.py         (fast; submit via slurm on a real node)
Outputs: outputs/reduced_fusion/{results.json, oof_predictions.csv, run_manifest.json}
"""
import os, sys, json, logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

K_ECG = int(os.environ.get("K_ECG", 16))
K_CXR = int(os.environ.get("K_CXR", 32))
USE_ECG = os.environ.get("USE_ECG", "1") == "1"
CXR_EMB = os.environ.get("CXR_EMB", "raddino_embeddings.pt")
MODEL_NAME = os.environ.get("MODEL_NAME", "reduced_fusion")


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


def oof(ECG, CXR, EHR, folds, idx_of, y, kind="clf"):
    """Fold-safe PCA(ECG->K_ECG, CXR->K_CXR) + raw EHR -> HGB, out-of-fold."""
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    out = np.full(len(y), np.nan)
    base = _clf() if kind == "clf" else _reg()
    for tr_ids, te_ids in folds:
        tr = np.array([idx_of[s] for s in tr_ids if s in idx_of and not np.isnan(y[idx_of[s]])])
        te = np.array([idx_of[s] for s in te_ids if s in idx_of and not np.isnan(y[idx_of[s]])])
        if kind == "clf" and len(np.unique(y[tr])) < 2:
            continue
        if kind == "reg" and len(tr) < 10:
            continue
        pca_c = PCA(n_components=min(K_CXR, len(tr) - 1), random_state=0).fit(CXR[tr])
        pca_e = (PCA(n_components=min(K_ECG, len(tr) - 1), random_state=0).fit(ECG[tr])
                 if USE_ECG else None)

        def feats(rows):
            parts = [pca_c.transform(CXR[rows])]
            if USE_ECG:
                parts.insert(0, pca_e.transform(ECG[rows]))
            parts.append(EHR[rows])
            return np.concatenate(parts, axis=1)

        m = clone(base).fit(feats(tr), y[tr] if kind == "reg" else y[tr].astype(int))
        out[te] = m.predict(feats(te)) if kind == "reg" else m.predict_proba(feats(te))[:, 1]
    return out


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    import torch
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.splits import load_fold_assignments, load_fold_id_map
    from multimodal_aorta.data.ehr import load_ehr_features, FEATURE_COLS
    from multimodal_aorta.training.bootstrap import bootstrap_ci, paired_bootstrap_diff, auroc, auprc, mae, r2, fmt
    from multimodal_aorta.training.oof_io import build_records, save_oof, write_manifest

    cfg = Config()
    pc = os.path.join(root, "pretrained_checkpoints")
    out_dir = os.path.join(root, "outputs", MODEL_NAME); os.makedirs(out_dir, exist_ok=True)
    fold_path = os.path.join(pc, "fold_assignments.csv")

    def load_emb(fn):
        d = torch.load(os.path.join(pc, fn), map_location="cpu", weights_only=False)
        return {int(k): (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}

    cohort = pd.read_csv(os.path.join(pc, "cohort_triple.csv"))
    pclr = load_emb("pclr_embeddings.pt")
    raddino = load_emb(CXR_EMB)
    ehr = {int(k): np.asarray(v, np.float32) for k, v in load_ehr_features(os.path.join(pc, "ehr_features.csv")).items()}
    ehr_dim = len(FEATURE_COLS)
    log.info("config: K_ECG=%d K_CXR=%d USE_ECG=%s CXR_EMB=%s -> model=%s",
             K_ECG, K_CXR, USE_ECG, CXR_EMB, MODEL_NAME)

    rows, ECG, CXR, EHR = [], [], [], []
    for r in cohort.itertuples(index=False):
        sid = int(r.subject_id)
        if sid not in pclr or sid not in raddino:
            continue
        ECG.append(pclr[sid].astype(np.float32))
        CXR.append(raddino[sid].astype(np.float32))
        EHR.append(ehr.get(sid, np.full(ehr_dim, np.nan, np.float32)))
        rows.append(r)
    df = pd.DataFrame(rows).reset_index(drop=True)
    ECG = np.vstack(ECG); CXR = np.vstack(CXR); EHR = np.vstack(EHR).astype(np.float32)
    idx_of = {int(s): i for i, s in enumerate(df["subject_id"])}
    folds = load_fold_assignments(fold_path, n_splits=5)
    fold_map = load_fold_id_map(fold_path)
    log.info("Assembled n=%d", len(df))

    targets = {}
    for site, col in [("root", "target_root"), ("asc", "target_asc")]:
        d = df[col].to_numpy(float)
        targets[site] = {
            "diam": d,
            "ge40": np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float)),
            "ge45": np.where(np.isnan(d), np.nan, (d >= 4.5).astype(float)),
        }

    subj = [int(s) for s in df["subject_id"]]
    fold_ids = [fold_map.get(s, -1) for s in subj]
    has_ehr = [int(s in ehr) for s in subj]

    results = {"n_patients": int(len(df)), "config":
               {"K_ECG": K_ECG, "K_CXR": K_CXR, "USE_ECG": USE_ECG, "CXR_EMB": CXR_EMB},
               "sites": {}}
    records = []
    for site in ("root", "asc"):
        t = targets[site]
        p40 = oof(ECG, CXR, EHR, folds, idx_of, t["ge40"], "clf")
        p45 = oof(ECG, CXR, EHR, folds, idx_of, t["ge45"], "clf")
        pdi = oof(ECG, CXR, EHR, folds, idx_of, t["diam"], "reg")
        results["sites"][site] = {
            "anyAD_ge4.0": {"auroc": fmt(bootstrap_ci(t["ge40"], p40, auroc)),
                            "auprc": fmt(bootstrap_ci(t["ge40"], p40, auprc)),
                            "pos": int(np.nansum(t["ge40"]))},
            "moderate_ge4.5": {"auroc": fmt(bootstrap_ci(t["ge45"], p45, auroc)),
                               "pos": int(np.nansum(t["ge45"]))},
            "diam_regression": {"mae": fmt(bootstrap_ci(t["diam"], pdi, mae, need_both_classes=False)),
                                "r2": fmt(bootstrap_ci(t["diam"], pdi, r2, need_both_classes=False))},
        }
        log.info("=== %s === ge40 AUROC=%s | ge45 AUROC=%s | diam R2=%s", site.upper(),
                 results["sites"][site]["anyAD_ge4.0"]["auroc"],
                 results["sites"][site]["moderate_ge4.5"]["auroc"],
                 results["sites"][site]["diam_regression"]["r2"])
        for ep, p in [("ge40", p40), ("ge45", p45)]:
            records += build_records(subject_ids=subj, fold_ids=fold_ids, model_name=MODEL_NAME,
                                     modality_set=("cxr+ehr" if not USE_ECG else "ecg+cxr+ehr"),
                                     site=site, endpoint=ep, target_type="binary",
                                     y_true=t[ep], pred_prob=p, has_ehr=has_ehr)
        records += build_records(subject_ids=subj, fold_ids=fold_ids, model_name=MODEL_NAME,
                                 modality_set=("cxr+ehr" if not USE_ECG else "ecg+cxr+ehr"),
                                 site=site, endpoint="diam", target_type="regression",
                                 y_true=t["diam"], pred_value=pdi, has_ehr=has_ehr)

    save_oof(records, os.path.join(out_dir, "oof_predictions.csv"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_manifest(os.path.join(out_dir, "run_manifest.json"), model_name=MODEL_NAME,
                   seed=cfg.data.split_seed, n_patients=len(df),
                   cohort_csv=os.path.join(pc, "cohort_triple.csv"), fold_csv=fold_path)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
