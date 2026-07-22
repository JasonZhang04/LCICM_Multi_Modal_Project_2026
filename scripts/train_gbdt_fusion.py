"""
v3 Milestone 3 — Gradient-boosted-tree fusion baseline on the triple cohort.

Early/feature fusion: concatenate frozen features
    [ PCLR(320) | RAD-DINO(768) | EHR(12) ]
and train sklearn HistGradientBoosting models (LightGBM-style, NaN-native so EHR
missingness needs no imputation) under stratified 5-fold CV with out-of-fold (OOF)
predictions.

Per site (root, ascending) we report:
    - binary >=4.0 cm  (any AD; PRIMARY)      AUROC + AUPRC
    - binary >=4.5 cm  (moderate+; 3-class)   AUROC + AUPRC
    - ordinal 4-grade  (via 3 cumulative clfs) quadratic-weighted kappa
    - diameter (cm)     regression             MAE + R²

Plus a prevalence baseline and a per-modality ablation (ECG / CXR / EHR / ECG+CXR /
all), and an "all + PCA-reduced embeddings" variant to probe the high-dim vs small-n
trade-off.

Run: python scripts/train_gbdt_fusion.py
Outputs: console log + outputs/gbdt_fusion/results.json
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

CUTS = [4.0, 4.5, 5.0]


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def assemble_features(cohort, pclr, raddino, ehr, ehr_cols):
    """
    Build the concatenated feature matrix for subjects that have BOTH PCLR and
    RAD-DINO embeddings (EHR may be missing -> NaN). Returns (df, X, blocks).
    blocks maps modality -> column index slice for ablation.
    """
    import torch
    rows, X = [], []
    n_drop = 0
    ehr_dim = len(ehr_cols)
    for r in cohort.itertuples(index=False):
        sid = int(r.subject_id)
        if sid not in pclr or sid not in raddino:
            n_drop += 1
            continue
        e = pclr[sid].numpy() if isinstance(pclr[sid], torch.Tensor) else np.asarray(pclr[sid])
        c = raddino[sid].numpy() if isinstance(raddino[sid], torch.Tensor) else np.asarray(raddino[sid])
        h = ehr.get(sid, np.full(ehr_dim, np.nan, dtype=np.float32))
        X.append(np.concatenate([e.astype(np.float32), c.astype(np.float32), h.astype(np.float32)]))
        rows.append(r)
    X = np.vstack(X)
    df = pd.DataFrame(rows)
    blocks = {
        "ecg": slice(0, 320),
        "cxr": slice(320, 320 + 768),
        "ehr": slice(320 + 768, 320 + 768 + ehr_dim),
    }
    log.info("Feature matrix: %s (dropped %d subjects missing PCLR/RAD-DINO)", X.shape, n_drop)
    return df, X, blocks


# ---------------------------------------------------------------------------
# Models / metrics
# ---------------------------------------------------------------------------

def _clf():
    # early_stopping=False: with rare positives (severe ~6 total) an internal
    # validation split could contain zero positives. Rely on regularization
    # (shallow trees, min_samples_leaf, l2) + modest max_iter instead.
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False,
        class_weight="balanced", random_state=0,
    )

def _reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=0,
    )


def oof_binary(X, y, folds, idx_of):
    """Out-of-fold predicted probabilities for a binary target (NaN y rows skipped)."""
    from sklearn.base import clone
    oof = np.full(len(y), np.nan)
    base = _clf()
    for train_ids, test_ids in folds:
        tr = [idx_of[s] for s in train_ids if s in idx_of]
        te = [idx_of[s] for s in test_ids if s in idx_of]
        tr = [i for i in tr if not np.isnan(y[i])]
        te = [i for i in te if not np.isnan(y[i])]
        if len(np.unique(y[tr])) < 2:
            continue
        m = clone(base).fit(X[tr], y[tr].astype(int))
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def oof_reg(X, y, folds, idx_of):
    from sklearn.base import clone
    oof = np.full(len(y), np.nan)
    base = _reg()
    for train_ids, test_ids in folds:
        tr = [idx_of[s] for s in train_ids if s in idx_of]
        te = [idx_of[s] for s in test_ids if s in idx_of]
        tr = [i for i in tr if not np.isnan(y[i])]
        te = [i for i in te if not np.isnan(y[i])]
        if len(tr) < 10:
            continue
        m = clone(base).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def binary_metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score
    m = ~np.isnan(p) & ~np.isnan(y)
    yy, pp = y[m].astype(int), p[m]
    if len(np.unique(yy)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": int(m.sum()), "pos": int(yy.sum())}
    return {"auroc": float(roc_auc_score(yy, pp)),
            "auprc": float(average_precision_score(yy, pp)),
            "n": int(m.sum()), "pos": int(yy.sum())}


def reg_metrics(y, p):
    from sklearn.metrics import mean_absolute_error, r2_score
    m = ~np.isnan(p) & ~np.isnan(y)
    if m.sum() < 2:
        return {"mae": float("nan"), "r2": float("nan"), "n": int(m.sum())}
    return {"mae": float(mean_absolute_error(y[m], p[m])),
            "r2": float(r2_score(y[m], p[m])), "n": int(m.sum())}


def ordinal_qwk(grade_true, cum_probs):
    """grade_hat = #cumulative-probs>0.5; QWK vs true grade."""
    from sklearn.metrics import cohen_kappa_score
    grade_hat = np.sum(np.stack([(p > 0.5).astype(int) for p in cum_probs], axis=1), axis=1)
    m = ~np.isnan(grade_true)
    for p in cum_probs:
        m &= ~np.isnan(p)
    if m.sum() < 2 or len(np.unique(grade_true[m])) < 2:
        return float("nan")
    return float(cohen_kappa_score(grade_true[m].astype(int), grade_hat[m], weights="quadratic"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.splits import load_fold_assignments, load_fold_id_map
    from multimodal_aorta.data.targets import diam_to_grade
    from multimodal_aorta.models.ecg_encoder import PCLREmbeddingEncoder
    from multimodal_aorta.models.cxr_encoder import CXREmbeddingEncoder
    from multimodal_aorta.data.ehr import load_ehr_features, FEATURE_COLS
    from multimodal_aorta.training.oof_io import build_records, save_oof, write_manifest

    cfg = Config()
    out_dir = os.path.join(root, "outputs", "gbdt_fusion")
    os.makedirs(out_dir, exist_ok=True)
    fold_path = os.path.join(root, "pretrained_checkpoints", "fold_assignments.csv")

    cohort = pd.read_csv(os.path.join(root, "pretrained_checkpoints", "cohort_triple.csv"))
    log.info("Triple cohort: %d patients", len(cohort))

    pclr = PCLREmbeddingEncoder.load_embeddings(
        os.path.join(root, "pretrained_checkpoints", "pclr_embeddings.pt"))
    raddino = CXREmbeddingEncoder.load_embeddings(
        os.path.join(root, "pretrained_checkpoints", "raddino_embeddings.pt"))
    ehr = load_ehr_features(os.path.join(root, "pretrained_checkpoints", "ehr_features.csv"))

    df, X, blocks = assemble_features(cohort, pclr, raddino, ehr, FEATURE_COLS)
    df = df.reset_index(drop=True)
    idx_of = {int(s): i for i, s in enumerate(df["subject_id"].tolist())}

    folds = load_fold_assignments(fold_path, n_splits=5)

    # Targets
    targets = {}
    for site, col in [("root", "target_root"), ("asc", "target_asc")]:
        diam = df[col].to_numpy(dtype=float)
        targets[site] = {
            "diam": diam,
            "grade": np.array([diam_to_grade(v) for v in diam]),
            "ge40": np.where(np.isnan(diam), np.nan, (diam >= 4.0).astype(float)),
            "ge45": np.where(np.isnan(diam), np.nan, (diam >= 4.5).astype(float)),
            "ge50": np.where(np.isnan(diam), np.nan, (diam >= 5.0).astype(float)),
        }

    results = {"n_patients": int(len(df)), "feature_dim": int(X.shape[1]), "sites": {}}
    oof_store = {}   # saved for the bootstrap-CI comparison (eval_bootstrap.py)

    def feat(cols):
        if cols == "all":
            return X
        if isinstance(cols, str):
            return X[:, blocks[cols]]
        return np.concatenate([X[:, blocks[c]] for c in cols], axis=1)

    # --- Full-feature results per site (primary) ---
    for site in ("root", "asc"):
        t = targets[site]
        ge40 = oof_binary(X, t["ge40"], folds, idx_of)
        ge45 = oof_binary(X, t["ge45"], folds, idx_of)
        ge50 = oof_binary(X, t["ge50"], folds, idx_of)
        dpred = oof_reg(X, t["diam"], folds, idx_of)
        oof_store[f"{site}_ge40_y"] = t["ge40"]; oof_store[f"{site}_ge40_p"] = ge40
        oof_store[f"{site}_ge45_y"] = t["ge45"]; oof_store[f"{site}_ge45_p"] = ge45
        oof_store[f"{site}_diam_y"] = t["diam"]; oof_store[f"{site}_diam_p"] = dpred
        res = {
            "anyAD_ge4.0": binary_metrics(t["ge40"], ge40),
            "moderate_ge4.5": binary_metrics(t["ge45"], ge45),
            "severe_ge5.0": binary_metrics(t["ge50"], ge50),
            "ordinal_qwk": ordinal_qwk(t["grade"], [ge40, ge45, ge50]),
            "diam_regression": reg_metrics(t["diam"], dpred),
            "prevalence_ge4.0": float(np.nanmean(t["ge40"])),
        }
        results["sites"][site] = res
        log.info("=== SITE %s (full features) ===", site.upper())
        log.info("  anyAD>=4.0   AUROC=%.3f AUPRC=%.3f (n=%d pos=%d, prev=%.3f)",
                 res["anyAD_ge4.0"]["auroc"], res["anyAD_ge4.0"]["auprc"],
                 res["anyAD_ge4.0"]["n"], res["anyAD_ge4.0"]["pos"], res["prevalence_ge4.0"])
        log.info("  mod>=4.5     AUROC=%.3f AUPRC=%.3f (pos=%d)",
                 res["moderate_ge4.5"]["auroc"], res["moderate_ge4.5"]["auprc"], res["moderate_ge4.5"]["pos"])
        log.info("  ordinal QWK=%.3f | diam MAE=%.3f R2=%.3f",
                 res["ordinal_qwk"], res["diam_regression"]["mae"], res["diam_regression"]["r2"])

    # --- Per-modality ablation on the PRIMARY endpoint (anyAD>=4.0) ---
    log.info("=== ABLATION (anyAD>=4.0 AUROC / AUPRC) ===")
    results["ablation"] = {}
    for name, cols in [("ecg", "ecg"), ("cxr", "cxr"), ("ehr", "ehr"),
                       ("ecg+cxr", ["ecg", "cxr"]), ("all", "all")]:
        Xs = feat(cols)
        results["ablation"][name] = {}
        line = []
        for site in ("root", "asc"):
            p = oof_binary(Xs, targets[site]["ge40"], folds, idx_of)
            mm = binary_metrics(targets[site]["ge40"], p)
            results["ablation"][name][site] = mm
            line.append(f"{site}:AUROC={mm['auroc']:.3f}/AUPRC={mm['auprc']:.3f}")
        log.info("  %-8s  %s", name, "  ".join(line))

    # --- PCA-reduced embeddings variant (probe high-dim vs small-n) ---
    try:
        from sklearn.decomposition import PCA
        log.info("=== PCA variant (PCLR->32, RAD-DINO->64, + EHR raw) ===")
        results["pca_variant"] = {}
        for site in ("root", "asc"):
            # PCA must be fit inside CV; approximate here by fold-wise reduction
            p = oof_binary_pca(X, blocks, targets[site]["ge40"], folds, idx_of, PCA)
            mm = binary_metrics(targets[site]["ge40"], p)
            results["pca_variant"][site] = mm
            log.info("  %s anyAD>=4.0  AUROC=%.3f AUPRC=%.3f", site, mm["auroc"], mm["auprc"])
    except Exception as e:  # noqa: BLE001
        log.warning("PCA variant skipped: %s", e)

    np.savez(os.path.join(out_dir, "oof.npz"), **oof_store)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- standardized per-patient OOF (M0) for cross-model paired comparison ---
    subj = [int(s) for s in df["subject_id"].tolist()]
    fold_map = load_fold_id_map(fold_path)
    fold_ids = [fold_map.get(s, -1) for s in subj]
    has_ehr = [int(s in ehr) for s in subj]
    records = []
    for site in ("root", "asc"):
        for ep in ("ge40", "ge45"):
            records += build_records(
                subject_ids=subj, fold_ids=fold_ids, model_name="gbdt_concat",
                modality_set="ecg+cxr+ehr", site=site, endpoint=ep, target_type="binary",
                y_true=oof_store[f"{site}_{ep}_y"], pred_prob=oof_store[f"{site}_{ep}_p"],
                has_ehr=has_ehr)
        records += build_records(
            subject_ids=subj, fold_ids=fold_ids, model_name="gbdt_concat",
            modality_set="ecg+cxr+ehr", site=site, endpoint="diam", target_type="regression",
            y_true=oof_store[f"{site}_diam_y"], pred_value=oof_store[f"{site}_diam_p"],
            has_ehr=has_ehr)
    save_oof(records, os.path.join(out_dir, "oof_predictions.csv"))
    write_manifest(
        os.path.join(out_dir, "run_manifest.json"), model_name="gbdt_concat",
        seed=cfg.data.split_seed, n_patients=len(df),
        cohort_csv=os.path.join(root, "pretrained_checkpoints", "cohort_triple.csv"),
        fold_csv=fold_path)
    log.info("Saved -> %s (+ oof.npz, oof_predictions.csv, run_manifest.json)",
             os.path.join(out_dir, "results.json"))


def oof_binary_pca(X, blocks, y, folds, idx_of, PCA):
    """OOF binary with per-fold PCA on the embedding blocks (EHR kept raw)."""
    from sklearn.base import clone
    oof = np.full(len(y), np.nan)
    base = _clf()
    ecg, cxr, ehr = blocks["ecg"], blocks["cxr"], blocks["ehr"]
    for train_ids, test_ids in folds:
        tr = [idx_of[s] for s in train_ids if s in idx_of and not np.isnan(y[idx_of[s]])]
        te = [idx_of[s] for s in test_ids if s in idx_of and not np.isnan(y[idx_of[s]])]
        if len(np.unique(y[tr])) < 2:
            continue
        pe = PCA(n_components=32, random_state=0).fit(X[tr, ecg])
        pc = PCA(n_components=64, random_state=0).fit(X[tr, cxr])
        def red(rows):
            return np.concatenate([pe.transform(X[rows, ecg]), pc.transform(X[rows, cxr]),
                                   X[rows, ehr]], axis=1)
        m = clone(base).fit(red(tr), y[tr].astype(int))
        oof[te] = m.predict_proba(red(te))[:, 1]
    return oof


if __name__ == "__main__":
    main()
