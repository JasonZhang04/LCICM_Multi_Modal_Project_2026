"""
Combined late-fusion stack — the best multimodal model as of 2026-07-16.

Combines every win from notes/experiments_2026-07-16.md:
  - CXR base learner  = MULTI-INSTANCE (all in-window CXRs, not 1/patient)
                        + ANATOMY-ROI (aorta+mediastinum+heart segmentation crop)
                        + PCA(32)  reduction (fold-safe)
                        restricted to FRONTAL (PA/AP) views (laterals fail segmentation
                        and don't show the aorta).
  - EHR base learner  = full-cohort HistGradientBoosting (repo `ehr_baseline` OOF).
  - Meta-learner      = nested-CV LogisticRegression (binary) / Ridge (regression)
                        over [p_cxr, p_ehr, d_ehr].

Everything is 5-fold OOF on the 522 triple cohort using the immutable
`fold_assignments.csv`, with paired bootstrap CIs vs the full-cohort EHR floor,
so numbers are directly comparable to outputs/eval_report.

Prereqs (one-time, in this order):
  1. scripts/extract_multi_cxr_instances.py         -> cxr_instances.csv (+ whole-image emb)
  2. scripts/extract_multi_anatomy_roi_embeddings.py-> raddino_multi_anatomy_embeddings.pt
  3. scripts/train_ehr_baseline.py                  -> outputs/ehr_baseline/oof_predictions.csv

Env knobs:
  VIEW=frontal|all           (default frontal — the best config)
  CXR_EMB=<file in pretrained_checkpoints/>  (default raddino_multi_anatomy_embeddings.pt)
  OUT_TAG=<outputs subdir>   (default combined_stack_frontal_anat)
  K_CXR=32

Run via SLURM: sbatch scripts/slurm_combined_stack.sh
Outputs: outputs/<OUT_TAG>/results.json
"""
import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src"))

K_CXR = int(os.environ.get("K_CXR", "32"))
CXR_EMB = os.environ.get("CXR_EMB", "raddino_multi_anatomy_embeddings.pt")
VIEW = os.environ.get("VIEW", "frontal")
OUT_TAG = os.environ.get("OUT_TAG", "combined_stack_frontal_anat")
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
ALPHA_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]
SEED = 42


# --------------------------------------------------------------------------- #
# model factories
# --------------------------------------------------------------------------- #
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


def _meta_clf(C):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("s", StandardScaler()),
                     ("m", LogisticRegression(C=C, class_weight="balanced", max_iter=2000))])


def _meta_reg(a):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=a))])


def _select_C(X, y):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    if len(np.unique(y)) < 2:
        return 1.0
    best_C, best = 1.0, -1
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    for C in C_GRID:
        try:
            s = cross_val_score(_meta_clf(C), X, y, cv=skf, scoring="roc_auc").mean()
        except Exception:  # noqa: BLE001
            continue
        if s > best:
            best, best_C = s, C
    return best_C


def _select_alpha(X, y):
    from sklearn.model_selection import KFold, cross_val_score
    best_a, best = 1.0, -1e9
    kf = KFold(3, shuffle=True, random_state=SEED)
    for a in ALPHA_GRID:
        try:
            s = cross_val_score(_meta_reg(a), X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
        except Exception:  # noqa: BLE001
            continue
        if s > best:
            best, best_a = s, a
    return best_a


# --------------------------------------------------------------------------- #
# base learner: multi-instance CXR, PCA, leakage-safe OOF over the outer folds
# --------------------------------------------------------------------------- #
def cxr_multi_oof(folds, y, kind, IX, Isid, Iw, row_of):
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    out = np.full(len(y), np.nan)
    base = _hgb_clf() if kind == "clf" else _hgb_reg()
    y_inst = np.array([y[row_of[s]] for s in Isid])
    for tr_ids, te_ids in folds:
        tr_sid, te_sid = set(int(s) for s in tr_ids), set(int(s) for s in te_ids)
        trm = np.where(np.isin(Isid, list(tr_sid)) & ~np.isnan(y_inst))[0]
        tem = np.where(np.isin(Isid, list(te_sid)) & ~np.isnan(y_inst))[0]
        if kind == "clf" and len(np.unique(y_inst[trm])) < 2:
            continue
        if kind == "reg" and len(trm) < 10:
            continue
        pca = PCA(min(K_CXR, len(trm) - 1), random_state=0).fit(IX[trm])
        Ztr, Zte = pca.transform(IX[trm]), pca.transform(IX[tem])
        if kind == "clf":
            m = clone(base).fit(Ztr, y_inst[trm].astype(int), sample_weight=Iw[trm])
            pi = m.predict_proba(Zte)[:, 1]
        else:
            m = clone(base).fit(Ztr, y_inst[trm], sample_weight=Iw[trm])
            pi = m.predict(Zte)
        for s, val in pd.DataFrame({"sid": Isid[tem], "p": pi}).groupby("sid")["p"].mean().items():
            out[row_of[int(s)]] = val
    return out


def stack_binary(feats, y, folds, row_of, sids):
    meta = np.full(len(y), np.nan)
    for tr_ids, te_ids in folds:
        tr = [row_of[s] for s in tr_ids if s in row_of]
        te = [row_of[s] for s in te_ids if s in row_of]
        trm = [i for i in tr if not np.isnan(y[i]) and not np.isnan(feats[i]).any()]
        tem = [i for i in te if not np.isnan(feats[i]).any()]
        if len(np.unique(y[trm])) < 2:
            continue
        C = _select_C(feats[trm], y[trm].astype(int))
        m = _meta_clf(C).fit(feats[trm], y[trm].astype(int))
        meta[tem] = m.predict_proba(feats[tem])[:, 1]
    return meta


def stack_reg(feats, y, folds, row_of):
    meta = np.full(len(y), np.nan)
    for tr_ids, te_ids in folds:
        tr = [row_of[s] for s in tr_ids if s in row_of]
        te = [row_of[s] for s in te_ids if s in row_of]
        trm = [i for i in tr if not np.isnan(y[i]) and not np.isnan(feats[i]).any()]
        tem = [i for i in te if not np.isnan(feats[i]).any()]
        if len(trm) < 10:
            continue
        a = _select_alpha(feats[trm], y[trm])
        m = _meta_reg(a).fit(feats[trm], y[trm])
        meta[tem] = m.predict(feats[tem])
    return meta


def main():
    from multimodal_aorta.data.splits import load_fold_assignments
    from multimodal_aorta.training.bootstrap import (
        bootstrap_ci, paired_bootstrap_diff, auroc, auprc, mae, r2, fmt)

    cohort = pd.read_csv(os.path.join(PC, "cohort_triple.csv"))
    sids = cohort["subject_id"].astype(int).tolist()
    row_of = {s: i for i, s in enumerate(sids)}
    folds = load_fold_assignments(os.path.join(PC, "fold_assignments.csv"), n_splits=5)
    diam = {"root": cohort["target_root"].to_numpy(float),
            "asc": cohort["target_asc"].to_numpy(float)}
    log.info("VIEW=%s CXR_EMB=%s K_CXR=%d OUT_TAG=%s | n=%d", VIEW, CXR_EMB, K_CXR, OUT_TAG, len(sids))

    # ---- multi-instance CXR arrays ----
    inst = pd.read_csv(os.path.join(PC, "cxr_instances.csv"))
    emb = torch.load(os.path.join(PC, CXR_EMB), map_location="cpu", weights_only=False)
    emb = {str(k): (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in emb.items()}
    inst = inst[inst.dicom_id.astype(str).isin(emb)].copy()
    inst["sid"] = inst.subject_id.astype(int)
    inst = inst[inst.sid.isin(set(sids))]
    if VIEW == "frontal":
        inst = inst[inst.view_position.isin(["PA", "AP"])]
    inst = inst.reset_index(drop=True)
    IX = np.vstack([emb[str(d)] for d in inst.dicom_id]).astype(np.float32)
    Isid = inst.sid.to_numpy()
    n_per = inst.groupby("sid").size().to_dict()
    Iw = np.array([1.0 / n_per[s] for s in Isid])                    # patient-equal weighting
    log.info("CXR instances: %d across %d patients (VIEW=%s)", len(inst), inst.sid.nunique(), VIEW)

    # ---- full-cohort EHR base OOF (the hardest floor) ----
    ehr = pd.read_csv(os.path.join(ROOT, "outputs", "ehr_baseline", "oof_predictions.csv"))

    def ehr_vec(site, ep, col):
        e = ehr[(ehr.site == site) & (ehr.endpoint == ep)]
        m = dict(zip(e.subject_id.astype(int), e[col]))
        return np.array([m.get(s, np.nan) for s in sids])

    results = {"cxr_emb": CXR_EMB, "view": VIEW, "k_cxr": K_CXR, "n_patients": len(sids), "sites": {}}
    for site in ("root", "asc"):
        d = diam[site]
        t = {"diam": d,
             "ge40": np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float)),
             "ge45": np.where(np.isnan(d), np.nan, (d >= 4.5).astype(float))}
        d_ehr = ehr_vec(site, "diam", "pred_value")
        site_res = {}
        for ep in ("ge40", "ge45"):
            y = t[ep]
            p_cxr = cxr_multi_oof(folds, y, "clf", IX, Isid, Iw, row_of)
            p_ehr = ehr_vec(site, ep, "pred_prob")
            meta = stack_binary(np.column_stack([p_cxr, p_ehr, d_ehr]), y, folds, row_of, sids)
            # parameter-free rank-mean ensemble (as good as the meta at this n)
            ra = pd.Series(p_cxr).rank(pct=True).to_numpy(); rb = pd.Series(p_ehr).rank(pct=True).to_numpy()
            ens = np.where(np.isnan(p_cxr) | np.isnan(p_ehr), np.nan, 0.5 * ra + 0.5 * rb)
            dv = paired_bootstrap_diff(y, meta, p_ehr, auroc)
            site_res[ep] = {
                "pos": int(np.nansum(y)),
                "stack_auroc": fmt(bootstrap_ci(y, meta, auroc)),
                "stack_auprc": fmt(bootstrap_ci(y, meta, auprc)),
                "cxr_base_auroc": fmt(bootstrap_ci(y, p_cxr, auroc)),
                "ehr_floor_auroc": fmt(bootstrap_ci(y, p_ehr, auroc)),
                "rankmean_auroc": fmt(bootstrap_ci(y, ens, auroc)),
                "delta_stack_minus_ehr": fmt(dv),
                "sig_vs_ehr": bool(dv[1] > 0 or dv[2] < 0),
            }
            log.info("[%s %s] stack=%s cxr=%s ehr=%s rankmean=%s | Δstack−EHR=%s %s",
                     site, ep, site_res[ep]["stack_auroc"], site_res[ep]["cxr_base_auroc"],
                     site_res[ep]["ehr_floor_auroc"], site_res[ep]["rankmean_auroc"],
                     site_res[ep]["delta_stack_minus_ehr"], "*" if site_res[ep]["sig_vs_ehr"] else "")
        # regression
        y = t["diam"]
        d_cxr = cxr_multi_oof(folds, y, "reg", IX, Isid, Iw, row_of)
        meta_r = stack_reg(np.column_stack([d_cxr, d_ehr]), y, folds, row_of)
        dr = paired_bootstrap_diff(y, meta_r, d_ehr, r2, need_both_classes=False)
        site_res["diam"] = {
            "stack_mae": fmt(bootstrap_ci(y, meta_r, mae, need_both_classes=False)),
            "stack_r2": fmt(bootstrap_ci(y, meta_r, r2, need_both_classes=False)),
            "ehr_floor_r2": fmt(bootstrap_ci(y, d_ehr, r2, need_both_classes=False)),
            "delta_stack_minus_ehr_r2": fmt(dr),
            "sig_r2_vs_ehr": bool(dr[1] > 0 or dr[2] < 0),
        }
        log.info("[%s diam] stack R2=%s (EHR floor %s) ΔR2=%s %s", site,
                 site_res["diam"]["stack_r2"], site_res["diam"]["ehr_floor_r2"],
                 site_res["diam"]["delta_stack_minus_ehr_r2"], "*" if site_res["diam"]["sig_r2_vs_ehr"] else "")
        results["sites"][site] = site_res

    out_dir = os.path.join(ROOT, "outputs", OUT_TAG)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s/results.json", out_dir)


if __name__ == "__main__":
    main()
