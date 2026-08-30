"""
Modality-value diagnostics (audit Problem A): does the ECG carry any signal UNIQUE
to it, or is the near-zero stacking increment an artifact of the scalar bottleneck?

The headline stack compresses each modality to one predicted-diameter scalar before
fusion, so any ECG signal orthogonal to "diameter as the ECG alone predicts it" is
destroyed before the ridge sees it. Two tests that bypass that bottleneck, both on
features we already have (no new extraction):

  1. EARLY FUSION. One HistGradientBoosting regressor on the FULL concatenated
     per-episode feature space — CXR (per-fold PCA-32 of the episode-mean cls /
     aorta-pool / heart-pool views + mean geometry), EHR (12), ECG (33). Trees model
     feature interactions natively. Compare CXR+EHR vs CXR+EHR+ECG, nested OOF,
     paired on the ECG-having episodes. If ECG adds nothing HERE, the null is real.

  2. RESIDUAL FIT. Train an HGB on the 33 ECG features to predict the CXR+EHR stack's
     out-of-fold RESIDUAL (true diameter - stack prediction). Nested OOF. If the
     residual R^2 ~ 0, the ECG has no information the CXR+EHR model is missing.

This is the modality-value ablation the paper's objective (1) needs, and it is what
makes the ECG null defensible (or overturns it) before investing in a waveform arm.

Run: sbatch scripts/slurm_modality_value_episode.sh   (SEEDS in the wrapper)
Out: outputs/modality_value_episode/results.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src"))

K = int(os.environ.get("K_PCA", "32"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
GEOM = ["thoracic_width", "cardiothoracic_ratio", "mediastinal_ratio", "med_upper_ratio",
        "med_mid_ratio", "med_lower_ratio", "med_upper_over_lower",
        "aorta_w_frac", "aorta_h_frac", "aorta_area_frac", "aorta_area_over_thorax",
        "aorta_knob_lateral", "aorta_centroid_offset", "aorta_top_y",
        "heart_w_frac", "heart_area_ratio", "med_area_ratio"]
EHR_COLS = ["age", "sex", "height_cm", "weight_kg", "bmi", "bsa", "sbp", "dbp",
            "height_missing", "weight_missing", "bsa_missing", "bp_missing"]


def _hgb():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def build_episode_cxr(eids, row_of):
    """Per-episode CXR features: mean over the episode's instances of each pooled
    view (cls/aorta/heart, 768 each) + mean geometry (17)."""
    import torch
    from multimodal_aorta.data.episodes import make_episode_id
    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    inst = inst[inst.episode_id.isin(set(eids))]
    pp = torch.load(os.path.join(PC, "raddino_patchpool_embeddings_episode.pt"),
                    map_location="cpu", weights_only=False)
    pp = {str(k): v for k, v in pp.items()}
    geo = pd.read_csv(os.path.join(PC, "cxr_geometry_features_episode.csv"))
    geo["dicom_id"] = geo["dicom_id"].astype(str)
    geo_map = geo.set_index("dicom_id")[GEOM]
    inst = inst[inst.dicom_id.astype(str).isin(pp)].reset_index(drop=True)

    n = len(eids)
    CLS = np.full((n, 768), np.nan, np.float32); AO = np.full((n, 768), np.nan, np.float32)
    HR = np.full((n, 768), np.nan, np.float32); GEOM_ep = np.full((n, len(GEOM)), np.nan, np.float32)
    for eid, g in inst.groupby("episode_id"):
        dids = [str(d) for d in g.dicom_id]
        cls = np.vstack([np.asarray(pp[d]["cls"]) for d in dids]).mean(0)
        ao = np.vstack([np.asarray(pp[d]["aortapool"]) for d in dids]).mean(0)
        hr = np.vstack([np.asarray(pp[d]["heartpool"]) for d in dids]).mean(0)
        gg = geo_map.reindex(dids).to_numpy(np.float32)
        i = row_of[eid]
        CLS[i], AO[i], HR[i], GEOM_ep[i] = cls, ao, hr, np.nanmean(gg, axis=0)
    log.info("episode CXR features built for %d/%d episodes", int((~np.isnan(CLS[:, 0])).sum()), n)
    return [CLS, AO, HR], GEOM_ep


def early_fusion_oof(folds, y, cxr_views, GEOM_ep, X_ehr, X_ecg, use_ecg, row_of, eids):
    """Nested-OOF early-fusion HGB. Per fold: PCA-32 each CXR view on train, concat
    [pca_views | geom | ehr | (ecg)] -> HGB. Predict every test episode."""
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    out = np.full(len(y), np.nan)
    base = _hgb()
    for tr_eids, te_eids in folds:
        tr = np.array([row_of[e] for e in map(str, tr_eids) if e in row_of])
        te = np.array([row_of[e] for e in map(str, te_eids) if e in row_of])
        trm = tr[~np.isnan(y[tr]) & ~np.isnan(cxr_views[0][tr]).any(1)]
        if len(trm) < 50:
            continue
        Xtr_parts, Xte_parts = [], []
        for V in cxr_views:
            pca = PCA(min(K, V.shape[1], len(trm) - 1), random_state=0).fit(V[trm])
            Xtr_parts.append(pca.transform(V[trm])); Xte_parts.append(pca.transform(V[te]))
        Xtr_parts += [GEOM_ep[trm], X_ehr[trm]]
        Xte_parts += [GEOM_ep[te], X_ehr[te]]
        if use_ecg:
            Xtr_parts.append(X_ecg[trm]); Xte_parts.append(X_ecg[te])
        m = clone(base).fit(np.concatenate(Xtr_parts, 1), y[trm])
        out[te] = m.predict(np.concatenate(Xte_parts, 1))
    return out


def residual_oof(folds, resid, X_ecg, has_ecg, row_of, eids):
    """HGB(ECG) predicting the CXR+EHR OOF residual; nested, ECG-having episodes only."""
    from sklearn.base import clone
    out = np.full(len(resid), np.nan); base = _hgb()
    for tr_eids, te_eids in folds:
        tr = [row_of[e] for e in map(str, tr_eids) if e in row_of and has_ecg[row_of[e]] and not np.isnan(resid[row_of[e]])]
        te = [row_of[e] for e in map(str, te_eids) if e in row_of and has_ecg[row_of[e]]]
        if len(tr) < 50 or not te:
            continue
        m = clone(base).fit(X_ecg[tr], resid[tr])
        out[te] = m.predict(X_ecg[te])
    return out


def main():
    from multimodal_aorta.data.episodes import load_episodes
    from multimodal_aorta.data.splits import make_grouped_cv_folds
    from multimodal_aorta.training.bootstrap import (
        cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt)

    ep = load_episodes(PC, require_ecg=False)
    eids = ep["episode_id"].astype(str).tolist()
    row_of = {e: i for i, e in enumerate(eids)}
    sid = ep["subject_id"].to_numpy(int)
    diam = {"root": ep["target_root"].to_numpy(float), "asc": ep["target_asc"].to_numpy(float)}

    cxr_views, GEOM_ep = build_episode_cxr(eids, row_of)

    ehr = pd.read_csv(os.path.join(PC, "ehr_features_episode.csv")); ehr["episode_id"] = ehr.episode_id.astype(str)
    em = ehr.set_index("episode_id")[EHR_COLS]
    X_ehr = np.full((len(eids), len(EHR_COLS)), np.nan, np.float32)
    for e in em.index:
        if e in row_of: X_ehr[row_of[e]] = em.loc[e].to_numpy(np.float32)

    ecg = pd.read_csv(os.path.join(PC, "ecg_features_episode.csv")); ecg["episode_id"] = ecg.episode_id.astype(str)
    ecg_cols = [c for c in ecg.columns if c.startswith("ecg_")]
    ecm = ecg.set_index("episode_id")[ecg_cols]
    X_ecg = np.full((len(eids), len(ecg_cols)), np.nan, np.float32); has_ecg = np.zeros(len(eids), bool)
    for e in ecm.index:
        if e in row_of: X_ecg[row_of[e]] = ecm.loc[e].to_numpy(np.float32); has_ecg[row_of[e]] = True

    stack = pd.read_csv(os.path.join(ROOT, "outputs", "geometry_stack_episode", "oof_predictions.csv"))
    stack["episode_id"] = stack.episode_id.astype(str)

    results = {"seeds": SEEDS, "n_ecg": int(has_ecg.sum()), "sites": {}}
    for site in ("root", "asc"):
        d = diam[site]
        y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        EF2, EF3, RES = [], [], []
        for seed in SEEDS:
            folds = make_grouped_cv_folds(ep, stratify_col="anyAD", n_splits=5, seed=seed)
            EF2.append(early_fusion_oof(folds, d, cxr_views, GEOM_ep, X_ehr, X_ecg, False, row_of, eids))
            EF3.append(early_fusion_oof(folds, d, cxr_views, GEOM_ep, X_ehr, X_ecg, True, row_of, eids))
            # residual of the SAVED headline stack (not seed-dependent) — compute once
            if not RES:
                sd = stack[stack.site == site].set_index("episode_id")
                pred = np.array([sd.pred_stack.get(e, np.nan) for e in eids])
                resid = d - pred
                RES.append(residual_oof(folds, resid, X_ecg, has_ecg, row_of, eids))
        ef2 = np.nanmean(np.column_stack(EF2), 1); ef3 = np.nanmean(np.column_stack(EF3), 1)
        rr = RES[0]
        m = has_ecg & ~np.isnan(ef3) & ~np.isnan(ef2)
        g = sid[m]
        sd_ = stack[stack.site == site].set_index("episode_id")
        pred = np.array([sd_.pred_stack.get(e, np.nan) for e in eids]); resid = d - pred
        sr = {
            "n_ecg_eval": int(m.sum()),
            "earlyfusion_cxr_ehr_r2": fmt(cluster_bootstrap_ci(d[m], ef2[m], g, r2, need_both_classes=False)),
            "earlyfusion_cxr_ehr_ecg_r2": fmt(cluster_bootstrap_ci(d[m], ef3[m], g, r2, need_both_classes=False)),
            "earlyfusion_ecg_increment_r2": fmt(paired_cluster_bootstrap_diff(d[m], ef3[m], ef2[m], g, r2, need_both_classes=False)),
            "earlyfusion_cxr_ehr_ge40": fmt(cluster_bootstrap_ci(y40[m], ef2[m], g, auroc)),
            "earlyfusion_cxr_ehr_ecg_ge40": fmt(cluster_bootstrap_ci(y40[m], ef3[m], g, auroc)),
            "earlyfusion_ecg_increment_ge40": fmt(paired_cluster_bootstrap_diff(y40[m], ef3[m], ef2[m], g, auroc)),
            # residual test: how much of the CXR+EHR residual can ECG explain?
            "residual_var_cm2": float(np.nanvar(resid[has_ecg])),
            "ecg_explains_residual_r2": fmt(cluster_bootstrap_ci(resid[has_ecg & ~np.isnan(rr)], rr[has_ecg & ~np.isnan(rr)], sid[has_ecg & ~np.isnan(rr)], r2, need_both_classes=False)),
        }
        results["sites"][site] = sr
        log.info("[%s] EARLY FUSION  CXR+EHR R2 %s  ->  +ECG %s  (increment %s)", site,
                 sr["earlyfusion_cxr_ehr_r2"], sr["earlyfusion_cxr_ehr_ecg_r2"], sr["earlyfusion_ecg_increment_r2"])
        log.info("[%s]   ge40 CXR+EHR %s -> +ECG %s (incr %s)", site,
                 sr["earlyfusion_cxr_ehr_ge40"], sr["earlyfusion_cxr_ehr_ecg_ge40"], sr["earlyfusion_ecg_increment_ge40"])
        log.info("[%s] RESIDUAL TEST  ECG explains %s of the CXR+EHR residual", site, sr["ecg_explains_residual_r2"])

    out_dir = os.path.join(ROOT, "outputs", "modality_value_episode"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
