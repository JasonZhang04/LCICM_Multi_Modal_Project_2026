"""
Episode-level rebuild of the v6 best model (geometry-augmented, regression-derived).

Same two mechanisms as scripts/train_geometry_stack.py — complementary engineered
geometry + regression-derived classification (rank by predicted diameter) — but the
modelling unit is the EPISODE, folds are patient-grouped, and CIs are patient-clustered.

Data flow
    episodes.csv                        one row per (patient, echo study)
    episode_cxr_instances.csv           (episode, dicom_id) pre-index frontal CXRs
    raddino_patchpool_embeddings.pt     {dicom_id: {cls, aortapool, heartpool}}
    cxr_geometry_features.csv           17 geometry features per dicom_id
    outputs/ehr_baseline_episode/oof_predictions.csv   per-episode EHR diameter OOF
  ->  per-instance [cls|aortapool|heartpool] (PCA-32 each, fit per training fold)
      + geometry -> HGB diameter regression, instance-weighted 1/n, averaged per
      episode -> ridge stack with EHR diameter -> rank by predicted diameter as the
      >=4.0 / >=4.5 score.

MOCK=1 synthesizes deterministic random embeddings + geometry per dicom_id so the
whole pipeline (episode keying, patient-grouped folds, instance->episode averaging,
per-fold PCA, ridge stack, cluster bootstrap) can be smoke-tested before the images
are downloaded. Real image features drop in unchanged once extraction has run.

Env: SEEDS="42" (single) or "1,2,3,4,5" (repeated-CV stability); MOCK=1 for the dry run.
Run: sbatch scripts/slurm_geometry_stack_episode.sh -> outputs/geometry_stack_episode/results.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src"))

K = int(os.environ.get("K_PCA", "32"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42").split(",")]
MOCK = os.environ.get("MOCK", "0") == "1"
GEOM = ["thoracic_width", "cardiothoracic_ratio", "mediastinal_ratio", "med_upper_ratio",
        "med_mid_ratio", "med_lower_ratio", "med_upper_over_lower",
        "aorta_w_frac", "aorta_h_frac", "aorta_area_frac", "aorta_area_over_thorax",
        "aorta_knob_lateral", "aorta_centroid_offset", "aorta_top_y",
        "heart_w_frac", "heart_area_ratio", "med_area_ratio"]
POOLS = ("cls", "aortapool", "heartpool")
MOCK_DIMS = {"cls": 768, "aortapool": 768, "heartpool": 768}


def _hgb_reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def load_instance_features(inst):
    """
    Return (blocks, Xgeom) for the instance rows in `inst` (must have dicom_id).
    blocks = list of (n_inst x dim) arrays per pool; Xgeom = (n_inst x 17).
    MOCK synthesizes deterministic per-dicom features so plumbing can be tested
    without the images; otherwise loads the real .pt + geometry CSV.
    """
    dids = [str(d) for d in inst.dicom_id]
    if MOCK:
        log.warning("MOCK=1 — synthesizing random embeddings + geometry (smoke test only)")
        blocks = []
        for pool in POOLS:
            X = np.zeros((len(dids), MOCK_DIMS[pool]), dtype=np.float32)
            for i, d in enumerate(dids):
                rng = np.random.default_rng(abs(hash((pool, d))) % (2**32))
                X[i] = rng.standard_normal(MOCK_DIMS[pool]).astype(np.float32)
            blocks.append(X)
        rng = np.random.default_rng(0)
        Xgeom = rng.standard_normal((len(dids), len(GEOM))).astype(np.float32)
        return blocks, Xgeom
    import torch
    pp = torch.load(os.path.join(PC, os.environ.get("PATCHPOOL_OUT",
                    "raddino_patchpool_embeddings_episode.pt")),
                    map_location="cpu", weights_only=False)
    pp = {str(k): {kk: (vv.numpy() if hasattr(vv, "numpy") else np.asarray(vv))
                   for kk, vv in v.items()} for k, v in pp.items()}
    geo = pd.read_csv(os.path.join(PC, os.environ.get("GEOMETRY_OUT",
                      "cxr_geometry_features_episode.csv")))
    inst = inst.merge(geo[["dicom_id"] + GEOM], on="dicom_id", how="left")
    blocks = [np.vstack([pp[d][pool] for d in dids]).astype(np.float32) for pool in POOLS]
    Xgeom = inst[GEOM].to_numpy(np.float32)
    return blocks, Xgeom


def cxr_base_oof(folds, y_row, blocks, Xgeom, I_eid, I_sid_ignored, Iw, row_of):
    """Multi-instance CXR diameter regression; PCA fit on TRAIN-fold instances only.
    y_row is the per-EPISODE-row diameter; instances inherit their episode's label.
    folds are (train_eids, test_eids) of episode_id strings."""
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    out = np.full(len(y_row), np.nan)
    base = _hgb_reg()
    y_inst = np.array([y_row[row_of[e]] for e in I_eid])
    for tr_eids, te_eids in folds:
        trs, tes = set(map(str, tr_eids)), set(map(str, te_eids))
        in_tr = np.array([e in trs for e in I_eid])
        in_te = np.array([e in tes for e in I_eid])
        trm = np.where(in_tr & ~np.isnan(y_inst))[0]
        tem = np.where(in_te)[0]
        if len(trm) < 10 or len(tem) == 0:
            continue
        ptr, pte = [], []
        for B in blocks:
            pca = PCA(min(K, B.shape[1], len(trm) - 1), random_state=0).fit(B[trm])
            ptr.append(pca.transform(B[trm])); pte.append(pca.transform(B[tem]))
        ptr.append(Xgeom[trm]); pte.append(Xgeom[tem])
        m = clone(base).fit(np.concatenate(ptr, 1), y_inst[trm], sample_weight=Iw[trm])
        pi = m.predict(np.concatenate(pte, 1))
        for e, v in pd.DataFrame({"e": I_eid[tem], "p": pi}).groupby("e")["p"].mean().items():
            out[row_of[str(e)]] = v
    return out


def ridge_stack(feats, y, folds, row_of):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    out = np.full(len(y), np.nan)
    for tr_eids, te_eids in folds:
        tr = [row_of[e] for e in map(str, tr_eids) if e in row_of]
        te = [row_of[e] for e in map(str, te_eids) if e in row_of]
        trm = [i for i in tr if not np.isnan(y[i]) and not np.isnan(feats[i]).any()]
        tem = [i for i in te if not np.isnan(feats[i]).any()]
        if len(trm) < 10 or not tem:
            continue
        best_a, best = 1.0, -1e9
        for a in (0.3, 1.0, 3.0, 10.0, 30.0):
            pipe = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=a))])
            try:
                sc = cross_val_score(pipe, feats[trm], y[trm],
                                     cv=KFold(3, shuffle=True, random_state=42),
                                     scoring="neg_mean_absolute_error").mean()
            except Exception:
                continue
            if sc > best:
                best, best_a = sc, a
        m = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=best_a))]).fit(feats[trm], y[trm])
        out[tem] = m.predict(feats[tem])
    return out


def main():
    from multimodal_aorta.data.episodes import load_episodes
    from multimodal_aorta.data.splits import make_grouped_cv_folds
    from multimodal_aorta.training.bootstrap import cluster_bootstrap_ci, auroc, mae, r2, fmt

    ep = load_episodes(PC, require_ecg=False)
    eids = ep["episode_id"].astype(str).tolist()
    row_of = {e: i for i, e in enumerate(eids)}
    sid_of_row = ep["subject_id"].to_numpy(int)
    diam = {"root": ep["target_root"].to_numpy(float), "asc": ep["target_asc"].to_numpy(float)}

    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    from multimodal_aorta.data.episodes import make_episode_id
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    inst = inst[inst.episode_id.isin(set(eids))].reset_index(drop=True)
    if not MOCK:
        import torch
        pp = torch.load(os.path.join(PC, os.environ.get("PATCHPOOL_OUT",
                        "raddino_patchpool_embeddings_episode.pt")),
                        map_location="cpu", weights_only=False)
        have = set(map(str, pp.keys()))
        inst = inst[inst.dicom_id.astype(str).isin(have)].reset_index(drop=True)
    blocks, Xgeom = load_instance_features(inst)
    I_eid = inst.episode_id.astype(str).to_numpy()
    I_sid = inst.subject_id.to_numpy(int)
    n_per = inst.groupby("episode_id").size().to_dict()
    Iw = np.array([1.0 / n_per[e] for e in I_eid])
    log.info("instances=%d episodes-with-cxr=%d patients=%d MOCK=%s",
             len(inst), inst.episode_id.nunique(), inst.subject_id.nunique(), MOCK)

    ehr = pd.read_csv(os.path.join(ROOT, "outputs", "ehr_baseline_episode", "oof_predictions.csv"))
    ehr["episode_id"] = ehr["episode_id"].astype(str)
    def ehr_diam(site):
        e = ehr[(ehr.site == site) & (ehr.endpoint == "diam")]
        m = dict(zip(e.episode_id, e.pred_value))
        return np.array([m.get(x, np.nan) for x in eids])

    from multimodal_aorta.training.bootstrap import paired_cluster_bootstrap_diff
    results = {"seeds": SEEDS, "mock": MOCK, "n_episodes": len(eids),
               "n_patients": int(ep.subject_id.nunique()), "sites": {}}
    oof_rows = []   # per-episode OOF for reproducibility (audit Problem 5)
    for site in ("root", "asc"):
        d = diam[site]; d_ehr = ehr_diam(site)
        y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        y45 = np.where(np.isnan(d), np.nan, (d >= 4.5).astype(float))
        per_seed = {"ge40": [], "ge45": [], "r2": [], "mae": []}
        seed_preds = []
        for seed in SEEDS:
            folds = make_grouped_cv_folds(ep, stratify_col="anyAD", n_splits=5, seed=seed)
            d_cxr = cxr_base_oof(folds, d, blocks, Xgeom, I_eid, I_sid, Iw, row_of)
            d_st = ridge_stack(np.column_stack([d_cxr, d_ehr]), d, folds, row_of)
            seed_preds.append(d_st)
            def _m(y, p, fn, nb=True):
                k = ~np.isnan(y) & ~np.isnan(p)
                if k.sum() < 3 or (nb and len(np.unique(y[k])) < 2):
                    return np.nan
                return float(fn(y[k], p[k]))
            per_seed["ge40"].append(_m(y40, d_st, auroc))
            per_seed["ge45"].append(_m(y45, d_st, auroc))
            per_seed["r2"].append(_m(d, d_st, r2, nb=False))
            per_seed["mae"].append(_m(d, d_st, mae, nb=False))
        # Average the per-seed OOF predictions -> a stable score for CIs / paired tests
        # / saved OOF (repeated-CV reduces fold-split variance in the per-patient score).
        d_bar = np.nanmean(np.column_stack(seed_preds), axis=1)
        sr = {k: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                  "per_seed": [round(float(x), 4) for x in v]} for k, v in per_seed.items()}
        sr["ge40_ci"] = fmt(cluster_bootstrap_ci(y40, d_bar, sid_of_row, auroc))
        sr["r2_ci"] = fmt(cluster_bootstrap_ci(d, d_bar, sid_of_row, r2, need_both_classes=False))
        # PAIRED cluster-bootstrap vs the EHR floor (both are predicted diameters;
        # ranking by them gives the ge40/ge45 score). CI excluding 0 => stack beats floor.
        sr["ge40_vs_floor"] = fmt(paired_cluster_bootstrap_diff(y40, d_bar, d_ehr, sid_of_row, auroc))
        sr["ge45_vs_floor"] = fmt(paired_cluster_bootstrap_diff(y45, d_bar, d_ehr, sid_of_row, auroc))
        sr["r2_vs_floor"] = fmt(paired_cluster_bootstrap_diff(
            d, d_bar, d_ehr, sid_of_row, r2, need_both_classes=False))
        results["sites"][site] = sr
        for i, e in enumerate(eids):
            oof_rows.append({"episode_id": e, "subject_id": int(sid_of_row[i]), "site": site,
                             "diam_true": d[i], "pred_stack": d_bar[i], "pred_floor": d_ehr[i]})
        log.info("[%s] ge40 %.3f+-%.3f | ge45 %.3f+-%.3f | R2 %.3f+-%.3f | MAE %.3f",
                 site, sr["ge40"]["mean"], sr["ge40"]["sd"], sr["ge45"]["mean"], sr["ge45"]["sd"],
                 sr["r2"]["mean"], sr["r2"]["sd"], sr["mae"]["mean"])
        log.info("    ge40 CI %s | vs floor ge40 %s | vs floor R2 %s",
                 sr["ge40_ci"], sr["ge40_vs_floor"], sr["r2_vs_floor"])

    out_dir = os.path.join(ROOT, "outputs",
                           "geometry_stack_episode" + ("_mock" if MOCK else ""))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(oof_rows).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    log.info("Saved -> %s/{results.json, oof_predictions.csv}", out_dir)


if __name__ == "__main__":
    main()
