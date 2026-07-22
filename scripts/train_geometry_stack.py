"""
BEST MODEL as of 2026-07-22 — geometry-augmented, regression-derived aortic screen.

Validated by repeated cross-validation over 5 independent fold seeds:
    root  ge40 AUROC 0.809 +- 0.007   ge45 0.900   diameter R2 0.354
    asc   ge40 AUROC 0.790 +- 0.014   ge45 0.869   diameter R2 0.221
(up from 0.772 / 0.756 with the previous combined_stack.)

Two mechanisms produced the gain:

1. COMPLEMENTARY ENGINEERED GEOMETRY. Alongside the RAD-DINO embedding we feed 17
   explicit measurements derived from the ChestX-Det segmentation masks (mediastinal
   ratios, cardiothoracic ratio, aortic-knob lateralization, areas) — i.e. what a
   radiologist actually reads off the film. Geometry ALONE is weaker than the embedding,
   but the two together beat either, because they are orthogonal.

2. REGRESSION-DERIVED CLASSIFICATION. We do NOT train a classifier on the 32-48 binary
   positives. We regress the CONTINUOUS diameter (all 522 graded labels) and then rank
   patients by predicted diameter to score ">=4.0 cm" / ">=4.5 cm". The binary label
   discards most of the information (3.9 cm and 2.5 cm are both "negative"); the gain
   from this change scales inversely with the number of positives (largest at the 8-10
   positive endpoints), confirming the mechanism.

Pipeline: multi-instance FRONTAL (PA/AP) CXRs -> per-image [cls | aortapool | heartpool]
RAD-DINO views (PCA-32 each, fit per training fold) + 17 geometry features -> HGB
regression on diameter, patient-weighted, averaged per patient -> ridge stack with the
full-cohort EHR diameter prediction -> rank as the classification score.

Prereqs (one-time):
  scripts/extract_multi_cxr_instances.py        -> cxr_instances.csv
  scripts/extract_raddino_patchpool.py          -> raddino_patchpool_embeddings.pt
  scripts/extract_cxr_geometry_features.py      -> cxr_geometry_features.csv
  scripts/train_ehr_baseline.py                 -> outputs/ehr_baseline/oof_predictions.csv

Env: SEEDS="42" (single split) or e.g. "1,2,3,4,5" for the repeated-CV stability estimate.
Run: sbatch scripts/slurm_geometry_stack.sh   -> outputs/geometry_stack/results.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd, torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src"))

K = int(os.environ.get("K_PCA", "32"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42").split(",")]
GEOM = ["thoracic_width", "cardiothoracic_ratio", "mediastinal_ratio", "med_upper_ratio",
        "med_mid_ratio", "med_lower_ratio", "med_upper_over_lower",
        "aorta_w_frac", "aorta_h_frac", "aorta_area_frac", "aorta_area_over_thorax",
        "aorta_knob_lateral", "aorta_centroid_offset", "aorta_top_y",
        "heart_w_frac", "heart_area_ratio", "med_area_ratio"]


def _hgb_reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def cxr_base_oof(folds, y, blocks, Xgeom, Isid, Iw, row_of):
    """Multi-instance CXR diameter regression; PCA fit on train-fold instances only."""
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    out = np.full(len(y), np.nan); base = _hgb_reg()
    y_inst = np.array([y[row_of[s]] for s in Isid])
    for tr_ids, te_ids in folds:
        trs, tes = set(int(s) for s in tr_ids), set(int(s) for s in te_ids)
        trm = np.where(np.isin(Isid, list(trs)) & ~np.isnan(y_inst))[0]
        tem = np.where(np.isin(Isid, list(tes)) & ~np.isnan(y_inst))[0]
        if len(trm) < 10:
            continue
        ptr, pte = [], []
        for B in blocks:
            pca = PCA(min(K, len(trm) - 1), random_state=0).fit(B[trm])
            ptr.append(pca.transform(B[trm])); pte.append(pca.transform(B[tem]))
        ptr.append(Xgeom[trm]); pte.append(Xgeom[tem])
        m = clone(base).fit(np.concatenate(ptr, 1), y_inst[trm], sample_weight=Iw[trm])
        pi = m.predict(np.concatenate(pte, 1))
        for s, v in pd.DataFrame({"sid": Isid[tem], "p": pi}).groupby("sid")["p"].mean().items():
            out[row_of[int(s)]] = v
    return out


def ridge_stack(feats, y, folds, row_of):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    out = np.full(len(y), np.nan)
    for tr_ids, te_ids in folds:
        tr = [row_of[s] for s in tr_ids if s in row_of]
        te = [row_of[s] for s in te_ids if s in row_of]
        trm = [i for i in tr if not np.isnan(y[i]) and not np.isnan(feats[i]).any()]
        tem = [i for i in te if not np.isnan(feats[i]).any()]
        if len(trm) < 10:
            continue
        best_a, best = 1.0, -1e9
        for a in (0.3, 1.0, 3.0, 10.0, 30.0):
            pipe = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=a))])
            try:
                sc = cross_val_score(pipe, feats[trm], y[trm], cv=KFold(3, shuffle=True, random_state=42),
                                     scoring="neg_mean_absolute_error").mean()
            except Exception:
                continue
            if sc > best:
                best, best_a = sc, a
        m = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=best_a))]).fit(feats[trm], y[trm])
        out[tem] = m.predict(feats[tem])
    return out


def main():
    from multimodal_aorta.training.bootstrap import bootstrap_ci, auroc, mae, r2, fmt
    from sklearn.model_selection import StratifiedKFold

    cohort = pd.read_csv(os.path.join(PC, "cohort_triple.csv"))
    s522 = cohort.subject_id.astype(int).tolist()
    row_of = {s: i for i, s in enumerate(s522)}
    strat = cohort.anyAD.fillna(0).astype(int).to_numpy()
    diam = {"root": cohort.target_root.to_numpy(float), "asc": cohort.target_asc.to_numpy(float)}

    inst = pd.read_csv(os.path.join(PC, "cxr_instances.csv"))
    geo = pd.read_csv(os.path.join(PC, "cxr_geometry_features.csv"))
    pp = torch.load(os.path.join(PC, "raddino_patchpool_embeddings.pt"), map_location="cpu", weights_only=False)
    pp = {str(k): {kk: (vv.numpy() if isinstance(vv, torch.Tensor) else np.asarray(vv))
                   for kk, vv in v.items()} for k, v in pp.items()}
    inst["sid"] = inst.subject_id.astype(int)
    inst = inst[inst.view_position.isin(["PA", "AP"]) & inst.sid.isin(set(s522))
                & inst.dicom_id.astype(str).isin(pp)]
    inst = inst.merge(geo[["dicom_id"] + GEOM], on="dicom_id", how="left").reset_index(drop=True)
    dids = [str(d) for d in inst.dicom_id]
    blocks = [np.vstack([pp[d][v] for d in dids]).astype(np.float32)
              for v in ("cls", "aortapool", "heartpool")]
    Xgeom = inst[GEOM].to_numpy(np.float32)
    Isid = inst.sid.to_numpy(); n_per = inst.groupby("sid").size().to_dict()
    Iw = np.array([1.0 / n_per[s] for s in Isid])
    log.info("frontal instances=%d patients=%d | seeds=%s", len(inst), inst.sid.nunique(), SEEDS)

    ehr = pd.read_csv(os.path.join(ROOT, "outputs", "ehr_baseline", "oof_predictions.csv"))
    def ehr_diam(site):
        e = ehr[(ehr.site == site) & (ehr.endpoint == "diam")]
        m = dict(zip(e.subject_id.astype(int), e.pred_value))
        return np.array([m.get(s, np.nan) for s in s522])

    results = {"seeds": SEEDS, "n_patients": len(s522), "sites": {}}
    sids = np.array(s522)
    for site in ("root", "asc"):
        d = diam[site]; d_ehr = ehr_diam(site)
        y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        y45 = np.where(np.isnan(d), np.nan, (d >= 4.5).astype(float))
        per_seed = {"ge40": [], "ge45": [], "r2": [], "mae": []}
        last = None
        for seed in SEEDS:
            skf = StratifiedKFold(5, shuffle=True, random_state=seed)
            folds = [(sids[tr], sids[te]) for tr, te in skf.split(sids, strat)]
            d_cxr = cxr_base_oof(folds, d, blocks, Xgeom, Isid, Iw, row_of)
            d_st = ridge_stack(np.column_stack([d_cxr, d_ehr]), d, folds, row_of)
            last = d_st
            def _m(y, p, fn, nb=True):
                k = ~np.isnan(y) & ~np.isnan(p)
                if k.sum() < 3 or (nb and len(np.unique(y[k])) < 2):
                    return np.nan
                return float(fn(y[k], p[k]))
            per_seed["ge40"].append(_m(y40, d_st, auroc))
            per_seed["ge45"].append(_m(y45, d_st, auroc))
            per_seed["r2"].append(_m(d, d_st, r2, nb=False))
            per_seed["mae"].append(_m(d, d_st, mae, nb=False))
        sr = {k: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                  "per_seed": [round(float(x), 4) for x in v]} for k, v in per_seed.items()}
        # bootstrap CI on the last seed's OOF (patient-level)
        sr["ge40_ci_lastseed"] = fmt(bootstrap_ci(y40, last, auroc))
        sr["r2_ci_lastseed"] = fmt(bootstrap_ci(d, last, r2, need_both_classes=False))
        results["sites"][site] = sr
        log.info("[%s] ge40 %.3f+-%.3f | ge45 %.3f+-%.3f | R2 %.3f+-%.3f | MAE %.3f",
                 site, sr["ge40"]["mean"], sr["ge40"]["sd"], sr["ge45"]["mean"], sr["ge45"]["sd"],
                 sr["r2"]["mean"], sr["r2"]["sd"], sr["mae"]["mean"])
        log.info("    ge40 per-seed: %s | ge40 CI(last seed) %s", sr["ge40"]["per_seed"], sr["ge40_ci_lastseed"])

    out_dir = os.path.join(ROOT, "outputs", "geometry_stack"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s/results.json", out_dir)


if __name__ == "__main__":
    main()
