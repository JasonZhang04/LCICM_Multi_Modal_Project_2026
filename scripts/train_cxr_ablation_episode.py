"""
Is the CXR reading the AORTA, or estimating BODY SIZE? (audit Problem B)

Two ablations of the imaging increment over the EHR floor, both on existing features:

  1. COMPLETE-EHR subset. Re-measure the CXR increment restricted to episodes with
     non-missing height, weight AND blood pressure. If the ROOT gain shrinks there
     (imaging was imputing missing body size) while the ASCENDING gain holds, we have
     found the mechanism — an interesting result, not a failure.

  2. AORTA-ONLY CXR. Strip the CXR branch to aorta-focused features only: aorta-pool
     embedding + the 7 aorta_* geometry features. Drop the whole-image cls, heart-pool,
     and the whole-chest geometry (thoracic width, cardiothoracic + mediastinal ratios,
     heart features). If this STILL beats the floor, the imaging signal is anatomically
     localized to the aorta, not body-size estimation — the strongest positive claim.

Floor = EHR-only HGB (same early-fusion pipeline, no CXR block). Increment = paired
cluster-bootstrap( CXR+EHR - floor ). Nested OOF, patient-grouped folds.

Run: sbatch scripts/slurm_cxr_ablation_episode.sh
Out: outputs/cxr_ablation_episode/results.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

K = int(os.environ.get("K_PCA", "32"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
import train_modality_value_episode as mv   # reuse build_episode_cxr, _hgb, GEOM, EHR_COLS

# aorta-specific geometry columns (indices into mv.GEOM): the aorta_* block only
AORTA_GEOM_IDX = [i for i, g in enumerate(mv.GEOM) if g.startswith("aorta_")]


def oof(folds, y, view_blocks, geom_idx, X_ehr, row_of, eids, use_cxr):
    """Nested-OOF HGB. Per fold: PCA-32 each given CXR view on train, concat
    [pca_views | selected geometry | EHR] (CXR blocks omitted if use_cxr=False)."""
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    out = np.full(len(y), np.nan); base = mv._hgb()
    GEOM_ep = view_blocks["geom"]
    for tr_eids, te_eids in folds:
        tr = np.array([row_of[e] for e in map(str, tr_eids) if e in row_of])
        te = np.array([row_of[e] for e in map(str, te_eids) if e in row_of])
        ok = ~np.isnan(y[tr])
        if use_cxr:
            ok &= ~np.isnan(view_blocks["views"][0][tr]).any(1)
        trm = tr[ok]
        if len(trm) < 50:
            continue
        Xtr, Xte = [], []
        if use_cxr:
            for V in view_blocks["views"]:
                pca = PCA(min(K, V.shape[1], len(trm) - 1), random_state=0).fit(V[trm])
                Xtr.append(pca.transform(V[trm])); Xte.append(pca.transform(V[te]))
            if geom_idx:
                Xtr.append(GEOM_ep[trm][:, geom_idx]); Xte.append(GEOM_ep[te][:, geom_idx])
        Xtr.append(X_ehr[trm]); Xte.append(X_ehr[te])
        m = clone(base).fit(np.concatenate(Xtr, 1), y[trm])
        out[te] = m.predict(np.concatenate(Xte, 1))
    return out


def main():
    from multimodal_aorta.data.episodes import load_episodes
    from multimodal_aorta.data.splits import make_grouped_cv_folds
    from multimodal_aorta.training.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt

    ep = load_episodes(PC, require_ecg=False)
    eids = ep["episode_id"].astype(str).tolist(); row_of = {e: i for i, e in enumerate(eids)}
    sid = ep["subject_id"].to_numpy(int)
    diam = {"root": ep["target_root"].to_numpy(float), "asc": ep["target_asc"].to_numpy(float)}
    (CLS, AO, HR), GEOM_ep = mv.build_episode_cxr(eids, row_of)

    ehr = pd.read_csv(os.path.join(PC, "ehr_features_episode.csv")); ehr["episode_id"] = ehr.episode_id.astype(str)
    em = ehr.set_index("episode_id")
    X_ehr = np.full((len(eids), len(mv.EHR_COLS)), np.nan, np.float32)
    complete = np.zeros(len(eids), bool)
    for e in em.index:
        if e in row_of:
            X_ehr[row_of[e]] = em.loc[e, mv.EHR_COLS].to_numpy(np.float32)
            r = em.loc[e]
            complete[row_of[e]] = (r.height_missing == 0 and r.weight_missing == 0 and r.bp_missing == 0)
    log.info("complete-EHR episodes: %d / %d (%.1f%%)", int(complete.sum()), len(eids), 100 * complete.mean())

    FULL = {"views": [CLS, AO, HR], "geom": GEOM_ep}
    AORTA = {"views": [AO], "geom": GEOM_ep}
    results = {"seeds": SEEDS, "n_complete_ehr": int(complete.sum()), "sites": {}}
    for site in ("root", "asc"):
        d = diam[site]
        y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        FLOOR, CXRFULL, CXRAO = [], [], []
        for seed in SEEDS:
            folds = make_grouped_cv_folds(ep, stratify_col="anyAD", n_splits=5, seed=seed)
            FLOOR.append(oof(folds, d, FULL, None, X_ehr, row_of, eids, use_cxr=False))
            CXRFULL.append(oof(folds, d, FULL, list(range(len(mv.GEOM))), X_ehr, row_of, eids, use_cxr=True))
            CXRAO.append(oof(folds, d, AORTA, AORTA_GEOM_IDX, X_ehr, row_of, eids, use_cxr=True))
        fl = np.nanmean(np.column_stack(FLOOR), 1)
        cf = np.nanmean(np.column_stack(CXRFULL), 1)
        ca = np.nanmean(np.column_stack(CXRAO), 1)
        site_out = {}
        for label, mask in [("all", np.ones(len(eids), bool)), ("complete_ehr", complete)]:
            m = mask & ~np.isnan(d) & ~np.isnan(fl) & ~np.isnan(cf) & ~np.isnan(ca)
            g = sid[m]
            site_out[label] = {
                "n": int(m.sum()),
                "floor_r2": fmt(cluster_bootstrap_ci(d[m], fl[m], g, r2, need_both_classes=False)),
                "cxrfull_r2": fmt(cluster_bootstrap_ci(d[m], cf[m], g, r2, need_both_classes=False)),
                "cxrfull_incr_r2": fmt(paired_cluster_bootstrap_diff(d[m], cf[m], fl[m], g, r2, need_both_classes=False)),
                "cxrfull_incr_ge40": fmt(paired_cluster_bootstrap_diff(y40[m], cf[m], fl[m], g, auroc)),
                "aortaonly_r2": fmt(cluster_bootstrap_ci(d[m], ca[m], g, r2, need_both_classes=False)),
                "aortaonly_incr_r2": fmt(paired_cluster_bootstrap_diff(d[m], ca[m], fl[m], g, r2, need_both_classes=False)),
                "aortaonly_incr_ge40": fmt(paired_cluster_bootstrap_diff(y40[m], ca[m], fl[m], g, auroc)),
            }
            log.info("[%s/%s n=%d] floor R2 %s | CXR-full incr R2 %s ge40 %s | AORTA-only incr R2 %s ge40 %s",
                     site, label, site_out[label]["n"], site_out[label]["floor_r2"],
                     site_out[label]["cxrfull_incr_r2"], site_out[label]["cxrfull_incr_ge40"],
                     site_out[label]["aortaonly_incr_r2"], site_out[label]["aortaonly_incr_ge40"])
        results["sites"][site] = site_out

    out_dir = os.path.join(ROOT, "outputs", "cxr_ablation_episode"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
