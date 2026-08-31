"""
Final integrated multimodal model: combine the three ablation winners.

  - CXR: the strong MULTI-INSTANCE representation (per-image RAD-DINO patchpool + geometry,
    per-fold PCA, instance-weighted HGB diameter regressor averaged per episode) — reused
    verbatim from train_geometry_stack_episode.cxr_base_oof. This beat episode-mean CXR.
  - Fusion: regularized LINEAR EARLY fusion (the fusion experiment's winner — a ridge over
    the concatenated modality features beat both the late ridge-of-scalars and the HGB tree;
    the cooperative-learning agreement penalty was never selected => modalities redundant).
  - ECG: the raw-WAVEFORM penultimate embedding (nested-OOF), which added a small but real
    R^2 increment where the ECG measurement summary added nothing.

Final = Ridge over [ d_cxr(multi-instance, OOF) | EHR 12 raw features | ECG-embedding PCA-32 ],
nested patient-grouped OOF, scored by predicted diameter. Reported on the CXR-having,
labelled episodes (~20,429 root / 18,510 asc; ECG-missing -> ECG block imputed to 0)
against two baselines on the same folds/episodes:
  FLOOR         = EHR-only diameter (the clinical floor)
  GEOM_STACK    = the current headline: late ridge over [d_cxr, d_ehr]

Env: SEEDS (default 1,2,3), K_PCA (32). Run: sbatch scripts/slurm_final_model_episode.sh
Out: outputs/final_model_episode/{results.json, oof_predictions.csv}
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
FOLD_MODE = os.environ.get("FOLD_MODE", "seeds")   # "seeds" | "immutable" (see main)
EHR_COLS = ["age", "sex", "height_cm", "weight_kg", "bmi", "bsa", "sbp", "dbp",
            "height_missing", "weight_missing", "bsa_missing", "bp_missing"]
import train_geometry_stack_episode as gs          # cxr_base_oof, load_instance_features, ridge_stack


def linear_early_oof(folds, y, d_cxr, X_ehr, X_ecg, row_of, sid):
    """Nested-OOF regularized LINEAR early fusion: per fold, standardize EHR + PCA the ECG
    embedding on the training rows only, concat with the (already-OOF) multi-instance CXR
    scalar, fit a Ridge (alpha by inner MAE CV), predict the held-out fold. ECG-missing rows
    get a zero ECG block (fit on present rows only)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GroupKFold
    from multimodal_aorta.training.bootstrap import mae
    out = np.full(len(y), np.nan)
    for tr_eids, te_eids in folds:
        tr = np.array([row_of[e] for e in map(str, tr_eids) if e in row_of])
        te = np.array([row_of[e] for e in map(str, te_eids) if e in row_of])
        trm = tr[~np.isnan(y[tr]) & ~np.isnan(d_cxr[tr])]
        tem = te[~np.isnan(d_cxr[te])]
        if len(trm) < 50 or not len(tem):
            continue
        sc = StandardScaler().fit(np.nan_to_num(X_ehr[trm]))
        fin = ~np.isnan(X_ecg[trm]).any(1)
        use_ecg = fin.sum() >= K + 1            # guard: enough ECG-present train rows to PCA
        if use_ecg:
            pca = PCA(min(K, X_ecg.shape[1], fin.sum() - 1), random_state=0).fit(X_ecg[trm][fin])
            esc = StandardScaler().fit(pca.transform(X_ecg[trm][fin]))

        def feat(idx):
            ehr = sc.transform(np.nan_to_num(X_ehr[idx])).astype(np.float32)
            if not use_ecg:
                return np.column_stack([d_cxr[idx], ehr])
            ec = np.zeros((len(idx), esc.mean_.shape[0]), np.float32)
            m = ~np.isnan(X_ecg[idx]).any(1)
            if m.any():
                ec[m] = esc.transform(pca.transform(X_ecg[idx][m])).astype(np.float32)
            return np.column_stack([d_cxr[idx], ehr, ec])

        Ftr, Fte = feat(trm), feat(tem)
        ytr = y[trm]; gtr = sid[trm]
        best_a, best = 1.0, 1e9
        for a in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            errs = [mae(ytr[ib], Ridge(alpha=a).fit(Ftr[ia], ytr[ia]).predict(Ftr[ib]))
                    for ia, ib in GroupKFold(3).split(Ftr, groups=gtr)]     # patient-grouped inner CV
            if np.mean(errs) < best:
                best, best_a = float(np.mean(errs)), a
        out[tem] = Ridge(alpha=best_a).fit(Ftr, ytr).predict(Fte)
    return out


def main():
    from multimodal_aorta.data.episodes import load_episodes, make_episode_id
    from multimodal_aorta.data.splits import make_grouped_cv_folds
    from multimodal_aorta.training.bootstrap import (
        cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt)
    import torch

    ep = load_episodes(PC, require_ecg=False)
    eids = ep.episode_id.astype(str).tolist(); row_of = {e: i for i, e in enumerate(eids)}
    sid = ep.subject_id.to_numpy(int)
    diam = {"root": ep.target_root.to_numpy(float), "asc": ep.target_asc.to_numpy(float)}

    # --- multi-instance CXR inputs (reuse the geometry-stack loader) ---
    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    inst = inst[inst.episode_id.isin(set(eids))].reset_index(drop=True)
    pp = torch.load(os.path.join(PC, "raddino_patchpool_embeddings_episode.pt"), map_location="cpu", weights_only=False)
    inst = inst[inst.dicom_id.astype(str).isin(set(map(str, pp.keys())))].reset_index(drop=True)
    blocks, Xgeom = gs.load_instance_features(inst)
    I_eid = inst.episode_id.astype(str).to_numpy(); I_sid = inst.subject_id.to_numpy(int)
    n_per = inst.groupby("episode_id").size().to_dict()
    Iw = np.array([1.0 / n_per[e] for e in I_eid])

    # --- EHR raw + EHR floor OOF ---
    ehr = pd.read_csv(os.path.join(PC, "ehr_features_episode.csv")); ehr["episode_id"] = ehr.episode_id.astype(str)
    em = ehr.set_index("episode_id")[EHR_COLS]
    X_ehr = np.full((len(eids), len(EHR_COLS)), np.nan, np.float32)
    for e in em.index:
        if e in row_of: X_ehr[row_of[e]] = em.loc[e].to_numpy(np.float32)
    floor = pd.read_csv(os.path.join(ROOT, "outputs", "ehr_baseline_episode", "oof_predictions.csv"))
    floor["episode_id"] = floor.episode_id.astype(str)
    def ehr_diam(site):
        e = floor[(floor.site == site) & (floor.endpoint == "diam")]
        m = dict(zip(e.episode_id, e.pred_value)); return np.array([m.get(x, np.nan) for x in eids])

    # --- ECG waveform embedding ---
    ew = os.path.join(ROOT, "outputs", "ecg_waveform_episode")
    E = np.load(os.path.join(ew, "ecg_embeddings.npy")); eix = pd.read_csv(os.path.join(ew, "ecg_embedding_index.csv"))
    X_ecg = np.full((len(eids), E.shape[1]), np.nan, np.float32)
    for r, e in zip(E, eix.episode_id.astype(str)):
        if e in row_of: X_ecg[row_of[e]] = r
    log.info("episodes %d | CXR instances %d | ECG-emb %d", len(eids), len(inst), int((~np.isnan(X_ecg[:, 0])).sum()))

    # FOLD_MODE: "seeds" = repeated CV over SEEDS (regenerated folds; fine when the
    # precomputed OOF features cancel in paired diffs). "immutable" = the single immutable
    # seed-42 partition, which MATCHES the folds the precomputed features (ECG embedding,
    # EHR floor, and the fine-tuned CXR) were generated on — use this once the fine-tuned
    # CXR is ingested, to avoid cross-scheme stacking optimism (review issue 1).
    from multimodal_aorta.data.splits import load_episode_folds
    if FOLD_MODE == "immutable":
        fold_sets = [load_episode_folds(os.path.join(PC, "episode_fold_assignments.csv"))]
        log.info("FOLD_MODE=immutable (single seed-42 partition, matched to precomputed OOF)")
    else:
        fold_sets = [make_grouped_cv_folds(ep, stratify_col="anyAD", n_splits=5, seed=s) for s in SEEDS]

    results = {"seeds": SEEDS, "fold_mode": FOLD_MODE, "sites": {}}; oof_rows = []
    for site in ("root", "asc"):
        d = diam[site]; d_ehr = ehr_diam(site)
        y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        FIN, GEO, FLOOR = [], [], []
        for folds in fold_sets:
            d_cxr = gs.cxr_base_oof(folds, d, blocks, Xgeom, I_eid, I_sid, Iw, row_of)
            GEO.append(gs.ridge_stack(np.column_stack([d_cxr, d_ehr]), d, folds, row_of))   # headline late fusion
            FIN.append(linear_early_oof(folds, d, d_cxr, X_ehr, X_ecg, row_of, sid))         # final linear early
        d_fin = np.nanmean(np.column_stack(FIN), 1); d_geo = np.nanmean(np.column_stack(GEO), 1)
        m = ~np.isnan(d) & ~np.isnan(d_fin) & ~np.isnan(d_geo) & ~np.isnan(d_ehr); g = sid[m]
        sr = {
            "n": int(m.sum()),
            "floor_r2": fmt(cluster_bootstrap_ci(d[m], d_ehr[m], g, r2, need_both_classes=False)),
            "geom_stack_r2": fmt(cluster_bootstrap_ci(d[m], d_geo[m], g, r2, need_both_classes=False)),
            "final_r2": fmt(cluster_bootstrap_ci(d[m], d_fin[m], g, r2, need_both_classes=False)),
            "floor_ge40": fmt(cluster_bootstrap_ci(y40[m], d_ehr[m], g, auroc)),
            "geom_stack_ge40": fmt(cluster_bootstrap_ci(y40[m], d_geo[m], g, auroc)),
            "final_ge40": fmt(cluster_bootstrap_ci(y40[m], d_fin[m], g, auroc)),
            "final_vs_floor_r2": fmt(paired_cluster_bootstrap_diff(d[m], d_fin[m], d_ehr[m], g, r2, need_both_classes=False)),
            "final_vs_floor_ge40": fmt(paired_cluster_bootstrap_diff(y40[m], d_fin[m], d_ehr[m], g, auroc)),
            "final_vs_geomstack_r2": fmt(paired_cluster_bootstrap_diff(d[m], d_fin[m], d_geo[m], g, r2, need_both_classes=False)),
            "final_vs_geomstack_ge40": fmt(paired_cluster_bootstrap_diff(y40[m], d_fin[m], d_geo[m], g, auroc)),
        }
        results["sites"][site] = sr
        log.info("[%s] FINAL R2 %s ge40 %s | floor R2 %s | geom-stack R2 %s", site,
                 sr["final_r2"], sr["final_ge40"], sr["floor_r2"], sr["geom_stack_r2"])
        log.info("[%s]   final vs floor: R2 %s ge40 %s | final vs geom-stack: R2 %s ge40 %s", site,
                 sr["final_vs_floor_r2"], sr["final_vs_floor_ge40"],
                 sr["final_vs_geomstack_r2"], sr["final_vs_geomstack_ge40"])
        for i in np.where(m)[0]:
            oof_rows.append({"episode_id": eids[i], "subject_id": int(sid[i]), "site": site,
                             "diam_true": d[i], "pred_final": d_fin[i], "pred_geom_stack": d_geo[i],
                             "pred_floor": d_ehr[i]})

    out_dir = os.path.join(ROOT, "outputs", "final_model_episode"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(oof_rows).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
