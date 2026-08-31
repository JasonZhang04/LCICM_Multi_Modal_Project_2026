"""
Temporal holdout evaluation of the final multimodal model.

The touch-once test: every base learner is trained on the TRAIN ERA (anchor_year_group
2008-2013) only, and the model is evaluated on the quarantined HOLDOUT ERA (2014-2016),
which no model has seen. This turns "cross-validated" into "temporally validated".

Base learners (all train-era -> holdout-era):
  d_cxr_frozen : multi-instance RAD-DINO patchpool+geometry HGB (gs.cxr_base_oof, single
                 train->holdout split; PCA fit on train-era instances)
  d_ehr        : HGB on the 12 EHR features, fit on train-era
  d_cxr_ft     : fine-tuned RAD-DINO, from outputs/cxr_finetune_holdout (HOLDOUT run)
  d_ecg        : raw-waveform ECG, from outputs/ecg_waveform_holdout (HOLDOUT run)

Fusion on the holdout is a simple average of the standardized base diameters (the fusion
study showed the modalities are redundant and a regularized linear blend ~= a fixed
average; training a ridge here would need train-era base OOF we don't cheaply have, and a
fixed blend keeps the holdout test clean). Reports holdout R2/AUROC/AUPRC for the floor
(EHR alone) and the ensemble, with cluster bootstrap.

Run: sbatch scripts/slurm_holdout_final.sh   Out: outputs/holdout_final/results.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_geometry_stack_episode as gs
gs.K = int(os.environ.get("K_PCA", "128"))
EHR_COLS = ["age", "sex", "height_cm", "weight_kg", "bmi", "bsa", "sbp", "dbp",
            "height_missing", "weight_missing", "bsa_missing", "bp_missing"]


def hgb_split(Xtr, ytr, Xte):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
                                      min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    ok = ~np.isnan(ytr)
    return m.fit(Xtr[ok], ytr[ok]).predict(Xte)


def load_site(path, col):
    df = pd.read_csv(path); df["episode_id"] = df.episode_id.astype(str)
    return {s: dict(zip(df[df.site == s].episode_id, df[df.site == s][col])) for s in ("root", "asc")}


def main():
    from multimodal_aorta.data.episodes import load_episodes, make_episode_id
    from multimodal_aorta.training.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, auprc, r2, fmt
    import torch

    ep = load_episodes(PC, require_ecg=False)
    hold = pd.read_csv(os.path.join(PC, "episode_temporal_holdout.csv"))
    hmap = dict(zip(hold.episode_id.astype(str), hold.holdout))
    ep["holdout"] = ep.episode_id.astype(str).map(hmap)
    eids = ep.episode_id.astype(str).tolist(); row_of = {e: i for i, e in enumerate(eids)}
    sid = ep.subject_id.to_numpy(int); hold_arr = ep.holdout.to_numpy()
    tr_eids = [e for e in eids if hmap.get(e) == 0]; ho_eids = [e for e in eids if hmap.get(e) == 1]
    log.info("train-era %d | holdout-era %d episodes", len(tr_eids), len(ho_eids))
    diam = {"root": ep.target_root.to_numpy(float), "asc": ep.target_asc.to_numpy(float)}

    # multi-instance CXR (frozen): single train->holdout split
    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    inst = inst[inst.episode_id.isin(set(eids))].reset_index(drop=True)
    pp = torch.load(os.path.join(PC, "raddino_patchpool_embeddings_episode.pt"), map_location="cpu", weights_only=False)
    inst = inst[inst.dicom_id.astype(str).isin(set(map(str, pp.keys())))].reset_index(drop=True)
    blocks, Xgeom = gs.load_instance_features(inst)
    I_eid = inst.episode_id.astype(str).to_numpy(); I_sid = inst.subject_id.to_numpy(int)
    n_per = inst.groupby("episode_id").size().to_dict(); Iw = np.array([1.0 / n_per[e] for e in I_eid])
    folds = [(np.array(tr_eids), np.array(ho_eids))]

    # EHR features
    ehr = pd.read_csv(os.path.join(PC, "ehr_features_episode.csv")); ehr["episode_id"] = ehr.episode_id.astype(str)
    em = ehr.set_index("episode_id")[EHR_COLS]; X_ehr = np.full((len(eids), len(EHR_COLS)), np.nan, np.float32)
    for e in em.index:
        if e in row_of: X_ehr[row_of[e]] = em.loc[e].to_numpy(np.float32)
    tr_idx = np.array([row_of[e] for e in tr_eids]); ho_idx = np.array([row_of[e] for e in ho_eids])

    # fine-tuned CXR + ECG holdout predictions
    cxr_ft = load_site(os.path.join(ROOT, "outputs", "cxr_finetune_holdout", "oof_predictions.csv"), "pred_cxr_ft")
    ecg = load_site(os.path.join(ROOT, "outputs", "ecg_waveform_holdout", "oof_predictions.csv"), "pred_ecg_waveform")

    results = {"k_pca": gs.K, "n_holdout": len(ho_eids), "sites": {}}
    for site in ("root", "asc"):
        d = diam[site]; y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        d_cxr = gs.cxr_base_oof(folds, d, blocks, Xgeom, I_eid, I_sid, Iw, row_of)   # holdout episodes filled
        d_ehr = np.full(len(eids), np.nan); d_ehr[ho_idx] = hgb_split(X_ehr[tr_idx], d[tr_idx], X_ehr[ho_idx])
        d_ft = np.array([cxr_ft[site].get(e, np.nan) for e in eids])
        d_eg = np.array([ecg[site].get(e, np.nan) for e in eids])
        # ensemble = mean of z-scored base diameters (z-scored over holdout episodes present)
        base = [d_cxr, d_ft, d_ehr, d_eg]
        Z = []
        for b in base:
            bb = b.copy(); mask = ~np.isnan(bb)
            if mask.sum() > 2:
                bb = (bb - np.nanmean(bb[ho_idx])) / (np.nanstd(bb[ho_idx]) + 1e-9)
            Z.append(bb)
        ens = np.nanmean(np.column_stack(Z), axis=1)
        m = (hold_arr == 1) & ~np.isnan(d) & ~np.isnan(ens) & ~np.isnan(d_ehr); g = sid[m]
        yy = y40[m]
        sr = {
            "n": int(m.sum()), "pos": int(np.nansum(yy)),
            "floor_r2": fmt(cluster_bootstrap_ci(d[m], d_ehr[m], g, r2, need_both_classes=False)),
            "floor_ge40": fmt(cluster_bootstrap_ci(yy, d_ehr[m], g, auroc)),
            "ensemble_r2": fmt(cluster_bootstrap_ci(d[m], ens[m], g, r2, need_both_classes=False)),
            "ensemble_ge40": fmt(cluster_bootstrap_ci(yy, ens[m], g, auroc)),
            "ensemble_auprc": fmt(cluster_bootstrap_ci(yy, ens[m], g, auprc)),
            "ensemble_vs_floor_r2": fmt(paired_cluster_bootstrap_diff(d[m], ens[m], d_ehr[m], g, r2, need_both_classes=False)),
            "ensemble_vs_floor_ge40": fmt(paired_cluster_bootstrap_diff(yy, ens[m], d_ehr[m], g, auroc)),
            # per-modality holdout (for transparency)
            "cxr_frozen_r2": fmt(cluster_bootstrap_ci(d[m], d_cxr[m], g, r2, need_both_classes=False)),
            "cxr_ft_r2": fmt(cluster_bootstrap_ci(d[m], d_ft[m], sid[m], r2, need_both_classes=False)) if (~np.isnan(d_ft[m])).sum() > 10 else "NA",
            "ecg_r2": fmt(cluster_bootstrap_ci(d[m], d_eg[m], sid[m], r2, need_both_classes=False)) if (~np.isnan(d_eg[m])).sum() > 10 else "NA",
        }
        results["sites"][site] = sr
        log.info("[%s HOLDOUT n=%d pos=%d] ensemble R2 %s ge40 %s auprc %s | floor R2 %s ge40 %s",
                 site, sr["n"], sr["pos"], sr["ensemble_r2"], sr["ensemble_ge40"], sr["ensemble_auprc"],
                 sr["floor_r2"], sr["floor_ge40"])
        log.info("[%s]   ensemble vs floor: R2 %s ge40 %s | per-mod: cxr-frozen %s cxr-ft %s ecg %s", site,
                 sr["ensemble_vs_floor_r2"], sr["ensemble_vs_floor_ge40"],
                 sr["cxr_frozen_r2"], sr["cxr_ft_r2"], sr["ecg_r2"])

    out_dir = os.path.join(ROOT, "outputs", "holdout_final"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
