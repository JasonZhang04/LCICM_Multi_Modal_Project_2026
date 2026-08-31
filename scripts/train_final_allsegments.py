"""
Final model across ALL measured aortic segments (root, ascending, ARCH, DESCENDING).

The echo report measures four aortic diameters; the main model reported only root +
ascending. The PI asked to "include the other measurements on the final OOF". arch_diam
(102k numeric) and descending_diam (30k) are well-populated in the structured EAV table
(annulus has only 165 numeric values -> too sparse, skipped).

We attach arch_cm/desc_cm to the EXISTING episode cohort (no fold/CXR/ECG disturbance) and
run the deployed fusion CORE for each segment:
  d_cxr  = multi-instance RAD-DINO patchpool+geometry HGB (gs.cxr_base_oof, retrained per target)
  + EHR 12 raw features + ECG-waveform embedding PCA-32, regularized LINEAR EARLY fusion.
The fine-tuned RAD-DINO scalar predicts root/asc only, so it is used for those two targets
and omitted for arch/descending (their CNN was never trained -> would be an off-target feature).
The frozen-CXR HGB and the ECG embedding are target-agnostic, so arch/descending are honest
same-architecture results; the ge40 AUROC is reported for consistency but note dilation
thresholds differ by segment (descending in particular is normally smaller), so R2/MAE are
the primary, threshold-free readouts here.

Immutable seed-42 folds (matched to the precomputed ECG/ft OOF). Out: outputs/final_model_allsegments/.
Run: sbatch scripts/slurm_final_allsegments.sh
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
STRUCT = os.path.join(ROOT, "data", "echo", "structured-measurement.csv")
EHR_COLS = ["age", "sex", "height_cm", "weight_kg", "bmi", "bsa", "sbp", "dbp",
            "height_missing", "weight_missing", "bsa_missing", "bp_missing"]
EXTRA = {"arch_diam": "arch_cm", "descending_diam": "desc_cm"}
RANGE = (1.5, 7.0)
import train_geometry_stack_episode as gs
from train_final_model_episode import linear_early_oof


def load_extra_labels(sites):
    keep = []
    for ch in pd.read_csv(STRUCT, usecols=["subject_id", "measurement_id", "test_type", "measurement", "result"],
                          chunksize=2_000_000, low_memory=False):
        ch = ch[ch.measurement.isin(sites) & (ch.test_type == "tte")]
        if len(ch): keep.append(ch)
    df = pd.concat(keep, ignore_index=True)
    df["value"] = pd.to_numeric(df.result, errors="coerce")
    df = df[df.value.notna() & df.value.between(*RANGE)]
    piv = df.pivot_table(index=["subject_id", "measurement_id"], columns="measurement",
                         values="value", aggfunc="median").rename(columns=sites).reset_index()
    return piv


def ehr_floor_oof(folds, y, X_ehr, row_of):
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = np.full(len(y), np.nan)
    for tr_eids, te_eids in folds:
        tr = np.array([row_of[e] for e in map(str, tr_eids) if e in row_of])
        te = np.array([row_of[e] for e in map(str, te_eids) if e in row_of])
        ok = tr[~np.isnan(y[tr])]
        if len(ok) < 50 or not len(te):
            continue
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        out[te] = m.fit(np.nan_to_num(X_ehr[ok]), y[ok]).predict(np.nan_to_num(X_ehr[te]))
    return out


def main():
    from multimodal_aorta.data.episodes import load_episodes, make_episode_id
    from multimodal_aorta.data.splits import load_episode_folds
    from multimodal_aorta.training.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, mae, fmt
    import torch

    ep = load_episodes(PC, require_ecg=False)
    eids = ep.episode_id.astype(str).tolist(); row_of = {e: i for i, e in enumerate(eids)}
    sid = ep.subject_id.to_numpy(int)
    diam = {"root": ep.target_root.to_numpy(float), "asc": ep.target_asc.to_numpy(float)}

    # attach arch/descending labels to the existing episodes
    log.info("scanning EAV for %s ...", list(EXTRA))
    piv = load_extra_labels(EXTRA)
    piv["episode_id"] = [make_episode_id(s, m) for s, m in zip(piv.subject_id, piv.measurement_id)]
    for col in EXTRA.values():
        mm = dict(zip(piv.episode_id, piv[col]))
        diam[col.replace("_cm", "")] = np.array([mm.get(e, np.nan) for e in eids])
    for site in ("root", "asc", "arch", "desc"):
        d = diam[site]; ok = ~np.isnan(d)
        log.info("  %-5s labelled %6d  >=4.0 %5d (%.1f%%)  median %.2f", site, int(ok.sum()),
                 int(np.nansum(d >= 4.0)), 100 * np.nanmean((d[ok] >= 4.0)), float(np.nanmedian(d[ok])))

    # multi-instance CXR inputs
    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    inst = inst[inst.episode_id.isin(set(eids))].reset_index(drop=True)
    pp = torch.load(os.path.join(PC, "raddino_patchpool_embeddings_episode.pt"), map_location="cpu", weights_only=False)
    inst = inst[inst.dicom_id.astype(str).isin(set(map(str, pp.keys())))].reset_index(drop=True)
    blocks, Xgeom = gs.load_instance_features(inst)
    I_eid = inst.episode_id.astype(str).to_numpy(); I_sid = inst.subject_id.to_numpy(int)
    n_per = inst.groupby("episode_id").size().to_dict(); Iw = np.array([1.0 / n_per[e] for e in I_eid])

    # EHR
    ehr = pd.read_csv(os.path.join(PC, "ehr_features_episode.csv")); ehr["episode_id"] = ehr.episode_id.astype(str)
    em = ehr.set_index("episode_id")[EHR_COLS]; X_ehr = np.full((len(eids), len(EHR_COLS)), np.nan, np.float32)
    for e in em.index:
        if e in row_of: X_ehr[row_of[e]] = em.loc[e].to_numpy(np.float32)

    # ECG embedding (deployed 18k arm)
    ew = os.path.join(ROOT, "outputs", os.environ.get("ECG_DIR", "ecg_waveform_episode"))
    E = np.load(os.path.join(ew, "ecg_embeddings.npy")); eix = pd.read_csv(os.path.join(ew, "ecg_embedding_index.csv"))
    X_ecg = np.full((len(eids), E.shape[1]), np.nan, np.float32)
    for r, e in zip(E, eix.episode_id.astype(str)):
        if e in row_of: X_ecg[row_of[e]] = r

    # fine-tuned CXR scalar (root/asc only)
    ft_map = {}
    ft = pd.read_csv(os.path.join(ROOT, "outputs", "cxr_finetune_episode", "oof_predictions.csv"))
    ft["episode_id"] = ft.episode_id.astype(str)
    for s in ("root", "asc"):
        fm = dict(zip(ft[ft.site == s].episode_id, ft[ft.site == s].pred_cxr_ft))
        ft_map[s] = np.array([fm.get(x, np.nan) for x in eids])
    log.info("episodes %d | CXR inst %d | ECG-emb %d", len(eids), len(inst), int((~np.isnan(X_ecg[:, 0])).sum()))

    folds = load_episode_folds(os.path.join(PC, "episode_fold_assignments.csv"))
    results = {"fold_mode": "immutable", "sites": {}}; oof_rows = []
    for site in ("root", "asc", "arch", "desc"):
        d = diam[site]
        d_cxr = gs.cxr_base_oof(folds, d, blocks, Xgeom, I_eid, I_sid, Iw, row_of)
        d_ehr = ehr_floor_oof(folds, d, X_ehr, row_of)
        d_fin = linear_early_oof(folds, d, d_cxr, X_ehr, X_ecg, row_of, sid, d_cxr_ft=ft_map.get(site))
        y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        m = ~np.isnan(d) & ~np.isnan(d_fin) & ~np.isnan(d_ehr); g = sid[m]
        pos = int(np.nansum(y40[m]))
        sr = {
            "n": int(m.sum()), "pos_ge40": pos, "median_cm": round(float(np.nanmedian(d[m])), 2),
            "floor_r2": fmt(cluster_bootstrap_ci(d[m], d_ehr[m], g, r2, need_both_classes=False)),
            "final_r2": fmt(cluster_bootstrap_ci(d[m], d_fin[m], g, r2, need_both_classes=False)),
            "final_mae": fmt(cluster_bootstrap_ci(d[m], d_fin[m], g, mae, need_both_classes=False)),
            "floor_mae": fmt(cluster_bootstrap_ci(d[m], d_ehr[m], g, mae, need_both_classes=False)),
            "final_vs_floor_r2": fmt(paired_cluster_bootstrap_diff(d[m], d_fin[m], d_ehr[m], g, r2, need_both_classes=False)),
            "final_ge40": fmt(cluster_bootstrap_ci(y40[m], d_fin[m], g, auroc)) if pos >= 20 else "NA (few pos)",
            "floor_ge40": fmt(cluster_bootstrap_ci(y40[m], d_ehr[m], g, auroc)) if pos >= 20 else "NA (few pos)",
            "used_cxr_ft": site in ("root", "asc"),
        }
        results["sites"][site] = sr
        log.info("[%-5s n=%d pos>=40=%d med=%.2f] FINAL R2 %s MAE %s | floor R2 %s | ge40 %s vs floor %s | vs-floor R2 %s",
                 site, sr["n"], pos, sr["median_cm"], sr["final_r2"], sr["final_mae"], sr["floor_r2"],
                 sr["final_ge40"], sr["floor_ge40"], sr["final_vs_floor_r2"])
        for i in np.where(m)[0]:
            oof_rows.append({"episode_id": eids[i], "subject_id": int(sid[i]), "site": site,
                             "diam_true": d[i], "pred_final": d_fin[i], "pred_floor": d_ehr[i]})

    out_dir = os.path.join(ROOT, "outputs", "final_model_allsegments"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(oof_rows).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
