"""
Does the ECG add anything beyond CXR + EHR, at episode scale?

The project's most interesting prior finding (v7, n=522) was that the ECG is
essentially redundant with EHR (body size) + CXR (cardiac silhouette). This re-runs
that test on the ~18,100 episodes with a pre-index ECG, so a null result carries real
weight instead of resting on 522 patients.

Protocol (identical machinery to the headline geometry stack, reused by import):
  base learners (nested OOF, patient-grouped folds, per-fold PCA on CXR):
    d_cxr  multi-instance RAD-DINO patchpool + geometry -> HGB diameter regressor
    d_ehr  episode-level causal EHR diameter OOF (the floor)
    d_ecg  HGB diameter regressor on the 33 per-episode ECG measurement features
  compared, on the SAME ECG-having episodes, both scored by predicted diameter:
    stack_2 = ridge([d_cxr, d_ehr])            (current headline model)
    stack_3 = ridge([d_cxr, d_ehr, d_ecg])     (+ ECG)
  paired cluster-bootstrap( stack_3 - stack_2 ) is the ECG increment. Also reports
  ECG-ALONE performance (is the ECG predictive of aortic size at all?).

Run: sbatch scripts/slurm_ecg_increment_episode.sh   (SEEDS in the wrapper)
Out: outputs/ecg_increment_episode/results.json (+ oof_predictions.csv)
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)

SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3,4,5").split(",")]

# Reuse the exact CXR base learner + ridge stack from the headline model.
import train_geometry_stack_episode as gs   # noqa: E402


def ecg_base_oof(folds, y_row, X_ecg, has_ecg, row_of, eids):
    """Nested-OOF HGB diameter regressor on per-episode ECG features.
    Only fits/predicts episodes that actually have an ECG; others stay NaN."""
    from sklearn.base import clone
    out = np.full(len(y_row), np.nan)
    base = gs._hgb_reg()
    for tr_eids, te_eids in folds:
        trs, tes = set(map(str, tr_eids)), set(map(str, te_eids))
        tr = [row_of[e] for e in eids if e in trs and has_ecg[row_of[e]]
              and not np.isnan(y_row[row_of[e]])]
        te = [row_of[e] for e in eids if e in tes and has_ecg[row_of[e]]]
        if len(tr) < 20 or not te:
            continue
        m = clone(base).fit(X_ecg[tr], y_row[tr])
        out[te] = m.predict(X_ecg[te])
    return out


def main():
    from multimodal_aorta.data.episodes import load_episodes, make_episode_id
    from multimodal_aorta.data.splits import make_grouped_cv_folds
    from multimodal_aorta.training.bootstrap import (
        cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt)

    ep = load_episodes(PC, require_ecg=False)
    eids = ep["episode_id"].astype(str).tolist()
    row_of = {e: i for i, e in enumerate(eids)}
    sid_of_row = ep["subject_id"].to_numpy(int)
    diam = {"root": ep["target_root"].to_numpy(float), "asc": ep["target_asc"].to_numpy(float)}

    # --- CXR instances + features (reuse headline loader) ---
    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    inst = inst[inst.episode_id.isin(set(eids))].reset_index(drop=True)
    import torch
    pp = torch.load(os.path.join(PC, "raddino_patchpool_embeddings_episode.pt"),
                    map_location="cpu", weights_only=False)
    inst = inst[inst.dicom_id.astype(str).isin(set(map(str, pp.keys())))].reset_index(drop=True)
    blocks, Xgeom = gs.load_instance_features(inst)
    I_eid = inst.episode_id.astype(str).to_numpy()
    I_sid = inst.subject_id.to_numpy(int)
    n_per = inst.groupby("episode_id").size().to_dict()
    Iw = np.array([1.0 / n_per[e] for e in I_eid])

    # --- EHR floor OOF diameter (per episode) ---
    ehr = pd.read_csv(os.path.join(ROOT, "outputs", "ehr_baseline_episode", "oof_predictions.csv"))
    ehr["episode_id"] = ehr["episode_id"].astype(str)
    def ehr_diam(site):
        e = ehr[(ehr.site == site) & (ehr.endpoint == "diam")]
        m = dict(zip(e.episode_id, e.pred_value))
        return np.array([m.get(x, np.nan) for x in eids])

    # --- ECG features per episode ---
    ecg = pd.read_csv(os.path.join(PC, "ecg_features_episode.csv"))
    ecg["episode_id"] = ecg["episode_id"].astype(str)
    ecg_cols = [c for c in ecg.columns if c.startswith("ecg_")]
    ecg_map = ecg.set_index("episode_id")[ecg_cols]
    X_ecg = np.full((len(eids), len(ecg_cols)), np.nan, dtype=np.float32)
    has_ecg = np.zeros(len(eids), bool)
    for e in ecg_map.index:
        if e in row_of:
            X_ecg[row_of[e]] = ecg_map.loc[e].to_numpy(np.float32)
            has_ecg[row_of[e]] = True
    log.info("episodes %d | with ECG %d (%.1f%%) | ECG features %d",
             len(eids), int(has_ecg.sum()), 100 * has_ecg.mean(), len(ecg_cols))

    results = {"seeds": SEEDS, "n_episodes": len(eids), "n_ecg": int(has_ecg.sum()), "sites": {}}
    oof_rows = []
    for site in ("root", "asc"):
        d = diam[site]; d_ehr = ehr_diam(site)
        y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        y45 = np.where(np.isnan(d), np.nan, (d >= 4.5).astype(float))
        S2, S3, ECG = [], [], []
        for seed in SEEDS:
            folds = make_grouped_cv_folds(ep, stratify_col="anyAD", n_splits=5, seed=seed)
            d_cxr = gs.cxr_base_oof(folds, d, blocks, Xgeom, I_eid, I_sid, Iw, row_of)
            d_ecg = ecg_base_oof(folds, d, X_ecg, has_ecg, row_of, eids)
            s2 = gs.ridge_stack(np.column_stack([d_cxr, d_ehr]), d, folds, row_of)
            s3 = gs.ridge_stack(np.column_stack([d_cxr, d_ehr, d_ecg]), d, folds, row_of)
            S2.append(s2); S3.append(s3); ECG.append(d_ecg)
        s2 = np.nanmean(np.column_stack(S2), axis=1)
        s3 = np.nanmean(np.column_stack(S3), axis=1)
        d_ecg = np.nanmean(np.column_stack(ECG), axis=1)
        # Evaluate everything on the SAME episodes: ECG-having AND scored by both stacks.
        m = has_ecg & ~np.isnan(s3) & ~np.isnan(s2)
        g = sid_of_row[m]
        def M(y, p, fn, nb=True):
            k = m & ~np.isnan(y) & ~np.isnan(p)
            return fmt(cluster_bootstrap_ci(y[k], p[k], sid_of_row[k], fn, need_both_classes=nb))
        sr = {
            "n_eval": int(m.sum()),
            "ecg_alone_ge40": M(y40, d_ecg, auroc),
            "ecg_alone_r2": M(d, d_ecg, r2, nb=False),
            "stack2_ge40": M(y40, s2, auroc), "stack3_ge40": M(y40, s3, auroc),
            "stack2_r2": M(d, s2, r2, nb=False), "stack3_r2": M(d, s3, r2, nb=False),
            # paired ECG increment (stack3 - stack2) on the same episodes
            "incr_ge40": fmt(paired_cluster_bootstrap_diff(y40[m], s3[m], s2[m], g, auroc)),
            "incr_ge45": fmt(paired_cluster_bootstrap_diff(y45[m], s3[m], s2[m], g, auroc)),
            "incr_r2": fmt(paired_cluster_bootstrap_diff(d[m], s3[m], s2[m], g, r2, need_both_classes=False)),
        }
        results["sites"][site] = sr
        log.info("[%s] ECG-alone ge40 %s R2 %s", site, sr["ecg_alone_ge40"], sr["ecg_alone_r2"])
        log.info("[%s] stack CXR+EHR ge40 %s -> +ECG %s", site, sr["stack2_ge40"], sr["stack3_ge40"])
        log.info("[%s] ECG INCREMENT ge40 %s | ge45 %s | R2 %s",
                 site, sr["incr_ge40"], sr["incr_ge45"], sr["incr_r2"])
        for i, e in enumerate(eids):
            if m[i]:
                oof_rows.append({"episode_id": e, "subject_id": int(sid_of_row[i]), "site": site,
                                 "diam_true": d[i], "pred_cxr_ehr": s2[i],
                                 "pred_cxr_ehr_ecg": s3[i], "pred_ecg_alone": d_ecg[i]})

    out_dir = os.path.join(ROOT, "outputs", "ecg_increment_episode"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(oof_rows).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
