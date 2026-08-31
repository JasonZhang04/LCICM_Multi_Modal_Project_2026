"""
Is the PCA-32 on the frozen CXR embeddings helping or hurting at episode scale?

At n=522 PCA-32 was the biggest cheap win (overfitting control). At ~16k episodes the
overfitting pressure is far lower, so PCA-32 might now be an information bottleneck. This
sweeps the CXR base learner's out-of-fold diameter R^2 over PCA dimensionality
(16 / 32 / 64 / 128 / 768=full-rank rotation, not raw features) with everything else fixed, single seed.

Reuses train_geometry_stack_episode.cxr_base_oof (the multi-instance CXR diameter
regressor); gs.K is overridden per sweep value.

Run: sbatch scripts/slurm_pca_sweep_episode.sh   Out: outputs/pca_sweep_episode/results.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_geometry_stack_episode as gs
KS = [int(x) for x in os.environ.get("KS", "16,32,64,128,768").split(",")]
SEED = int(os.environ.get("SEED", "42"))


def main():
    from multimodal_aorta.data.episodes import load_episodes, make_episode_id
    from multimodal_aorta.data.splits import make_grouped_cv_folds
    from multimodal_aorta.training.bootstrap import (
        cluster_bootstrap_ci, paired_cluster_bootstrap_diff, r2, auroc, fmt)
    import torch

    ep = load_episodes(PC, require_ecg=False)
    eids = ep.episode_id.astype(str).tolist(); row_of = {e: i for i, e in enumerate(eids)}
    sid = ep.subject_id.to_numpy(int)
    diam = {"root": ep.target_root.to_numpy(float), "asc": ep.target_asc.to_numpy(float)}
    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    inst = inst[inst.episode_id.isin(set(eids))].reset_index(drop=True)
    pp = torch.load(os.path.join(PC, "raddino_patchpool_embeddings_episode.pt"), map_location="cpu", weights_only=False)
    inst = inst[inst.dicom_id.astype(str).isin(set(map(str, pp.keys())))].reset_index(drop=True)
    blocks, Xgeom = gs.load_instance_features(inst)
    I_eid = inst.episode_id.astype(str).to_numpy(); I_sid = inst.subject_id.to_numpy(int)
    n_per = inst.groupby("episode_id").size().to_dict(); Iw = np.array([1.0 / n_per[e] for e in I_eid])
    folds = make_grouped_cv_folds(ep, stratify_col="anyAD", n_splits=5, seed=SEED)

    results = {"seed": SEED, "ks": KS, "sites": {}}
    for site in ("root", "asc"):
        d = diam[site]; y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        results["sites"][site] = {}
        preds = {}
        for k in KS:
            gs.K = k                                    # override PCA dim used inside cxr_base_oof
            d_cxr = gs.cxr_base_oof(folds, d, blocks, Xgeom, I_eid, I_sid, Iw, row_of)
            preds[k] = d_cxr
            m = ~np.isnan(d) & ~np.isnan(d_cxr); g = sid[m]
            r2c = fmt(cluster_bootstrap_ci(d[m], d_cxr[m], g, r2, need_both_classes=False))
            auc = fmt(cluster_bootstrap_ci(y40[m], d_cxr[m], g, auroc))
            results["sites"][site][str(k)] = {"cxr_r2": r2c, "cxr_ge40": auc}
            log.info("[%s] PCA K=%3d -> CXR-alone R2 %s | ge40 %s", site, k, r2c, auc)
        # PAIRED delta-R2 (same episodes/folds, differ only by K) — the correct test for
        # whether K=128 really beats K=32 (marginal CIs overlap; paired is far tighter).
        if 128 in preds and 32 in preds:
            m = ~np.isnan(d) & ~np.isnan(preds[128]) & ~np.isnan(preds[32])
            dd = fmt(paired_cluster_bootstrap_diff(d[m], preds[128][m], preds[32][m], sid[m], r2, need_both_classes=False))
            results["sites"][site]["r2_128_vs_32_paired"] = dd
            log.info("[%s] PAIRED delta-R2 (K=128 - K=32): %s", site, dd)
    out_dir = os.path.join(ROOT, "outputs", "pca_sweep_episode"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
