"""
Review issue A9, roadmap experiment C1 step 1: does the RAD-DINO checkpoint's own
grayscale normalization beat the legacy ImageNet per-channel normalization?

One factor differs between the two runs (normalization + bicubic); geometry, episodes,
folds, seed and fine-tuning depth are identical. Compares the two fine-tuned CXR arms
on the episodes they share, with paired patient-clustered bootstrap differences.

PRESPECIFIED (recorded in notes/repairs_2026-09-14.md before the result was known):
a difference smaller than ~0.01 R^2 is INCONCLUSIVE at a single seed, not a win.

Run: python scripts/compare_cxr_preproc.py
Out: outputs/cxr_preproc_comparison/results.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

INCONCLUSIVE_R2 = 0.01          # prespecified
A_DIR = os.environ.get("BASE_DIR", "cxr_finetune_episode")             # legacy
B_DIR = os.environ.get("NEW_DIR", "cxr_finetune_episode_ckpt_norm")    # ckpt_norm


def main():
    from multimodal_aorta.training.bootstrap import (
        cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt)
    from multimodal_aorta.data.episodes import load_episodes

    ep = load_episodes(os.path.join(ROOT, "pretrained_checkpoints"), require_ecg=False)
    sid_of = dict(zip(ep.episode_id.astype(str), ep.subject_id))
    truth = {"root": dict(zip(ep.episode_id.astype(str), ep.target_root)),
             "asc": dict(zip(ep.episode_id.astype(str), ep.target_asc))}

    def load(d):
        p = os.path.join(ROOT, "outputs", d, "oof_predictions.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        t = pd.read_csv(p); t["episode_id"] = t.episode_id.astype(str)
        return t

    A, B = load(A_DIR), load(B_DIR)
    results = {"baseline_dir": A_DIR, "new_dir": B_DIR,
               "inconclusive_band_r2": INCONCLUSIVE_R2, "sites": {}}

    for site in ("root", "asc"):
        a = A[A.site == site][["episode_id", "pred_cxr_ft"]].rename(columns={"pred_cxr_ft": "legacy"})
        b = B[B.site == site][["episode_id", "pred_cxr_ft"]].rename(columns={"pred_cxr_ft": "ckpt"})
        m = a.merge(b, on="episode_id", how="inner")
        m["d"] = m.episode_id.map(truth[site]); m["g"] = m.episode_id.map(sid_of)
        m = m.dropna()
        d = m.d.to_numpy(float); g = m.g.to_numpy(int); y40 = (d >= 4.0).astype(float)
        pl, pc = m.legacy.to_numpy(float), m.ckpt.to_numpy(float)
        log.info("[%s] matched episodes %d | patients %d | >=4cm %d",
                 site, len(m), len(np.unique(g)), int(y40.sum()))

        sr = {"n": len(m), "n_pos": int(y40.sum()),
              "legacy_r2": fmt(cluster_bootstrap_ci(d, pl, g, r2, need_both_classes=False)),
              "ckpt_norm_r2": fmt(cluster_bootstrap_ci(d, pc, g, r2, need_both_classes=False)),
              "legacy_ge40": fmt(cluster_bootstrap_ci(y40, pl, g, auroc)),
              "ckpt_norm_ge40": fmt(cluster_bootstrap_ci(y40, pc, g, auroc)),
              "d_r2": fmt(paired_cluster_bootstrap_diff(d, pc, pl, g, r2, need_both_classes=False)),
              "d_ge40": fmt(paired_cluster_bootstrap_diff(y40, pc, pl, g, auroc))}
        point = float(r2(d, pc)) - float(r2(d, pl))
        sr["d_r2_point"] = point
        sr["verdict"] = ("INCONCLUSIVE (within the prespecified band at a single seed)"
                         if abs(point) < INCONCLUSIVE_R2 else
                         ("ckpt_norm better" if point > 0 else "legacy better"))
        results["sites"][site] = sr
        log.info("   legacy    R2 %s | AUROC %s", sr["legacy_r2"], sr["legacy_ge40"])
        log.info("   ckpt_norm R2 %s | AUROC %s", sr["ckpt_norm_r2"], sr["ckpt_norm_ge40"])
        log.info("   paired dR2 %s | dAUROC %s -> %s", sr["d_r2"], sr["d_ge40"], sr["verdict"])

    out = os.path.join(ROOT, "outputs", "cxr_preproc_comparison"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
