"""
Review issue A7: the multimodal model must beat the STRONGEST unimodal comparator,
not just EHR or the older frozen-CXR stack.

Compares, on the identical nested OOF predictions and identical episodes:
    fine-tuned CXR alone   (the strongest current image-only arm)
    frozen multi-instance CXR alone
    the full nested multimodal model

with paired patient-clustered bootstrap differences. A negative result here is a
legitimate finding; an overstated positive one is not.

Run: python scripts/compare_a7_baselines.py
Out: outputs/a7_comparison/results.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
NESTED = os.environ.get("NESTED_DIR", "final_model_nested_full")


def main():
    from multimodal_aorta.training.bootstrap import (
        cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt)

    nes = pd.read_csv(os.path.join(ROOT, "outputs", NESTED, "oof_predictions.csv"))
    nes["episode_id"] = nes.episode_id.astype(str)
    ft = pd.read_csv(os.path.join(ROOT, "outputs", "cxr_finetune_episode", "oof_predictions.csv"))
    ft["episode_id"] = ft.episode_id.astype(str)

    results = {"nested_dir": NESTED, "sites": {}}
    for site in ("root", "asc"):
        a = nes[nes.site == site][["episode_id", "subject_id", "diam_true",
                                   "pred_final", "pred_cxr_only"]]
        b = ft[ft.site == site][["episode_id", "pred_cxr_ft"]]
        m = a.merge(b, on="episode_id", how="inner").dropna()
        d = m.diam_true.to_numpy(float); g = m.subject_id.to_numpy(int)
        y40 = (d >= 4.0).astype(float)
        P = {"full": m.pred_final.to_numpy(float),
             "cxr_frozen": m.pred_cxr_only.to_numpy(float),
             "cxr_finetuned": m.pred_cxr_ft.to_numpy(float)}
        log.info("[%s] matched episodes %d | patients %d | >=4cm %d",
                 site, len(m), len(np.unique(g)), int(y40.sum()))

        sr = {"n": len(m), "n_patients": int(len(np.unique(g))), "n_pos": int(y40.sum()),
              "arms": {}, "paired_vs_finetuned_cxr": {}}
        for k, p in P.items():
            sr["arms"][k] = {
                "r2": fmt(cluster_bootstrap_ci(d, p, g, r2, need_both_classes=False)),
                "auroc_ge40": fmt(cluster_bootstrap_ci(y40, p, g, auroc))}
            log.info("   %-14s R2 %s | AUROC %s", k, sr["arms"][k]["r2"], sr["arms"][k]["auroc_ge40"])
        for k in ("full", "cxr_frozen"):
            sr["paired_vs_finetuned_cxr"][k] = {
                "d_r2": fmt(paired_cluster_bootstrap_diff(d, P[k], P["cxr_finetuned"], g, r2, need_both_classes=False)),
                "d_auroc_ge40": fmt(paired_cluster_bootstrap_diff(y40, P[k], P["cxr_finetuned"], g, auroc))}
            log.info("   %s - finetuned_cxr: dR2 %s | dAUROC %s", k,
                     sr["paired_vs_finetuned_cxr"][k]["d_r2"],
                     sr["paired_vs_finetuned_cxr"][k]["d_auroc_ge40"])
        results["sites"][site] = sr

    out = os.path.join(ROOT, "outputs", "a7_comparison"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
