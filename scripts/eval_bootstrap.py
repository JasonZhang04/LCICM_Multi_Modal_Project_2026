"""
v3 Milestone 5 — eval polish: bootstrap-CI comparison across approaches.

Loads the OOF prediction dumps each model saved (outputs/<approach>/oof.npz) and
prints AUROC/AUPRC (classification) and MAE/R^2 (regression) with 95% percentile
bootstrap CIs, so GBDT vs deep vs late-fusion is comparable with uncertainty.

Pure consumer of saved OOF arrays — does not retrain anything.

Run: python scripts/eval_bootstrap.py
"""

import os
import sys
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

APPROACHES = [
    ("GBDT",        "outputs/gbdt_fusion/oof.npz"),
    ("Deep",        "outputs/deep_fusion/oof.npz"),
    ("LateFusion",  "outputs/late_fusion/oof.npz"),
]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.training.bootstrap import bootstrap_ci, auroc, auprc, mae, r2, fmt

    loaded = {}
    for name, rel in APPROACHES:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            loaded[name] = dict(np.load(p))
        else:
            log.info("(skip %s — %s not found)", name, rel)

    if not loaded:
        log.info("No OOF dumps found. Run the training scripts first.")
        return

    def row(site, tname, kind):
        key_y, key_p = f"{site}_{tname}_y", f"{site}_{tname}_p"
        log.info("\n=== %s / %s ===", site.upper(), tname)
        for name, d in loaded.items():
            if key_y not in d or key_p not in d:
                continue
            y, p = d[key_y], d[key_p]
            if kind == "clf":
                a = fmt(bootstrap_ci(y, p, auroc))
                ap = fmt(bootstrap_ci(y, p, auprc))
                log.info("  %-11s AUROC=%-22s AUPRC=%s", name, a, ap)
            else:
                m = fmt(bootstrap_ci(y, p, mae, need_both_classes=False))
                rr = fmt(bootstrap_ci(y, p, r2, need_both_classes=False))
                log.info("  %-11s MAE=%-22s R2=%s", name, m, rr)

    log.info("Bootstrap 95%% CIs (2000 resamples). [point, lo, hi]")
    for site in ("root", "asc"):
        row(site, "ge40", "clf")   # anyAD >= 4.0 (primary)
        row(site, "ge45", "clf")   # moderate+ >= 4.5
        row(site, "diam", "reg")   # diameter regression


if __name__ == "__main__":
    main()
