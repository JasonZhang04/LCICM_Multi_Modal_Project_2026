"""
Clinical evaluation of the episode-level models from their saved OOF predictions.

AUROC alone is optimistic at ~4.5% prevalence (see PROGRESS_UPDATE_2026-08-04 §7.2),
so this reports the full operating picture the PI asked for: confusion matrices with
precision (PPV), recall (sensitivity), specificity, NPV, F1 and number-needed-to-echo,
at several clinically meaningful thresholds, plus AUPRC and diameter Bland-Altman.

Reads outputs/<model_dir>/oof_predictions.csv (columns: episode_id, subject_id, site,
diam_true, pred_stack, pred_floor) — the geometry-stack run saves exactly this. Each
`pred_*` is a predicted diameter (cm); binary endpoints are scored by ranking it, so
the operating points are thresholds ON the predicted diameter.

Operating points reported per (site, endpoint, model):
  - dx>=cut : flag if PREDICTED diameter >= the clinical cutoff (directly interpretable)
  - sens90  : highest threshold with recall >= 0.90 (screening-friendly)
  - top10 / top5 : refer the highest-scoring 10% / 5% to echo (capacity-limited screen)

Metrics with CIs use the patient-clustered bootstrap.

Run: python scripts/eval_episode.py [--model-dir outputs/geometry_stack_episode]
Out: <model_dir>/clinical_metrics.json  (+ a printed table)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

CUTS = {"ge40": 4.0, "ge45": 4.5}


def _f1(ppv, sens):
    if np.isnan(ppv) or np.isnan(sens) or (ppv + sens) == 0:
        return float("nan")
    return 2 * ppv * sens / (ppv + sens)


def bland_altman(y, p):
    """Agreement of predicted vs true diameter (cm): bias and 95% limits."""
    m = ~np.isnan(y) & ~np.isnan(p)
    d = p[m] - y[m]
    bias = float(np.mean(d)); sd = float(np.std(d, ddof=1))
    return {"bias": bias, "sd": sd,
            "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd,
            "mae": float(np.mean(np.abs(d))), "n": int(m.sum())}


def main():
    from multimodal_aorta.training.clinical_metrics import (
        _confusion, metrics_at_topk, threshold_for_target)
    from multimodal_aorta.training.bootstrap import (
        cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, auprc, r2, fmt)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "outputs", "geometry_stack_episode"))
    ap.add_argument("--pred-col", default="pred_stack", help="score column for the model of interest")
    ap.add_argument("--floor-col", default="pred_floor", help="score column for the comparison floor")
    a = ap.parse_args()

    oof = pd.read_csv(os.path.join(a.model_dir, "oof_predictions.csv"))
    out = {"model_dir": a.model_dir, "pred_col": a.pred_col, "sites": {}}

    def confusion_row(y, score, tag, **op):
        flag = op.pop("_flag")            # ndarray — must not land in the JSON dict
        c = dict(op); c.update(_confusion(y, flag))
        c["f1"] = _f1(c["ppv"], c["sens"]); c["op"] = tag
        return c

    for site in ("root", "asc"):
        s = oof[oof.site == site]
        d = s.diam_true.to_numpy(float)
        groups = s.subject_id.to_numpy(int)
        p = s[a.pred_col].to_numpy(float)
        pf = s[a.floor_col].to_numpy(float)
        site_out = {"n_episodes": int(len(s)), "n_patients": int(s.subject_id.nunique()),
                    "diameter": {}, "endpoints": {}}

        # ---- diameter regression ----
        site_out["diameter"] = {
            "r2": fmt(cluster_bootstrap_ci(d, p, groups, r2, need_both_classes=False)),
            "r2_vs_floor": fmt(paired_cluster_bootstrap_diff(
                d, p, pf, groups, r2, need_both_classes=False)),
            "bland_altman": bland_altman(d, p),
        }

        # ---- binary endpoints ----
        for ep, cut in CUTS.items():
            y = np.where(np.isnan(d), np.nan, (d >= cut).astype(float))
            m = ~np.isnan(y) & ~np.isnan(p)
            yy, pp, gg = y[m], p[m], groups[m]
            npos = int(np.nansum(yy))
            row = {
                "cutoff_cm": cut, "n_pos": npos, "prevalence": npos / len(yy) if len(yy) else float("nan"),
                "auroc": fmt(cluster_bootstrap_ci(yy, pp, gg, auroc)),
                "auprc": fmt(cluster_bootstrap_ci(yy, pp, gg, auprc)),
                "auroc_vs_floor": fmt(paired_cluster_bootstrap_diff(yy, pp, pf[m], gg, auroc)),
                "auprc_vs_floor": fmt(paired_cluster_bootstrap_diff(yy, pp, pf[m], gg, auprc)),
                "confusion": [],
            }
            # operating points (thresholds on the predicted diameter). Note: the raw
            # clinical cutoff is useless here — regression-derived scores shrink toward
            # the mean, so almost nobody is *predicted* >= 4.0/4.5; the model is used by
            # RANKING. Youden's J is the principled single balanced threshold.
            ops = []
            cand = np.unique(pp)
            if len(cand) > 1:
                j = [(_confusion(yy, pp >= t)["sens"] + _confusion(yy, pp >= t)["spec"] - 1, t)
                     for t in cand[::max(1, len(cand) // 400)]]
                jt = max(j, key=lambda z: (z[0] if not np.isnan(z[0]) else -1))[1]
                ops.append(("youdenJ", confusion_row(yy, pp, "youdenJ", _flag=(pp >= jt),
                                                      threshold=float(jt))))
            t90 = threshold_for_target(yy, pp, 0.90, "sensitivity")
            if not np.isnan(t90.get("sens", float("nan"))):
                t90["f1"] = _f1(t90["ppv"], t90["sens"]); t90["op"] = "sens>=0.90"
                ops.append(("sens90", t90))
            for frac, tag in [(0.10, "top10"), (0.05, "top5")]:
                c = metrics_at_topk(yy, pp, frac); c["f1"] = _f1(c["ppv"], c["sens"]); c["op"] = tag
                ops.append((tag, c))
            row["confusion"] = [c for _, c in ops]
            site_out["endpoints"][ep] = row
        out["sites"][site] = site_out

    def _js(o):   # numpy-safe fallback for anything that slips through
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o)}")

    # ---- printed table (before dump, so it shows even if serialization complains) ----
    print(f"\n=== Clinical evaluation: {a.model_dir}  (score = {a.pred_col}) ===")
    for site in ("root", "asc"):
        so = out["sites"][site]
        print(f"\n########## {site.upper()}  (n={so['n_episodes']} ep / {so['n_patients']} pt) ##########")
        ba = so["diameter"]["bland_altman"]
        print(f"  diameter: R2 {so['diameter']['r2']}  (vs floor {so['diameter']['r2_vs_floor']})")
        print(f"            Bland-Altman bias {ba['bias']:+.3f} cm, 95% LoA [{ba['loa_lower']:+.2f},"
              f" {ba['loa_upper']:+.2f}], MAE {ba['mae']:.3f}")
        for ep, r in so["endpoints"].items():
            print(f"\n  -- {site} {ep} (>= {r['cutoff_cm']} cm)  pos={r['n_pos']} "
                  f"prev={r['prevalence']*100:.1f}%")
            print(f"     AUROC {r['auroc']} (vs floor {r['auroc_vs_floor']})")
            print(f"     AUPRC {r['auprc']} (vs floor {r['auprc_vs_floor']})")
            print(f"     {'op':10s} {'TP':>5} {'FP':>6} {'FN':>5} {'TN':>6}  "
                  f"{'recall':>6} {'prec':>6} {'spec':>6} {'F1':>5} {'NNE':>5}")
            for c in r["confusion"]:
                print(f"     {c['op']:10s} {c['tp']:5d} {c['fp']:6d} {c['fn']:5d} {c['tn']:6d}  "
                      f"{c['sens']:6.3f} {c['ppv']:6.3f} {c['spec']:6.3f} "
                      f"{c['f1']:5.2f} {c['nne']:5.1f}")
    with open(os.path.join(a.model_dir, "clinical_metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=_js)
    print(f"\nSaved -> {a.model_dir}/clinical_metrics.json")


if __name__ == "__main__":
    main()
