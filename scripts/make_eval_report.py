"""
M2 — unified evaluation report over all standardized OOF predictions.

Reads every outputs/<model>/oof_predictions.csv (the M0 schema) and produces one
comparison, aligned patient-for-patient, in outputs/eval_report/:

  metrics_summary.csv    AUROC/AUPRC (+95% CI), prevalence, Brier, calibration
                         slope/intercept per (model, site, binary endpoint);
                         MAE/R2 (+CI) per regression endpoint.
  deltas_vs_ehr.csv      PAIRED bootstrap difference of every fusion/echo model
                         MINUS the EHR floor (ehr_baseline), per endpoint/metric,
                         with 95% CI. CI excluding 0 => distinguishable.
  clinical_utility.csv   sensitivity/PPV/number-needed-to-echo at top 5/10/20%
                         risk, and operating points at 90% sens / 90% spec.
  calibration_metrics.csv  Brier + calibration slope/intercept per binary endpoint.
  event_counts.csv       per-fold and overall positives for each endpoint,
                         including ge50 marked not-estimable (0 positives).
  plots/calibration_grid.png, plots/model_comparison_auroc.png

The EHR baseline is the reference "floor"; the tiered comparison asks whether the
multimodal star (and later the echo ceiling) add value beyond it.

Run via SLURM: sbatch scripts/slurm_eval_report.sh
"""

import os
import sys
import glob
import json
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FLOOR = "ehr_baseline"
BIN_ENDPOINTS = ["ge40", "ge45"]
MODEL_ORDER = ["ehr_baseline", "gbdt_concat", "gbdt_fusion", "deep_fusion",
               "late_fusion_meta", "late_fusion", "echo_panecho"]


def _order_key(m):
    return MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)


def load_all_oof(out_root):
    frames = []
    for p in sorted(glob.glob(os.path.join(out_root, "*", "oof_predictions.csv"))):
        d = pd.read_csv(p)
        d["_source"] = os.path.basename(os.path.dirname(p))
        frames.append(d)
    if not frames:
        raise SystemExit(f"no oof_predictions.csv found under {out_root}")
    df = pd.concat(frames, ignore_index=True)
    df["subject_id"] = df["subject_id"].astype(int)
    return df


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.training.bootstrap import (
        bootstrap_ci, paired_bootstrap_diff, auroc, auprc, mae, r2)
    from multimodal_aorta.training import clinical_metrics as cm

    out_root = os.path.join(root, "outputs")
    rep_dir = os.path.join(out_root, "eval_report"); os.makedirs(rep_dir, exist_ok=True)
    plot_dir = os.path.join(rep_dir, "plots"); os.makedirs(plot_dir, exist_ok=True)

    df = load_all_oof(out_root)
    models = sorted(df["model_name"].unique(), key=_order_key)
    log.info("Models found: %s", models)

    def slab(model, site, endpoint):
        s = df[(df.model_name == model) & (df.site == site) & (df.endpoint == endpoint)]
        return s.sort_values("subject_id")

    # ---------- summary + calibration + utility ----------
    summary, calib, util = [], [], []
    for model in models:
        for site in ("root", "asc"):
            for ep in BIN_ENDPOINTS:
                s = slab(model, site, ep)
                if s.empty:
                    continue
                y, p = s.y_true.to_numpy(float), s.pred_prob.to_numpy(float)
                m = ~(np.isnan(y) | np.isnan(p))
                yv, pv = y[m], p[m]
                pos = int(np.nansum(yv))
                a = bootstrap_ci(y, p, auroc)
                ap = bootstrap_ci(y, p, auprc)
                ci = cm.calibration_slope_intercept(y, p)
                summary.append({
                    "model": model, "site": site, "endpoint": ep, "target": "binary",
                    "n": int(m.sum()), "pos": pos, "prevalence": round(pos / max(m.sum(), 1), 4),
                    "auroc": round(a[0], 4), "auroc_lo": round(a[1], 4), "auroc_hi": round(a[2], 4),
                    "auprc": round(ap[0], 4), "auprc_lo": round(ap[1], 4), "auprc_hi": round(ap[2], 4),
                })
                calib.append({
                    "model": model, "site": site, "endpoint": ep,
                    "brier": round(cm.brier_score(y, p), 4),
                    "cal_slope": round(ci["slope"], 4), "cal_intercept": round(ci["intercept"], 4),
                })
                for frac in (0.05, 0.10, 0.20):
                    u = cm.metrics_at_topk(y, p, frac)
                    util.append({"model": model, "site": site, "endpoint": ep,
                                 "operating_point": f"top{int(frac*100)}pct",
                                 "n_flagged": u["n_flagged"], "sens": round(u["sens"], 4),
                                 "ppv": round(u["ppv"], 4), "nne": round(u["nne"], 3)})
                for tgt, mode in [(0.90, "sensitivity"), (0.90, "specificity")]:
                    o = cm.threshold_for_target(y, p, tgt, mode)
                    util.append({"model": model, "site": site, "endpoint": ep,
                                 "operating_point": f"{int(tgt*100)}pct_{mode[:4]}",
                                 "sens": round(o.get("sens", float('nan')), 4),
                                 "spec": round(o.get("spec", float('nan')), 4),
                                 "ppv": round(o.get("ppv", float('nan')), 4),
                                 "npv": round(o.get("npv", float('nan')), 4)})
            # regression
            s = slab(model, site, "diam")
            if not s.empty:
                y, p = s.y_true.to_numpy(float), s.pred_value.to_numpy(float)
                mm = bootstrap_ci(y, p, mae, need_both_classes=False)
                rr = bootstrap_ci(y, p, r2, need_both_classes=False)
                summary.append({
                    "model": model, "site": site, "endpoint": "diam", "target": "regression",
                    "n": int((~np.isnan(y) & ~np.isnan(p)).sum()), "pos": "",
                    "mae": round(mm[0], 4), "mae_lo": round(mm[1], 4), "mae_hi": round(mm[2], 4),
                    "r2": round(rr[0], 4), "r2_lo": round(rr[1], 4), "r2_hi": round(rr[2], 4),
                })

    pd.DataFrame(summary).to_csv(os.path.join(rep_dir, "metrics_summary.csv"), index=False)
    pd.DataFrame(calib).to_csv(os.path.join(rep_dir, "calibration_metrics.csv"), index=False)
    pd.DataFrame(util).to_csv(os.path.join(rep_dir, "clinical_utility.csv"), index=False)

    # ---------- paired deltas vs EHR floor ----------
    deltas = []
    if FLOOR in models:
        for model in [m for m in models if m != FLOOR]:
            for site in ("root", "asc"):
                for ep in BIN_ENDPOINTS:
                    a, b = slab(model, site, ep), slab(FLOOR, site, ep)
                    j = a.merge(b, on="subject_id", suffixes=("_m", "_f"))
                    if j.empty:
                        continue
                    y = j.y_true_m.to_numpy(float)
                    pm, pf = j.pred_prob_m.to_numpy(float), j.pred_prob_f.to_numpy(float)
                    for metric_name, fn in [("auroc", auroc), ("auprc", auprc)]:
                        d = paired_bootstrap_diff(y, pm, pf, fn)
                        deltas.append({"model": model, "vs": FLOOR, "site": site, "endpoint": ep,
                                       "metric": metric_name, "delta": round(d[0], 4),
                                       "lo": round(d[1], 4), "hi": round(d[2], 4),
                                       "excludes_0": bool(d[1] > 0 or d[2] < 0)})
                # regression deltas (model - floor; MAE lower is better)
                a, b = slab(model, site, "diam"), slab(FLOOR, site, "diam")
                j = a.merge(b, on="subject_id", suffixes=("_m", "_f"))
                if not j.empty:
                    y = j.y_true_m.to_numpy(float)
                    pm, pf = j.pred_value_m.to_numpy(float), j.pred_value_f.to_numpy(float)
                    for metric_name, fn in [("mae", mae), ("r2", r2)]:
                        d = paired_bootstrap_diff(y, pm, pf, fn, need_both_classes=False)
                        deltas.append({"model": model, "vs": FLOOR, "site": site, "endpoint": "diam",
                                       "metric": metric_name, "delta": round(d[0], 4),
                                       "lo": round(d[1], 4), "hi": round(d[2], 4),
                                       "excludes_0": bool(d[1] > 0 or d[2] < 0)})
    pd.DataFrame(deltas).to_csv(os.path.join(rep_dir, "deltas_vs_ehr.csv"), index=False)

    # ---------- event counts per fold (incl. ge50 not-estimable) ----------
    ec = []
    ref = df[df.model_name == models[0]]
    for site in ("root", "asc"):
        diam = df[(df.model_name == models[0]) & (df.site == site) & (df.endpoint == "diam")]
        diam = diam.dropna(subset=["y_true"])
        for cut, name in [(4.0, "ge40"), (4.5, "ge45"), (5.0, "ge50")]:
            pos = int((diam.y_true >= cut).sum())
            ec.append({"site": site, "endpoint": name, "cut_cm": cut, "n": int(len(diam)),
                       "pos": pos, "estimable": pos >= 1})
    pd.DataFrame(ec).to_csv(os.path.join(rep_dir, "event_counts.csv"), index=False)

    _plots(df, models, slab, cm, plot_dir)

    # ---------- console highlight: floor vs star ----------
    log.info("=== headline AUROC (from metrics_summary) ===")
    sm = pd.DataFrame(summary)
    hb = sm[sm.target == "binary"].pivot_table(index="model", columns=["site", "endpoint"],
                                               values="auroc", aggfunc="first")
    log.info("\n%s", hb.reindex([m for m in models if m in hb.index]).to_string())
    if deltas:
        dd = pd.DataFrame(deltas)
        sig = dd[dd.excludes_0]
        log.info("=== deltas vs EHR floor with CI excluding 0: %d ===", len(sig))
        if len(sig):
            log.info("\n%s", sig.to_string(index=False))
    log.info("Saved report -> %s", rep_dir)


def _plots(df, models, slab, cm, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # calibration grid: root/asc x ge40/ge45
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (site, ep) in zip(axes.ravel(),
                              [("root", "ge40"), ("root", "ge45"), ("asc", "ge40"), ("asc", "ge45")]):
        ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
        for model in models:
            s = slab(model, site, ep)
            if s.empty:
                continue
            rc = cm.reliability_curve(s.y_true.to_numpy(float), s.pred_prob.to_numpy(float), n_bins=6)
            if rc["mean_pred"]:
                ax.plot(rc["mean_pred"], rc["obs_freq"], marker="o", ms=3, lw=1, label=model)
        ax.set_title(f"{site} {ep}"); ax.set_xlabel("predicted"); ax.set_ylabel("observed")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Calibration (reliability) — floor vs fusion")
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "calibration_grid.png"), dpi=120)
    plt.close(fig)

    # AUROC bar comparison
    from multimodal_aorta.training.bootstrap import auroc

    def _auroc_safe(s):
        if s.empty:
            return np.nan
        y, p = s.y_true.to_numpy(float), s.pred_prob.to_numpy(float)
        m = ~(np.isnan(y) | np.isnan(p))
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            return np.nan
        return auroc(y[m], p[m])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, site in zip(axes, ("root", "asc")):
        eps = ["ge40", "ge45"]
        xw = np.arange(len(eps)); w = 0.8 / max(len(models), 1)
        for i, model in enumerate(models):
            vals = [_auroc_safe(slab(model, site, ep)) for ep in eps]
            ax.bar(xw + i * w, vals, w, label=model)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xticks(xw + 0.4 - w / 2); ax.set_xticklabels(eps)
        ax.set_title(f"{site} AUROC"); ax.set_ylim(0.4, 1.0)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "model_comparison_auroc.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
