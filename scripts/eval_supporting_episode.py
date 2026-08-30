"""
Two supporting analyses on already-saved OOF predictions (no retraining):

  1. LAG SENSITIVITY. Does the CXR model read stable anatomy or acute state? Stratify
     the headline stack's performance by how many days the radiograph precedes the echo.
     Flat performance across lag => stable-anatomy signal, which is what an opportunistic
     screen needs.

  2. TOST EQUIVALENCE on the ECG-measurement increment. The 5-Aug report argued in prose
     that the ECG increment's magnitude is negligible even where its CI excludes zero.
     A pre-specified minimal clinically important difference (MCID) plus two one-sided
     tests turns that into a formal equivalence result: if the 90% CI of the increment
     lies entirely within +/-MCID, we conclude equivalence (no meaningful difference).

Run: sbatch scripts/slurm_eval_supporting_episode.sh
Out: outputs/eval_supporting_episode/results.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src"))

MCID_AUROC = float(os.environ.get("MCID_AUROC", "0.01"))   # pre-specified
MCID_R2 = float(os.environ.get("MCID_R2", "0.01"))
LAG_BINS = [(0, 7), (8, 30), (31, 90), (91, 365)]


def cluster_boot_diff_dist(y, pa, pb, groups, metric, n_boot=2000, seed=0, need_both=True):
    """Return the bootstrap distribution of metric(pa)-metric(pb), resampling patients."""
    m = ~np.isnan(y) & ~np.isnan(pa) & ~np.isnan(pb)
    y, pa, pb, groups = y[m], pa[m], pb[m], np.asarray(groups)[m]
    uniq = np.unique(groups); rows = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed); out = []
    point = float(metric(y, pa) - metric(y, pb))
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if need_both and len(np.unique(y[idx])) < 2:
            continue
        try:
            out.append(float(metric(y[idx], pa[idx]) - metric(y[idx], pb[idx])))
        except Exception:
            continue
    return point, np.array(out)


def main():
    from multimodal_aorta.data.episodes import load_episodes, make_episode_id
    from multimodal_aorta.training.bootstrap import cluster_bootstrap_ci, auroc, r2, fmt

    ep = load_episodes(PC, require_ecg=False)
    diam = {"root": dict(zip(ep.episode_id.astype(str), ep.target_root)),
            "asc": dict(zip(ep.episode_id.astype(str), ep.target_asc))}
    sid_of = dict(zip(ep.episode_id.astype(str), ep.subject_id))

    # --- per-episode CXR lag (mean days_before across the episode's instances) ---
    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    lag = inst.groupby("episode_id")["days_before_echo"].mean()

    stack = pd.read_csv(os.path.join(ROOT, "outputs", "geometry_stack_episode", "oof_predictions.csv"))
    stack["episode_id"] = stack.episode_id.astype(str)

    results = {"mcid_auroc": MCID_AUROC, "mcid_r2": MCID_R2, "lag_sensitivity": {}, "tost_ecg": {}}

    # ===== 1. lag sensitivity =====
    for site in ("root", "asc"):
        s = stack[stack.site == site].copy()
        s["lag"] = s.episode_id.map(lag)
        rows = []
        for lo, hi in LAG_BINS:
            b = s[(s.lag >= lo) & (s.lag <= hi)]
            d = b.diam_true.to_numpy(float); p = b.pred_stack.to_numpy(float)
            g = b.episode_id.map(sid_of).to_numpy()
            y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
            rows.append({"bin": f"{lo}-{hi}d", "n": len(b),
                         "ge40": fmt(cluster_bootstrap_ci(y40, p, g, auroc)),
                         "r2": fmt(cluster_bootstrap_ci(d, p, g, r2, need_both_classes=False))})
        results["lag_sensitivity"][site] = rows
        log.info("[%s] lag sensitivity:", site)
        for r in rows:
            log.info("   %-9s n=%-5d ge40 %s | R2 %s", r["bin"], r["n"], r["ge40"], r["r2"])

    # ===== 2. TOST on the ECG-measurement increment =====
    inc = pd.read_csv(os.path.join(ROOT, "outputs", "ecg_increment_episode", "oof_predictions.csv"))
    inc["episode_id"] = inc.episode_id.astype(str)
    for site in ("root", "asc"):
        b = inc[inc.site == site]
        d = b.diam_true.to_numpy(float)
        p2 = b.pred_cxr_ehr.to_numpy(float); p3 = b.pred_cxr_ehr_ecg.to_numpy(float)
        g = b.subject_id.to_numpy(); y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        out = {}
        for name, y, metric, mcid, nb in [("ge40", y40, auroc, MCID_AUROC, True),
                                          ("r2", d, r2, MCID_R2, False)]:
            pt, dist = cluster_boot_diff_dist(y, p3, p2, g, metric, need_both=nb)
            lo, hi = float(np.percentile(dist, 5)), float(np.percentile(dist, 95))   # 90% CI (TOST)
            equivalent = (lo > -mcid) and (hi < mcid)
            out[name] = {"increment": round(pt, 4), "ci90": [round(lo, 4), round(hi, 4)],
                         "mcid": mcid, "equivalent_within_mcid": bool(equivalent)}
            log.info("[%s TOST %s] increment %.4f  90%%CI [%.4f, %.4f]  vs +/-%.3f  => %s",
                     site, name, pt, lo, hi, mcid, "EQUIVALENT" if equivalent else "not equivalent")
        results["tost_ecg"][site] = out

    out_dir = os.path.join(ROOT, "outputs", "eval_supporting_episode"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
