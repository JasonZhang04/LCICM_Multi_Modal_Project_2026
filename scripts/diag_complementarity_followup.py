"""
Follow-up to diag_complementarity.py. Two questions it could not answer:

1. Is the "interaction" gain real conditional signal, or just the CXR arm's own
   shrinkage? Control: regress the CXR residual on the CXR prediction ALONE. Anything
   the interaction arm explains beyond that is genuinely conditional on ECG/EHR.

2. Does strictly PRE-INDEX clinical history (hypertension, aortic regurgitation,
   bicuspid valve, connective-tissue disease, prior aneurysm dx, ...) add signal on
   top of fine-tuned CXR? The existing EHR arm is anthropometric only, so the current
   "EHR adds nothing" is a null for BODY SIZE, not for CLINICAL HISTORY. Flags come
   from admissions discharged before the echo timestamp (nothing from the index stay).

Same nested patient-grouped ridge as the parent script. Outputs ->
outputs/diag_complementarity/followup.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_complementarity import nested_ridge_oof, ROOT, PC, N_BOOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
sys.path.insert(0, os.path.join(ROOT, "src"))
HIST = ["hypertension", "diabetes", "aortic_aneurysm_dx", "aortic_regurg", "bicuspid_congenital",
        "marfan_ctd", "atrial_fib", "ckd", "tobacco", "aortic_stenosis"]


def main():
    from multimodal_aorta.training.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt
    try:
        from multimodal_aorta.training.bootstrap import auprc
    except ImportError:
        from sklearn.metrics import average_precision_score as auprc
    from multimodal_aorta.data.episodes import make_episode_id
    _ci = lambda *a, **k: cluster_bootstrap_ci(*a, n_boot=N_BOOT, **k)
    _pd = lambda *a, **k: paired_cluster_bootstrap_diff(*a, n_boot=N_BOOT, **k)

    ft = pd.read_csv(os.path.join(ROOT, "outputs/cxr_finetune_episode/oof_predictions.csv"))
    eg = pd.read_csv(os.path.join(ROOT, "outputs/ecg_waveform_episode/oof_predictions.csv"))
    eh = pd.read_csv(os.path.join(ROOT, "outputs/ehr_baseline_episode/oof_predictions.csv"))
    eh = eh[eh.endpoint == "diam"][["episode_id", "site", "pred_value"]].rename(columns={"pred_value": "ehr"})
    for d in (ft, eg, eh): d["episode_id"] = d.episode_id.astype(str)
    fa = pd.read_csv(os.path.join(PC, "episode_fold_assignments.csv")); fa["episode_id"] = fa.episode_id.astype(str)
    fold_of = dict(zip(fa.episode_id, fa.fold_id))
    ehrf = pd.read_csv(os.path.join(PC, "ehr_features_episode.csv")); ehrf["episode_id"] = ehrf.episode_id.astype(str)
    bsa = ehrf.set_index("episode_id")["bsa"]
    hist = pd.read_csv(os.path.join(PC, "ehr_history_flags_FEASIBILITY.csv"))
    hist["episode_id"] = [make_episode_id(s, m) for s, m in zip(hist.subject_id, hist.measurement_id)]
    hist = hist.set_index("episode_id")[HIST + ["n_pre_adm"]].astype(float)

    out = {"n_boot": N_BOOT, "sites": {}}
    for site in ("root", "asc"):
        a = ft[ft.site == site][["episode_id", "subject_id", "diam_true", "pred_cxr_ft"]].rename(columns={"pred_cxr_ft": "cxr"})
        b = eg[eg.site == site][["episode_id", "pred_ecg_waveform"]].rename(columns={"pred_ecg_waveform": "ecg"})
        c = eh[eh.site == site][["episode_id", "ehr"]]
        e = a.merge(b, on="episode_id").merge(c, on="episode_id")
        e["fold"] = e.episode_id.map(fold_of); e["bsa"] = e.episode_id.map(bsa)
        e = e.join(hist, on="episode_id").dropna(subset=["fold", "ecg", "ehr"]).reset_index(drop=True)
        e["bsa"] = e.bsa.fillna(e.bsa.median()); e[HIST + ["n_pre_adm"]] = e[HIST + ["n_pre_adm"]].fillna(0)
        e["any_hist"] = (e.n_pre_adm > 0).astype(float)
        y = e.diam_true.to_numpy(float); g = e.subject_id.to_numpy(int); f = e.fold.to_numpy(int)
        y40 = (y >= 4.0).astype(float); rc = y - e.cxr.to_numpy(float)
        S = {"n": int(len(e)), "n_pos": int(y40.sum()),
             "frac_with_any_pre_index_admission": float(e.any_hist.mean())}
        log.info("[%s] n=%d pos=%d | with >=1 pre-index admission %.1f%%", site, len(e), int(y40.sum()), 100 * e.any_hist.mean())

        H = e[HIST].to_numpy(float); X = {"cxr": e[["cxr"]].to_numpy(float), "ecg": e[["ecg"]].to_numpy(float),
                                          "ehr": e[["ehr"]].to_numpy(float), "bsa": e[["bsa"]].to_numpy(float)}
        inter = np.column_stack([X["ecg"] * X["cxr"], X["ehr"] * X["cxr"], X["bsa"] * X["cxr"]])
        hist_inter = H * X["cxr"]

        # ---- 1. residual arms, with the recalibration CONTROL --------------------
        R = {"cxr_only (shrinkage control)": X["cxr"],
             "cxr + ecg + ehr (main effects)": np.column_stack([X["cxr"], X["ecg"], X["ehr"]]),
             "cxr + ecg + ehr + interactions": np.column_stack([X["cxr"], X["ecg"], X["ehr"], inter]),
             "history flags only": np.column_stack([H, e[["any_hist"]].to_numpy()]),
             "cxr + history": np.column_stack([X["cxr"], H]),
             "cxr + ecg + ehr + history": np.column_stack([X["cxr"], X["ecg"], X["ehr"], H]),
             "cxr + ecg + ehr + history + all interactions": np.column_stack([X["cxr"], X["ecg"], X["ehr"], H, inter, hist_inter])}
        S["explains_cxr_residual_r2"] = {}
        for name, F in R.items():
            p = nested_ridge_oof(F, rc, g, f)
            S["explains_cxr_residual_r2"][name] = fmt(_ci(rc, p, g, r2, need_both_classes=False))
            log.info("[%s] residual R2 | %-46s %s", site, name, S["explains_cxr_residual_r2"][name])

        # ---- 2. factorial on the TARGET, paired vs cxr-only ------------------------
        A = {"cxr": X["cxr"],
             "cxr+ehr+ecg": np.column_stack([X["cxr"], X["ehr"], X["ecg"]]),
             "cxr+ehr+ecg+interactions": np.column_stack([X["cxr"], X["ehr"], X["ecg"], inter]),
             "cxr+history": np.column_stack([X["cxr"], H]),
             "cxr+ehr+history": np.column_stack([X["cxr"], X["ehr"], H]),
             "cxr+ehr+ecg+history": np.column_stack([X["cxr"], X["ehr"], X["ecg"], H]),
             "cxr+ehr+ecg+history+interactions": np.column_stack([X["cxr"], X["ehr"], X["ecg"], H, inter, hist_inter])}
        P = {k: nested_ridge_oof(F, y, g, f) for k, F in A.items()}
        S["factorial"] = {}
        for k, p in P.items():
            d = {"r2": fmt(_ci(y, p, g, r2, need_both_classes=False)), "auroc": fmt(_ci(y40, p, g, auroc)),
                 "auprc": fmt(_ci(y40, p, g, auprc))}
            if k != "cxr":
                d["d_r2_vs_cxr"] = fmt(_pd(y, p, P["cxr"], g, r2, need_both_classes=False))
                d["d_auroc_vs_cxr"] = fmt(_pd(y40, p, P["cxr"], g, auroc))
                d["d_auprc_vs_cxr"] = fmt(_pd(y40, p, P["cxr"], g, auprc))
            S["factorial"][k] = d
            log.info("[%s] %-34s R2 %s AUROC %s AUPRC %s %s", site, k, d["r2"], d["auroc"], d["auprc"],
                     ("| d: R2 " + d["d_r2_vs_cxr"] + " AUROC " + d["d_auroc_vs_cxr"] + " AUPRC " + d["d_auprc_vs_cxr"]) if k != "cxr" else "")

        # ---- 3. which history flags carry weight? (univariate, within the residual) --
        S["history_flag_univariate"] = {}
        for h in HIST:
            v = e[h].to_numpy(float)
            if v.sum() < 30: continue
            S["history_flag_univariate"][h] = {"n_flag": int(v.sum()), "mean_cxr_resid_flag": float(rc[v == 1].mean()),
                                              "mean_cxr_resid_noflag": float(rc[v == 0].mean()),
                                              "pos_rate_flag": float(y40[v == 1].mean()), "pos_rate_noflag": float(y40[v == 0].mean())}
        log.info("[%s] mean CXR residual (cm) with flag vs without: %s", site,
                 {h: (round(d["mean_cxr_resid_flag"], 3), round(d["mean_cxr_resid_noflag"], 3)) for h, d in S["history_flag_univariate"].items()})
        out["sites"][site] = S

    json.dump(out, open(os.path.join(ROOT, "outputs/diag_complementarity/followup.json"), "w"), indent=2)
    log.info("Saved followup.json")


if __name__ == "__main__":
    main()
