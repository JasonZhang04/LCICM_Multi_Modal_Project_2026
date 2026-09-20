"""
Complementarity diagnostics (Codex brief, "Required complementarity diagnostics" 3-7),
computed from EXISTING out-of-fold predictions -- no retraining.

Question: after the strongest arm (fine-tuned CXR) has spoken, do the ECG waveform
and EHR predictions explain any of what it got WRONG -- and if so, for whom?

All arms are per-fold-model SCALAR predictions in cm, so their units are comparable
across folds (no A2 basis problem). Every CV here is patient-grouped on the immutable
folds; alpha is chosen by patient-grouped inner CV.

Outputs -> outputs/diag_complementarity/{results.json, per_episode.csv}
Env: N_BOOT (default 500) to trade CI precision for speed.
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src"))
N_BOOT = int(os.environ.get("N_BOOT", "500"))
CXR_META = "/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/mimic-cxr-2.0.0-metadata.csv.gz"


def nested_ridge_oof(F, y, sid, fold_id):
    """Patient-grouped outer folds (immutable ids); alpha by patient-grouped inner CV."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    out = np.full(len(y), np.nan)
    for k in np.unique(fold_id):
        tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        sc = StandardScaler().fit(F[tr]); Ztr, Zte = sc.transform(F[tr]), sc.transform(F[te])
        best_a, best = 1.0, 1e18
        for a in (0.1, 0.3, 1, 3, 10, 30, 100):
            e = [np.mean(np.abs(y[tr][ib] - Ridge(alpha=a).fit(Ztr[ia], y[tr][ia]).predict(Ztr[ib])))
                 for ia, ib in GroupKFold(3).split(Ztr, groups=sid[tr])]
            if np.mean(e) < best: best, best_a = np.mean(e), a
        out[te] = Ridge(alpha=best_a).fit(Ztr, y[tr]).predict(Zte)
    return out


def main():
    from multimodal_aorta.training.bootstrap import (
        cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt)
    from multimodal_aorta.data.episodes import make_episode_id

    def _ci(*a, **k):
        try: return cluster_bootstrap_ci(*a, n_boot=N_BOOT, **k)
        except TypeError: return cluster_bootstrap_ci(*a, **k)
    def _pd(*a, **k):
        try: return paired_cluster_bootstrap_diff(*a, n_boot=N_BOOT, **k)
        except TypeError: return paired_cluster_bootstrap_diff(*a, **k)

    # ---- load OOF scalars -------------------------------------------------------
    ft = pd.read_csv(os.path.join(ROOT, "outputs/cxr_finetune_episode/oof_predictions.csv"))
    eg = pd.read_csv(os.path.join(ROOT, "outputs/ecg_waveform_episode/oof_predictions.csv"))
    egb = pd.read_csv(os.path.join(ROOT, "outputs/ecg_waveform_episode_broad/oof_predictions.csv"))
    eh = pd.read_csv(os.path.join(ROOT, "outputs/ehr_baseline_episode/oof_predictions.csv"))
    eh = eh[(eh.endpoint == "diam")]
    mn = "hgb" if (eh.model_name == "hgb").any() else eh.model_name.iloc[0]
    eh = eh[eh.model_name == mn][["episode_id", "site", "pred_value"]].rename(columns={"pred_value": "ehr"})
    for d in (ft, eg, egb, eh): d["episode_id"] = d.episode_id.astype(str)
    fa = pd.read_csv(os.path.join(PC, "episode_fold_assignments.csv")); fa["episode_id"] = fa.episode_id.astype(str)
    fold_of = dict(zip(fa.episode_id, fa.fold_id))

    # ---- covariates for stratification -----------------------------------------
    ehrf = pd.read_csv(os.path.join(PC, "ehr_features_episode.csv")); ehrf["episode_id"] = ehrf.episode_id.astype(str)
    cov = ehrf.set_index("episode_id")[["age", "sex", "bsa", "sbp", "height_cm", "weight_kg"]]
    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    inst["dicom_id"] = inst.dicom_id.astype(str)
    meta = pd.read_csv(CXR_META, usecols=["dicom_id", "ViewPosition"]); meta["dicom_id"] = meta.dicom_id.astype(str)
    inst = inst.merge(meta, on="dicom_id", how="left")
    per_ep = inst.groupby("episode_id").agg(frac_pa=("ViewPosition", lambda v: float((v == "PA").mean())),
                                            lag=("days_before_echo", "mean"), n_img=("dicom_id", "size"))

    results = {"n_boot": N_BOOT, "ehr_model": mn, "sites": {}}
    rows_out = []
    for site in ("root", "asc"):
        a = ft[ft.site == site][["episode_id", "subject_id", "diam_true", "pred_cxr_ft"]].rename(columns={"pred_cxr_ft": "cxr"})
        b = eg[eg.site == site][["episode_id", "pred_ecg_waveform"]].rename(columns={"pred_ecg_waveform": "ecg"})
        bb = egb[egb.site == site][["episode_id", "pred_ecg_waveform"]].rename(columns={"pred_ecg_waveform": "ecg_broad"})
        c = eh[eh.site == site][["episode_id", "ehr"]]
        m = a.merge(b, on="episode_id", how="left").merge(bb, on="episode_id", how="left").merge(c, on="episode_id", how="left")
        m["fold"] = m.episode_id.map(fold_of)
        m = m.join(cov, on="episode_id").join(per_ep, on="episode_id")
        m = m[m.fold.notna() & m.ehr.notna()].reset_index(drop=True)
        has_ecg = m.ecg.notna()
        y = m.diam_true.to_numpy(float); g = m.subject_id.to_numpy(int); fold = m.fold.to_numpy(int)
        y40 = (y >= 4.0).astype(float)
        S = {"n_all": int(len(m)), "n_ecg_present": int(has_ecg.sum()), "n_pos_all": int(y40.sum())}
        log.info("[%s] episodes %d (ECG present %d) | >=4cm %d", site, len(m), int(has_ecg.sum()), int(y40.sum()))

        # ---- 1. residual correlation between arms (ECG-present subset) ------------
        e = m[has_ecg].reset_index(drop=True)
        ye = e.diam_true.to_numpy(float)
        r_cxr, r_ecg, r_ehr = ye - e.cxr, ye - e.ecg, ye - e.ehr
        S["resid_corr"] = {"cxr_vs_ecg": float(np.corrcoef(r_cxr, r_ecg)[0, 1]),
                           "cxr_vs_ehr": float(np.corrcoef(r_cxr, r_ehr)[0, 1]),
                           "ecg_vs_ehr": float(np.corrcoef(r_ecg, r_ehr)[0, 1])}
        S["pred_corr"] = {"cxr_vs_ecg": float(np.corrcoef(e.cxr, e.ecg)[0, 1]),
                          "cxr_vs_ehr": float(np.corrcoef(e.cxr, e.ehr)[0, 1]),
                          "ecg_vs_ehr": float(np.corrcoef(e.ecg, e.ehr)[0, 1])}
        log.info("[%s] residual corr: cxr~ecg %.3f | cxr~ehr %.3f | ecg~ehr %.3f", site, *S["resid_corr"].values())
        log.info("[%s] prediction corr: cxr~ecg %.3f | cxr~ehr %.3f | ecg~ehr %.3f", site, *S["pred_corr"].values())

        # ---- 2. does ECG / EHR / both explain the CXR residual? (nested, grouped) ---
        ge, fe = e.subject_id.to_numpy(int), e.fold.to_numpy(int)
        rc = r_cxr.to_numpy(float)
        S["explains_cxr_residual_r2"] = {}
        for name, cols in (("ecg", ["ecg"]), ("ehr", ["ehr"]), ("ecg+ehr", ["ecg", "ehr"]),
                           ("ecg_broad", ["ecg_broad"]), ("ecg+ehr+cxr_interactions", None)):
            if cols is None:
                # cheap conditional model: let the correction depend on the CXR prediction
                F = np.column_stack([e.ecg, e.ehr, e.cxr, e.ecg * e.cxr, e.ehr * e.cxr, e.bsa.fillna(e.bsa.median()) * e.cxr])
            else:
                F = e[cols].to_numpy(float)
                if np.isnan(F).any():
                    continue
            p = nested_ridge_oof(F, rc, ge, fe)
            S["explains_cxr_residual_r2"][name] = fmt(_ci(rc, p, ge, r2, need_both_classes=False))
            log.info("[%s] R2 of CXR residual explained by %-26s %s", site, name + ":", S["explains_cxr_residual_r2"][name])

        # ---- 3. factorial: CXR, +EHR, +ECG, +both (ECG-present subset, matched) -----
        arms = {"cxr": ["cxr"], "cxr+ehr": ["cxr", "ehr"], "cxr+ecg": ["cxr", "ecg"],
                "cxr+ehr+ecg": ["cxr", "ehr", "ecg"], "cxr+ehr+ecg_broad": ["cxr", "ehr", "ecg_broad"]}
        P = {}
        for name, cols in arms.items():
            F = e[cols].to_numpy(float)
            if np.isnan(F).any(): continue
            P[name] = nested_ridge_oof(F, ye, ge, fe)
        y40e = (ye >= 4.0).astype(float)
        S["factorial_ecg_present"] = {}
        for name, p in P.items():
            d = {"r2": fmt(_ci(ye, p, ge, r2, need_both_classes=False)), "auroc": fmt(_ci(y40e, p, ge, auroc))}
            if name != "cxr":
                d["d_r2_vs_cxr"] = fmt(_pd(ye, p, P["cxr"], ge, r2, need_both_classes=False))
                d["d_auroc_vs_cxr"] = fmt(_pd(y40e, p, P["cxr"], ge, auroc))
            S["factorial_ecg_present"][name] = d
            log.info("[%s] %-18s R2 %s AUROC %s %s", site, name, d["r2"], d["auroc"],
                     ("| dR2 " + d["d_r2_vs_cxr"] + " dAUROC " + d["d_auroc_vs_cxr"]) if name != "cxr" else "")

        # ---- 4. WHERE does the correction land? stratify the full-arm increment ------
        full, base = P["cxr+ehr+ecg"], P["cxr"]
        e["pred_full"], e["pred_cxr_only"] = full, base
        strata = {
            "cxr_pred_band": pd.cut(e.cxr, [-np.inf, 3.2, 3.6, 4.0, np.inf], labels=["<3.2", "3.2-3.6", "3.6-4.0", ">=4.0"]),
            "view": np.where(e.frac_pa >= 0.5, "mostly_PA", "mostly_AP"),
            "bsa_tertile": pd.qcut(e.bsa.rank(method="first"), 3, labels=["low", "mid", "high"]) if e.bsa.notna().mean() > 0.5 else None,
            "lag_days": pd.cut(e.lag, [-1, 7, 30, 90, 366], labels=["0-7", "7-30", "30-90", "90-365"]),
            "age": pd.cut(e.age, [0, 50, 65, 80, 200], labels=["<50", "50-65", "65-80", ">80"]),
        }
        S["increment_by_stratum"] = {}
        for sname, lab in strata.items():
            if lab is None: continue
            lab = pd.Series(np.asarray(lab).astype(str), index=e.index)
            S["increment_by_stratum"][sname] = {}
            for lv in [l for l in pd.unique(lab) if l not in ("nan", "None")]:
                idx = np.where(lab.to_numpy() == lv)[0]
                if len(idx) < 200 or len(np.unique(ge[idx])) < 100: continue
                d = {"n": int(len(idx)), "n_pos": int(y40e[idx].sum()),
                     "mae_cxr": float(np.mean(np.abs(ye[idx] - base[idx]))),
                     "mae_full": float(np.mean(np.abs(ye[idx] - full[idx]))),
                     "d_r2": fmt(_pd(ye[idx], full[idx], base[idx], ge[idx], r2, need_both_classes=False))}
                if y40e[idx].sum() >= 15 and y40e[idx].sum() < len(idx) - 15:
                    d["d_auroc"] = fmt(_pd(y40e[idx], full[idx], base[idx], ge[idx], auroc))
                S["increment_by_stratum"][sname][lv] = d
                log.info("[%s]   %s=%-9s n=%5d pos=%4d  MAE cxr %.3f -> full %.3f | dR2 %s | dAUROC %s",
                         site, sname, lv, d["n"], d["n_pos"], d["mae_cxr"], d["mae_full"], d["d_r2"], d.get("d_auroc", "-"))

        # ---- 5. is the CXR residual structured by covariates the CXR cannot see? -----
        S["cxr_resid_corr_with_covariates"] = {}
        for cname in ("age", "sex", "bsa", "sbp", "frac_pa", "lag", "n_img"):
            v = e[cname].to_numpy(float); ok = ~np.isnan(v)
            if ok.sum() > 500:
                S["cxr_resid_corr_with_covariates"][cname] = float(np.corrcoef(rc[ok], v[ok])[0, 1])
        log.info("[%s] corr(CXR residual, covariate): %s", site,
                 {k: round(v, 3) for k, v in S["cxr_resid_corr_with_covariates"].items()})

        # ---- 6. tail behaviour: mean error among true >=4 cm, per arm ----------------
        pos = y40e == 1
        S["mean_error_in_true_ge40"] = {n: float(np.mean(P[n][pos] - ye[pos])) for n in P}
        log.info("[%s] mean signed error among TRUE >=4cm: %s", site, {k: round(v, 3) for k, v in S["mean_error_in_true_ge40"].items()})

        results["sites"][site] = S
        for i in range(len(e)):
            rows_out.append({"episode_id": e.episode_id[i], "subject_id": int(ge[i]), "site": site,
                             "diam_true": ye[i], "pred_cxr": e.cxr[i], "pred_ecg": e.ecg[i], "pred_ehr": e.ehr[i],
                             "pred_cxr_only_stack": base[i], "pred_full_stack": full[i]})

    out = os.path.join(ROOT, "outputs", "diag_complementarity"); os.makedirs(out, exist_ok=True)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=2)
    pd.DataFrame(rows_out).to_csv(os.path.join(out, "per_episode.csv"), index=False)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
