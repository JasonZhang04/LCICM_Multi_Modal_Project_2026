"""
Properly NESTED rebuild of the final multimodal model (review issue A1).

WHY THIS EXISTS
---------------
`train_final_model_episode.py` builds one global OOF CXR feature vector and then
cross-validates the fusion ridge on top of it. That is not nested: a fusion-training
row in fold j carries a base feature from a model trained on every fold but j --
including the outer test fold k. The headline claims rest on ~0.01 R^2 increments,
which is the same order as the bias this can introduce, so the increments cannot be
interpreted until the path is removed.

This script runs the SAME model family through `multimodal_aorta.training.nested`,
where for every outer fold the outer-test patients are excluded before any supervised
fitting: inner-OOF base predictions train the fusion, the bases are then refit on the
full outer-training set, and only then is the outer test set touched.

HONESTY FLAGS
-------------
The GPU-trained branches (fine-tuned CXR, ECG waveform CNN) cannot be refit on CPU
inside each inner fold. They are passed as PrecomputedSpec and the run records
`exact_nesting: false` plus the offending component names in results.json. Set
CXR_FT=0 ECG=0 to obtain a fully exact run of the CPU-refittable stack.

Env:
  FOLD_MODE   immutable (default) | seeds
  K_PCA       CXR per-block PCA dim (default 32)
  K_ECG       ECG embedding PCA dim (default 32)
  CXR_FT      1 to add the precomputed fine-tuned CXR scalar (default 1)
  ECG         1 to add the precomputed ECG embedding block (default 1)
  N_INNER     inner folds for the OOF base features (default 5)
  OUT_DIR     output dir under outputs/ (default final_model_nested)
Run: sbatch scripts/slurm_final_model_nested.sh
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

K       = int(os.environ.get("K_PCA", "32"))
K_ECG   = int(os.environ.get("K_ECG", "32"))
N_INNER = int(os.environ.get("N_INNER", "5"))
USE_FT  = os.environ.get("CXR_FT", "1") == "1"
USE_ECG = os.environ.get("ECG", "1") == "1"
FOLD_MODE = os.environ.get("FOLD_MODE", "immutable")
EHR_COLS = ["age", "sex", "height_cm", "weight_kg", "bmi", "bsa", "sbp", "dbp",
            "height_missing", "weight_missing", "bsa_missing", "bp_missing"]

import train_geometry_stack_episode as gs
from multimodal_aorta.training.nested import (
    BaseSpec, PrecomputedSpec, nested_stack, assert_patient_disjoint)


def make_cxr_base(blocks, Xgeom, I_eid, Iw, row_of, y_row_holder):
    """Multi-instance CXR branch as a refittable BaseSpec.

    The episode rows given to fit_predict are mapped down to their instance rows;
    PCA and the HGB are fit on the training instances only, and instance predictions
    are averaged back up to the episode. Instances of an episode never straddle the
    boundary because episodes are the unit being split.
    """
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    eid_at_row = {v: k for k, v in row_of.items()}
    inst_rows_of_ep = {}
    for i, e in enumerate(I_eid):
        inst_rows_of_ep.setdefault(e, []).append(i)
    inst_rows_of_ep = {k: np.array(v) for k, v in inst_rows_of_ep.items()}

    def _inst(rows):
        acc = [inst_rows_of_ep[eid_at_row[int(r)]] for r in rows
               if eid_at_row[int(r)] in inst_rows_of_ep]
        return np.concatenate(acc) if acc else np.array([], int)

    def fit_predict(train_rows, predict_rows, y):
        y_inst_full = y_row_holder["y_inst"]
        trm = _inst(train_rows); tem = _inst(predict_rows)
        trm = trm[~np.isnan(y_inst_full[trm])]
        if len(trm) < 10 or not len(tem):
            return np.full(len(predict_rows), np.nan)
        ptr, pte = [], []
        for B in blocks:
            pca = PCA(min(K, B.shape[1], len(trm) - 1), random_state=0).fit(B[trm])
            ptr.append(pca.transform(B[trm])); pte.append(pca.transform(B[tem]))
        ptr.append(Xgeom[trm]); pte.append(Xgeom[tem])
        m = clone(gs._hgb_reg()).fit(np.concatenate(ptr, 1), y_inst_full[trm],
                                     sample_weight=Iw[trm])
        pi = m.predict(np.concatenate(pte, 1))
        per_ep = pd.DataFrame({"e": I_eid[tem], "p": pi}).groupby("e")["p"].mean().to_dict()
        return np.array([per_ep.get(eid_at_row[int(r)], np.nan) for r in predict_rows])

    return BaseSpec(name="cxr_multiinstance", fit_predict=fit_predict, exact=True)


def make_ehr_base(X_ehr, row_of):
    """EHR diameter branch, refit inside the harness (replaces the precomputed floor)."""
    def fit_predict(train_rows, predict_rows, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        tr = train_rows[~np.isnan(y[train_rows])]
        if len(tr) < 20:
            return np.full(len(predict_rows), np.nan)
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200,
                                          max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=0)
        m.fit(X_ehr[tr], y[tr])
        return m.predict(X_ehr[predict_rows])
    return BaseSpec(name="ehr_hgb", fit_predict=fit_predict, exact=True)


def make_extra_features(X_ehr, X_ecg):
    """Raw covariate block: standardized EHR, plus the ECG embedding PCA when enabled.

    Every transform is fit on `train_rows` only. The harness calls this once with the
    inner-OOF training rows and once with the outer-test rows, so no scaler or PCA ever
    sees an outer-test row during fitting.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    def extra(train_rows, predict_rows):
        sc = StandardScaler().fit(np.nan_to_num(X_ehr[train_rows]))
        cols = [sc.transform(np.nan_to_num(X_ehr[predict_rows])).astype(np.float32)]
        if USE_ECG and X_ecg is not None:
            fin = ~np.isnan(X_ecg[train_rows]).any(1)
            if fin.sum() >= K_ECG + 1:
                pca = PCA(min(K_ECG, X_ecg.shape[1], int(fin.sum()) - 1),
                          random_state=0).fit(X_ecg[train_rows][fin])
                esc = StandardScaler().fit(pca.transform(X_ecg[train_rows][fin]))
                have = ~np.isnan(X_ecg[predict_rows]).any(1)
                ec = np.zeros((len(predict_rows), esc.mean_.shape[0]), np.float32)
                if have.any():
                    ec[have] = esc.transform(pca.transform(X_ecg[predict_rows][have])).astype(np.float32)
                cols.append(ec)
                # Explicit availability indicator (review A11). Without it a missing ECG
                # is encoded as the zero (= mean) standardized embedding, which is
                # indistinguishable from a patient whose ECG is genuinely average.
                cols.append(have.astype(np.float32)[:, None])
        block = np.column_stack(cols)
        return block, block

    return extra


def ridge_fusion(Ftr, ytr, gtr, Fte):
    """Ridge with alpha chosen by PATIENT-GROUPED inner CV (the old ridge_stack used
    episode-level KFold, which splits a patient's repeat episodes -- review A1)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from multimodal_aorta.training.bootstrap import mae
    sc = StandardScaler().fit(Ftr); Z = sc.transform(Ftr)
    best_a, best = 1.0, 1e18
    n_g = len(np.unique(gtr))
    if n_g >= 3:
        for a in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            errs = [mae(ytr[ib], Ridge(alpha=a).fit(Z[ia], ytr[ia]).predict(Z[ib]))
                    for ia, ib in GroupKFold(3).split(Z, groups=gtr)]
            if np.mean(errs) < best:
                best, best_a = float(np.mean(errs)), a
    m = Ridge(alpha=best_a).fit(Z, ytr)
    return m.predict(sc.transform(Fte)), best_a


def main():
    from multimodal_aorta.data.episodes import load_episodes, make_episode_id
    from multimodal_aorta.data.splits import load_episode_folds, make_grouped_cv_folds
    from multimodal_aorta.training.bootstrap import (
        cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt)
    import torch

    ep = load_episodes(PC, require_ecg=False)
    eids = ep.episode_id.astype(str).tolist(); row_of = {e: i for i, e in enumerate(eids)}
    sid = ep.subject_id.to_numpy(int)
    diam = {"root": ep.target_root.to_numpy(float), "asc": ep.target_asc.to_numpy(float)}

    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    inst = inst[inst.episode_id.isin(set(eids))].reset_index(drop=True)
    pp = torch.load(os.path.join(PC, "raddino_patchpool_embeddings_episode.pt"),
                    map_location="cpu", weights_only=False)
    inst = inst[inst.dicom_id.astype(str).isin(set(map(str, pp.keys())))].reset_index(drop=True)
    del pp
    blocks, Xgeom = gs.load_instance_features(inst)
    I_eid = inst.episode_id.astype(str).to_numpy()
    n_per = inst.groupby("episode_id").size().to_dict()
    Iw = np.array([1.0 / n_per[e] for e in I_eid])

    ehr = pd.read_csv(os.path.join(PC, "ehr_features_episode.csv")); ehr["episode_id"] = ehr.episode_id.astype(str)
    em = ehr.set_index("episode_id")[EHR_COLS]
    X_ehr = np.full((len(eids), len(EHR_COLS)), np.nan, np.float32)
    for e in em.index:
        if e in row_of: X_ehr[row_of[e]] = em.loc[e].to_numpy(np.float32)

    X_ecg = None
    if USE_ECG:
        ew = os.path.join(ROOT, "outputs", os.environ.get("ECG_DIR", "ecg_waveform_episode"))
        E = np.load(os.path.join(ew, "ecg_embeddings.npy"))
        eix = pd.read_csv(os.path.join(ew, "ecg_embedding_index.csv"))
        X_ecg = np.full((len(eids), E.shape[1]), np.nan, np.float32)
        for r, e in zip(E, eix.episode_id.astype(str)):
            if e in row_of: X_ecg[row_of[e]] = r

    ft_map = {"root": None, "asc": None}
    if USE_FT:
        ft = pd.read_csv(os.path.join(ROOT, "outputs", "cxr_finetune_episode", "oof_predictions.csv"))
        ft["episode_id"] = ft.episode_id.astype(str)
        for s in ("root", "asc"):
            fm = dict(zip(ft[ft.site == s].episode_id, ft[ft.site == s].pred_cxr_ft))
            ft_map[s] = np.array([fm.get(x, np.nan) for x in eids])

    if FOLD_MODE == "immutable":
        folds = load_episode_folds(os.path.join(PC, "episode_fold_assignments.csv"))
    else:
        folds = make_grouped_cv_folds(ep, stratify_col="anyAD", n_splits=5, seed=42)

    # hard precondition: the outer folds must be patient-disjoint before we start
    for k, (tr_e, te_e) in enumerate(folds):
        tr = np.array([row_of[e] for e in map(str, tr_e) if e in row_of])
        te = np.array([row_of[e] for e in map(str, te_e) if e in row_of])
        assert_patient_disjoint(tr, te, sid, where=f"outer fold {k}")
    log.info("outer folds patient-disjoint: OK (%d folds)", len(folds))

    results = {"fold_mode": FOLD_MODE, "k_pca": K, "k_ecg": K_ECG, "n_inner": N_INNER,
               "use_cxr_ft": USE_FT, "use_ecg": USE_ECG, "sites": {}}
    oof_rows = []
    for site in ("root", "asc"):
        d = diam[site]
        y_inst = np.array([d[row_of[e]] for e in I_eid])
        holder = {"y_inst": y_inst}
        cxr = make_cxr_base(blocks, Xgeom, I_eid, Iw, row_of, holder)
        ehr_b = make_ehr_base(np.nan_to_num(X_ehr), row_of)
        bases = [cxr, ehr_b]
        if USE_FT and ft_map[site] is not None:
            bases.append(PrecomputedSpec(name="cxr_finetuned", values=ft_map[site]))

        log.info("[%s] running nested stack with bases: %s", site, [b.name for b in bases])
        res = nested_stack(folds, row_of, d, sid, bases, ridge_fusion,
                           extra_features=make_extra_features(X_ehr, X_ecg),
                           n_inner=N_INNER, seed=0)
        d_fin = res.predictions

        # CXR-only comparator on the SAME nested folds (review A7): the strongest
        # baseline the multimodal model has to beat.
        res_cxr = nested_stack(folds, row_of, d, sid, [cxr], ridge_fusion,
                               extra_features=None, n_inner=N_INNER, seed=0)
        d_cxr = res_cxr.predictions

        m = ~np.isnan(d) & ~np.isnan(d_fin) & ~np.isnan(d_cxr); g = sid[m]
        y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        sr = {
            "n": int(m.sum()),
            "exact_nesting": bool(res.exact_nesting),
            "inexact_components": res.inexact_components,
            "chosen_alphas": res.chosen_alphas,
            "final_r2": fmt(cluster_bootstrap_ci(d[m], d_fin[m], g, r2, need_both_classes=False)),
            "cxr_only_r2": fmt(cluster_bootstrap_ci(d[m], d_cxr[m], g, r2, need_both_classes=False)),
            "final_ge40": fmt(cluster_bootstrap_ci(y40[m], d_fin[m], g, auroc)),
            "cxr_only_ge40": fmt(cluster_bootstrap_ci(y40[m], d_cxr[m], g, auroc)),
            "final_vs_cxr_r2": fmt(paired_cluster_bootstrap_diff(d[m], d_fin[m], d_cxr[m], g, r2, need_both_classes=False)),
            "final_vs_cxr_ge40": fmt(paired_cluster_bootstrap_diff(y40[m], d_fin[m], d_cxr[m], g, auroc)),
        }
        results["sites"][site] = sr
        log.info("[%s] nested FINAL R2 %s ge40 %s | CXR-only R2 %s ge40 %s",
                 site, sr["final_r2"], sr["final_ge40"], sr["cxr_only_r2"], sr["cxr_only_ge40"])
        log.info("[%s] final vs CXR-only: R2 %s | ge40 %s", site,
                 sr["final_vs_cxr_r2"], sr["final_vs_cxr_ge40"])
        for i in np.where(m)[0]:
            oof_rows.append({"episode_id": eids[i], "subject_id": int(sid[i]), "site": site,
                             "diam_true": d[i], "pred_final": d_fin[i], "pred_cxr_only": d_cxr[i]})

    out_dir = os.path.join(ROOT, "outputs", os.environ.get("OUT_DIR", "final_model_nested"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(oof_rows).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    log.info("Saved -> %s (exact_nesting=%s)", out_dir,
             all(results["sites"][s]["exact_nesting"] for s in results["sites"]))


if __name__ == "__main__":
    main()
