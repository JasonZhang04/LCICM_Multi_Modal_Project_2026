"""
Multimodal fusion experiment: which fusion architecture, and does the raw-waveform ECG
add anything over CXR+EHR at the FEATURE level (not the scalar bottleneck)?

Per-episode modality feature blocks (all nested, per-fold, patient-grouped):
  X_cxr  = PCA-32 of episode-mean [cls | aorta-pool | heart-pool] + 17 geometry features
  X_ehr  = 12 tabular EHR features
  X_ecg  = PCA-32 of the ECG 1D-ResNet PENULTIMATE embedding (from train_ecg_waveform_episode)

Fusion arms compared on identical folds, scored by predicted diameter (regression-derived):
  LATE   ridge over per-modality nested-OOF diameter scalars  [d_cxr, d_ehr, d_ecg]
  EARLY  one HistGradientBoosting over the concatenated feature blocks
  COOP   cooperative learning (Ding & Tibshirani agreement penalty) over the blocks, rho
         tuned by inner CV — rho=0 reduces to linear early fusion, rho->inf to late fusion.

Each arm is run with and without the ECG block, so the paired increment (cluster
bootstrap on ECG-having episodes) is the feature-level waveform increment. See
[[fusion-plan-2026-08-30]] for the design rationale (redundant modalities => simple
fusion expected to win; COOP is the principled "let them connect" test).

Env: SEEDS (default 1,2,3), K_PCA (32). Run: sbatch scripts/slurm_fusion_episode.sh
Out: outputs/fusion_episode/results.json
"""
import os, sys, json, logging
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

K = int(os.environ.get("K_PCA", "32"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
RHO_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
LAM_GRID = [1.0, 3.0, 10.0, 30.0]
import train_modality_value_episode as mv           # build_episode_cxr, _hgb, GEOM, EHR_COLS


def _pca_std(train, test, k):
    """Fit PCA(k)+standardize on the PRESENT (non-NaN) train rows only, apply to both;
    rows with any NaN feature (e.g. episodes lacking an ECG embedding) map to the
    standardized mean (0). Leakage-safe: the transform is fit on training rows only."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    fin = ~np.isnan(train).any(1)
    Xtr = train[fin]
    pca = PCA(min(k, Xtr.shape[1], max(Xtr.shape[0] - 1, 1)), random_state=0).fit(Xtr)
    sc = StandardScaler().fit(pca.transform(Xtr))

    def emb(X):
        out = np.zeros((X.shape[0], sc.mean_.shape[0]), np.float32)   # missing -> mean 0
        m = ~np.isnan(X).any(1)
        if m.any():
            out[m] = sc.transform(pca.transform(X[m])).astype(np.float32)
        return out
    return emb(train), emb(test)


def _std(train, test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.nan_to_num(train))
    return (sc.transform(np.nan_to_num(train)).astype(np.float32),
            sc.transform(np.nan_to_num(test)).astype(np.float32))


def fold_blocks(cxr_views, GEOM_ep, X_ehr, X_ecg, trm, te):
    """Per-fold standardized modality blocks for train(trm)/test(te) row-index arrays."""
    parts_tr, parts_te = [], []
    for V in cxr_views:
        a, b = _pca_std(V[trm], V[te], K); parts_tr.append(a); parts_te.append(b)
    g_tr, g_te = _std(GEOM_ep[trm], GEOM_ep[te]); parts_tr.append(g_tr); parts_te.append(g_te)
    Xcxr_tr, Xcxr_te = np.hstack(parts_tr), np.hstack(parts_te)
    Xehr_tr, Xehr_te = _std(X_ehr[trm], X_ehr[te])
    Xecg_tr, Xecg_te = _pca_std(X_ecg[trm], X_ecg[te], K)
    return (Xcxr_tr, Xehr_tr, Xecg_tr), (Xcxr_te, Xehr_te, Xecg_te)


def coop_predict(blocks_tr, ytr, blocks_te, rho, lam):
    """Cooperative learning (Ding/Tibshirani agreement penalty) via the augmented-data
    trick: prediction block [X1..XK] with target y, plus for each pair (i,j) a row block
    with +sqrt(rho)Xi, -sqrt(rho)Xj, target 0; solved by a single ridge. rho=0 => early
    fusion (concatenated ridge); large rho forces the per-view predictions to agree."""
    from sklearn.linear_model import Ridge
    K_ = len(blocks_tr); n = blocks_tr[0].shape[0]; dims = [b.shape[1] for b in blocks_tr]
    # Center y and drop the intercept: fit_intercept=True would fit the intercept over the
    # zero-target agreement rows too, dragging it toward 0 as rho grows. Since every block is
    # standardized (mean~0), the intercept is the only carrier of y's mean, so that would
    # collapse predictions for rho>0. Centering y + fit_intercept=False is the correct form.
    ybar = float(ytr.mean())
    Xcat_tr = np.hstack(blocks_tr)
    A_rows, b_rows = [Xcat_tr], [ytr - ybar]
    if rho > 0:
        sr = np.sqrt(rho)
        for i in range(K_):
            for j in range(i + 1, K_):
                blk = np.zeros((n, sum(dims)), np.float32)
                ci, cj = sum(dims[:i]), sum(dims[:j])
                blk[:, ci:ci + dims[i]] = sr * blocks_tr[i]
                blk[:, cj:cj + dims[j]] = -sr * blocks_tr[j]
                A_rows.append(blk); b_rows.append(np.zeros(n, np.float32))
    m = Ridge(alpha=lam, fit_intercept=False).fit(np.vstack(A_rows), np.concatenate(b_rows))
    return m.predict(np.hstack(blocks_te)) + ybar


def gbdt_scalar_oof(folds, y, block_fn, row_of, use_blocks):
    """Nested-OOF per-modality/early diameter regressor (HGB on the selected blocks)."""
    from sklearn.base import clone
    out = np.full(len(y), np.nan); base = mv._hgb()
    for tr_eids, te_eids in folds:
        tr = np.array([row_of[e] for e in map(str, tr_eids) if e in row_of])
        te = np.array([row_of[e] for e in map(str, te_eids) if e in row_of])
        trm = tr[~np.isnan(y[tr])]
        if len(trm) < 50:
            continue
        Btr, Bte = block_fn(trm, te)
        Xtr = np.hstack([Btr[i] for i in use_blocks]); Xte = np.hstack([Bte[i] for i in use_blocks])
        m = clone(base).fit(Xtr, y[trm])
        out[te] = m.predict(Xte)
    return out


def late_ridge_oof(folds, y, d_mods, row_of):
    """Ridge meta-learner over per-modality scalar OOF predictions."""
    from sklearn.linear_model import Ridge
    out = np.full(len(y), np.nan)
    F = np.column_stack(d_mods)
    for tr_eids, te_eids in folds:
        tr = np.array([row_of[e] for e in map(str, tr_eids) if e in row_of])
        te = np.array([row_of[e] for e in map(str, te_eids) if e in row_of])
        trm = tr[~np.isnan(y[tr]) & ~np.isnan(F[tr]).any(1)]
        tem = te[~np.isnan(F[te]).any(1)]
        if len(trm) < 50 or not len(tem):
            continue
        out[tem] = Ridge(alpha=10.0).fit(F[trm], y[trm]).predict(F[tem])
    return out


def coop_oof(folds, y, block_fn, row_of, use_blocks, sid):
    """Cooperative-learning OOF with inner-CV rho/lambda selection. Inner CV is
    PATIENT-GROUPED (GroupKFold on subject_id) so the selected (rho, lambda) is not
    optimistically chosen on a patient's own straddling episodes."""
    from sklearn.model_selection import GroupKFold
    from multimodal_aorta.training.bootstrap import r2
    out = np.full(len(y), np.nan); chosen = []
    for tr_eids, te_eids in folds:
        tr = np.array([row_of[e] for e in map(str, tr_eids) if e in row_of])
        te = np.array([row_of[e] for e in map(str, te_eids) if e in row_of])
        trm = tr[~np.isnan(y[tr])]
        if len(trm) < 100:
            continue
        Btr, Bte = block_fn(trm, te)
        Btr = [Btr[i] for i in use_blocks]; Bte = [Bte[i] for i in use_blocks]
        ytr = y[trm]; gtr = sid[trm]
        best, best_cfg = -1e9, (0.0, 10.0)
        for rho in RHO_GRID:
            for lam in LAM_GRID:
                sc = []
                for a, b in GroupKFold(3).split(trm, groups=gtr):
                    p = coop_predict([bl[a] for bl in Btr], ytr[a], [bl[b] for bl in Btr], rho, lam)
                    sc.append(r2(ytr[b], p))
                s = float(np.mean(sc))
                if s > best:
                    best, best_cfg = s, (rho, lam)
        out[te] = coop_predict(Btr, ytr, Bte, *best_cfg)
        chosen.append(best_cfg[0])
    return out, chosen


def main():
    from multimodal_aorta.data.episodes import load_episodes
    from multimodal_aorta.data.splits import make_grouped_cv_folds
    from multimodal_aorta.training.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt

    ep = load_episodes(PC, require_ecg=False)
    eids = ep.episode_id.astype(str).tolist(); row_of = {e: i for i, e in enumerate(eids)}
    sid = ep.subject_id.to_numpy(int)
    diam = {"root": ep.target_root.to_numpy(float), "asc": ep.target_asc.to_numpy(float)}
    (CLS, AO, HR), GEOM_ep = mv.build_episode_cxr(eids, row_of)
    cxr_views = [CLS, AO, HR]

    ehr = pd.read_csv(os.path.join(PC, "ehr_features_episode.csv")); ehr["episode_id"] = ehr.episode_id.astype(str)
    em = ehr.set_index("episode_id")[mv.EHR_COLS]
    X_ehr = np.full((len(eids), len(mv.EHR_COLS)), np.nan, np.float32)
    for e in em.index:
        if e in row_of: X_ehr[row_of[e]] = em.loc[e].to_numpy(np.float32)

    # ECG penultimate embedding (nested OOF) from the waveform arm
    ew = os.path.join(ROOT, "outputs", "ecg_waveform_episode")
    E = np.load(os.path.join(ew, "ecg_embeddings.npy"))
    eix = pd.read_csv(os.path.join(ew, "ecg_embedding_index.csv"))
    X_ecg = np.full((len(eids), E.shape[1]), np.nan, np.float32); has_ecg = np.zeros(len(eids), bool)
    for r, e in zip(E, eix.episode_id.astype(str)):
        if e in row_of: X_ecg[row_of[e]] = r; has_ecg[row_of[e]] = True
    log.info("episodes %d | ECG-embedding present %d | emb dim %d", len(eids), int(has_ecg.sum()), E.shape[1])

    # blocks index: 0=CXR, 1=EHR, 2=ECG
    CXR_EHR, CXR_EHR_ECG = [0, 1], [0, 1, 2]
    results = {"seeds": SEEDS, "n_ecg": int(has_ecg.sum()), "sites": {}}
    for site in ("root", "asc"):
        d = diam[site]; y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        acc = {k: [] for k in ["late2", "late3", "early2", "early3", "coop3"]}
        coop_rhos = []
        for seed in SEEDS:
            folds = make_grouped_cv_folds(ep, stratify_col="anyAD", n_splits=5, seed=seed)
            block_fn = lambda trm, te: fold_blocks(cxr_views, GEOM_ep, X_ehr, X_ecg, trm, te)
            # per-modality scalars for late fusion
            d_cxr = gbdt_scalar_oof(folds, d, block_fn, row_of, [0])
            d_ehr = gbdt_scalar_oof(folds, d, block_fn, row_of, [1])
            d_ecg = gbdt_scalar_oof(folds, d, block_fn, row_of, [2])
            acc["late2"].append(late_ridge_oof(folds, d, [d_cxr, d_ehr], row_of))
            acc["late3"].append(late_ridge_oof(folds, d, [d_cxr, d_ehr, d_ecg], row_of))
            acc["early2"].append(gbdt_scalar_oof(folds, d, block_fn, row_of, CXR_EHR))
            acc["early3"].append(gbdt_scalar_oof(folds, d, block_fn, row_of, CXR_EHR_ECG))
            c3, rhos = coop_oof(folds, d, block_fn, row_of, CXR_EHR_ECG, sid)
            acc["coop3"].append(c3); coop_rhos += rhos
        P = {k: np.nanmean(np.column_stack(v), 1) for k, v in acc.items()}
        # evaluate ECG-including arms on ECG-having episodes; 2-mod on all
        mall = ~np.isnan(d)
        mecg = has_ecg & ~np.isnan(d)
        def R2(p, m): return fmt(cluster_bootstrap_ci(d[m], p[m], sid[m], r2, need_both_classes=False))
        def AU(p, m): return fmt(cluster_bootstrap_ci(y40[m], p[m], sid[m], auroc))
        def incr(pa, pb, m, fn, y=None, nb=False):
            yy = y if y is not None else d
            return fmt(paired_cluster_bootstrap_diff(yy[m], pa[m], pb[m], sid[m], fn, need_both_classes=nb))
        sr = {
            "late2_r2": R2(P["late2"], mall), "early2_r2": R2(P["early2"], mall),
            "late3_r2": R2(P["late3"], mecg), "early3_r2": R2(P["early3"], mecg), "coop3_r2": R2(P["coop3"], mecg),
            "late2_ge40": AU(P["late2"], mall), "early2_ge40": AU(P["early2"], mall),
            "early3_ge40": AU(P["early3"], mecg), "coop3_ge40": AU(P["coop3"], mecg),
            # fusion architecture: early vs late (on ECG-having, 3-modality)
            "early_vs_late_r2": incr(P["early3"], P["late3"], mecg, r2),
            "coop_vs_early_r2": incr(P["coop3"], P["early3"], mecg, r2),
            "coop_vs_late_r2": incr(P["coop3"], P["late3"], mecg, r2),
            # ECG feature-level increment (3-mod vs 2-mod, same episodes, early fusion)
            "ecg_incr_early_r2": incr(P["early3"], P["early2"], mecg, r2),
            "ecg_incr_early_ge40": incr(P["early3"], P["early2"], mecg, auroc, y=y40, nb=True),
            "coop_rho_median": float(np.median(coop_rhos)) if coop_rhos else None,
            "coop_rho_counts": {str(r_): int(np.sum(np.array(coop_rhos) == r_)) for r_ in RHO_GRID},
        }
        results["sites"][site] = sr
        log.info("[%s] R2  late2 %s early2 %s | late3 %s early3 %s coop3 %s", site,
                 sr["late2_r2"], sr["early2_r2"], sr["late3_r2"], sr["early3_r2"], sr["coop3_r2"])
        log.info("[%s]   early-vs-late %s | coop-vs-early %s | coop-vs-late %s | coop rho median %s", site,
                 sr["early_vs_late_r2"], sr["coop_vs_early_r2"], sr["coop_vs_late_r2"], sr["coop_rho_median"])
        log.info("[%s]   ECG feature-level increment: R2 %s | ge40 %s", site,
                 sr["ecg_incr_early_r2"], sr["ecg_incr_early_ge40"])

    out_dir = os.path.join(ROOT, "outputs", "fusion_episode"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
