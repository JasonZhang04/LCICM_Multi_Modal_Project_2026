"""
Cross-attention fusion arm (Q3): directly test whether letting the modalities attend to
each other beats the regularized LINEAR EARLY fusion that won the fusion study. The
cooperative-learning rho=0 result and the redundancy diagnostics predict it will NOT help,
but we run it so the paper can state it was tried, not just predicted.

Per episode, three modality tokens (CXR PCA-views+geometry, EHR tabular, ECG-embedding PCA)
are each projected to a common width, passed through a small, heavily-regularized
cross-attention (transformer-encoder) block, mean-pooled, and mapped to [root, asc]
diameter. Nested patient-grouped OOF with an inner patient val split for early stopping.
Compared head-to-head, on the SAME folds/episodes, against linear early fusion (coop rho=0).

Env: SEEDS (default 1,2,3), K_PCA (32). Run: sbatch scripts/slurm_crossattn_episode.sh
Out: outputs/crossattn_episode/results.json
"""
import os, sys, json, logging, time
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
K = int(os.environ.get("K_PCA", "32"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
import train_fusion_episode as F           # fold_blocks, coop_predict, build helpers via mv
import train_modality_value_episode as mv


def make_model(dims, d=48, heads=3, drop=0.3):
    import torch, torch.nn as nn
    class CA(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleList([nn.Linear(x, d) for x in dims])
            self.enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=drop,
                                                  batch_first=True, activation="gelu")
            self.drop = nn.Dropout(drop); self.head = nn.Linear(d, 2)
        def forward(self, xs):
            toks = torch.stack([self.proj[i](xs[i]) for i in range(len(xs))], dim=1)   # (B,3,d)
            h = self.enc(toks).mean(1)                                                  # (B,d)
            return self.head(self.drop(h))
    return CA()


def train_ca(Btr, ytr, gtr, Bte, tmean, tstd, seed):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import GroupKFold
    torch.manual_seed(seed)
    dims = [b.shape[1] for b in Btr]
    Z = ((ytr - tmean) / tstd).astype(np.float32); mask = (~np.isnan(ytr)).astype(np.float32)
    Z = np.nan_to_num(Z)
    # inner patient val split
    (ia, ib) = next(GroupKFold(5).split(np.arange(len(ytr)), groups=gtr))
    def tens(idx): return [torch.tensor(b[idx]) for b in Btr]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(dims).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    Xa = [torch.tensor(b[ia]).to(dev) for b in Btr]; Za = torch.tensor(Z[ia]).to(dev); Ma = torch.tensor(mask[ia]).to(dev)
    Xv = [torch.tensor(b[ib]).to(dev) for b in Btr]; Zv = torch.tensor(Z[ib]).to(dev); Mv = torch.tensor(mask[ib]).to(dev)
    best, best_state, bad = 1e9, None, 0
    for epoch in range(200):
        model.train()
        perm = torch.randperm(len(ia))
        for s in range(0, len(ia), 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            p = model([x[b] for x in Xa])
            loss = (((p - Za[b]) ** 2) * Ma[b]).sum() / Ma[b].sum().clamp(min=1)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = ((((model(Xv) - Zv) ** 2) * Mv).sum() / Mv.sum().clamp(min=1)).item()
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= 8:
            break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        P = model([torch.tensor(b).to(dev) for b in Bte]).cpu().numpy()
    return P * tstd + tmean       # (n_te, 2) back to cm


def main():
    from multimodal_aorta.data.episodes import load_episodes
    from multimodal_aorta.data.splits import make_grouped_cv_folds
    from multimodal_aorta.training.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap_diff, auroc, r2, fmt
    import torch

    ep = load_episodes(PC, require_ecg=False)
    eids = ep.episode_id.astype(str).tolist(); row_of = {e: i for i, e in enumerate(eids)}
    sid = ep.subject_id.to_numpy(int)
    D = np.column_stack([ep.target_root.to_numpy(float), ep.target_asc.to_numpy(float)])
    (CLS, AO, HR), GEOM_ep = mv.build_episode_cxr(eids, row_of); cxr_views = [CLS, AO, HR]
    ehr = pd.read_csv(os.path.join(PC, "ehr_features_episode.csv")); ehr["episode_id"] = ehr.episode_id.astype(str)
    em = ehr.set_index("episode_id")[mv.EHR_COLS]; X_ehr = np.full((len(eids), len(mv.EHR_COLS)), np.nan, np.float32)
    for e in em.index:
        if e in row_of: X_ehr[row_of[e]] = em.loc[e].to_numpy(np.float32)
    E = np.load(os.path.join(ROOT, "outputs", "ecg_waveform_episode", "ecg_embeddings.npy"))
    eix = pd.read_csv(os.path.join(ROOT, "outputs", "ecg_waveform_episode", "ecg_embedding_index.csv"))
    X_ecg = np.full((len(eids), E.shape[1]), np.nan, np.float32); has_ecg = np.zeros(len(eids), bool)
    for r, e in zip(E, eix.episode_id.astype(str)):
        if e in row_of: X_ecg[row_of[e]] = r; has_ecg[row_of[e]] = True

    results = {"seeds": SEEDS, "sites": {}}
    CA = {0: [], 1: []}; LE = {0: [], 1: []}
    for seed in SEEDS:
        folds = make_grouped_cv_folds(ep, stratify_col="anyAD", n_splits=5, seed=seed)
        ca = np.full((len(eids), 2), np.nan); le = np.full((len(eids), 2), np.nan)
        for tr_eids, te_eids in folds:
            tr = np.array([row_of[e] for e in map(str, tr_eids) if e in row_of])
            te = np.array([row_of[e] for e in map(str, te_eids) if e in row_of])
            trm = tr[~np.isnan(D[tr]).all(1)]
            Btr, Bte = F.fold_blocks(cxr_views, GEOM_ep, X_ehr, X_ecg, trm, te)
            # cross-attention (2-output, per site handled by mask); train on episodes with any label
            tmean = np.nanmean(D[trm], 0); tstd = np.nanstd(D[trm], 0); tstd[tstd == 0] = 1
            P = train_ca(list(Btr), D[trm], sid[trm], list(Bte), tmean, tstd, seed)
            ca[te] = P
            # linear early fusion (coop rho=0) per site as the comparator
            for j in range(2):
                ok = ~np.isnan(D[trm, j])
                le[te, j] = F.coop_predict([b[ok] for b in Btr], D[trm, j][ok], list(Bte), 0.0, 10.0)
        for j in range(2): CA[j].append(ca[:, j]); LE[j].append(le[:, j])
    for j, site in ((0, "root"), (1, "asc")):
        d = D[:, j]; y40 = np.where(np.isnan(d), np.nan, (d >= 4.0).astype(float))
        ca = np.nanmean(np.column_stack(CA[j]), 1); le = np.nanmean(np.column_stack(LE[j]), 1)
        m = has_ecg & ~np.isnan(d) & ~np.isnan(ca) & ~np.isnan(le); g = sid[m]
        sr = {
            "n": int(m.sum()),
            "crossattn_r2": fmt(cluster_bootstrap_ci(d[m], ca[m], g, r2, need_both_classes=False)),
            "linear_early_r2": fmt(cluster_bootstrap_ci(d[m], le[m], g, r2, need_both_classes=False)),
            "crossattn_vs_linear_r2": fmt(paired_cluster_bootstrap_diff(d[m], ca[m], le[m], g, r2, need_both_classes=False)),
            "crossattn_ge40": fmt(cluster_bootstrap_ci(y40[m], ca[m], g, auroc)),
            "linear_early_ge40": fmt(cluster_bootstrap_ci(y40[m], le[m], g, auroc)),
            "crossattn_vs_linear_ge40": fmt(paired_cluster_bootstrap_diff(y40[m], ca[m], le[m], g, auroc)),
        }
        results["sites"][site] = sr
        log.info("[%s] cross-attn R2 %s | linear-early R2 %s | CA vs linear: R2 %s ge40 %s", site,
                 sr["crossattn_r2"], sr["linear_early_r2"], sr["crossattn_vs_linear_r2"], sr["crossattn_vs_linear_ge40"])
    out_dir = os.path.join(ROOT, "outputs", "crossattn_episode"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
