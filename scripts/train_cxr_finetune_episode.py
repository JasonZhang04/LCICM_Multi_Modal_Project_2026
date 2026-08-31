"""
CXR fine-tuning arm (Q4): does end-to-end fine-tuning of RAD-DINO beat the frozen
embedding + PCA pipeline? The PCA sweep showed the frozen representation saturates at
~128 dims, so the only way past that ceiling is to update the backbone.

Partial fine-tuning (peft/LoRA is not installed and the base env is read-only): unfreeze
RAD-DINO's last FT_BLOCKS transformer blocks + a linear [root, asc] head, train end-to-end
on the frontal CXR images. Conservative (most of the ViT stays frozen) to control overfit.

Image-level, nested-in-fold OOF (patient-grouped): for outer fold k, fine-tune on images of
episodes in folds != k (inner patient val split for early stopping), predict fold-k images,
average per episode -> d_cxr_finetuned. Compared downstream to the frozen d_cxr.

Env: SMOKE=1 (1 fold, few steps), SEED, FT_BLOCKS (default 2 => unfreeze blocks 10-11),
BATCH (32), MAX_STEPS. Run: sbatch scripts/slurm_cxr_finetune_episode.sh
Out: outputs/cxr_finetune_episode/{oof_predictions.csv, results.json}
"""
import os, sys, json, logging, time
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "pretrained_checkpoints")
sys.path.insert(0, os.path.join(ROOT, "src"))

SMOKE = os.environ.get("SMOKE", "0") == "1"
SEED = int(os.environ.get("SEED", "42"))
FT_BLOCKS = int(os.environ.get("FT_BLOCKS", "2"))          # unfreeze the last FT_BLOCKS of 12
BATCH = int(os.environ.get("BATCH", "32"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "60" if SMOKE else "6000"))
VAL_EVERY = 20 if SMOKE else 300
PATIENCE = 2 if SMOKE else 6
LR = 1e-4


def main():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from multimodal_aorta.models.cxr_encoder import CXREncoder
    from multimodal_aorta.data.preprocessing import load_cxr
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.episodes import load_episodes, make_episode_id
    from multimodal_aorta.data.splits import load_episode_fold_id_map
    from multimodal_aorta.training.bootstrap import cluster_bootstrap_ci, auroc, r2, fmt

    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    ep = load_episodes(PC, require_ecg=False)
    fold = load_episode_fold_id_map(os.path.join(PC, "episode_fold_assignments.csv"))
    root_of = dict(zip(ep.episode_id.astype(str), ep.target_root))
    asc_of = dict(zip(ep.episode_id.astype(str), ep.target_asc))

    # image table: (episode, dicom) pairs with on-disk frontal JPGs
    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["episode_id"] = [make_episode_id(s, m) for s, m in zip(inst.subject_id, inst.measurement_id)]
    paths = pd.read_csv(os.path.join(PC, "cxr_instances_episode.csv"))
    paths["dicom_id"] = paths.dicom_id.astype(str)
    pmap = dict(zip(paths.dicom_id, paths.cxr_path)); dmap = dict(zip(paths.dicom_id, paths.on_disk))
    inst["dicom_id"] = inst.dicom_id.astype(str)
    inst = inst[inst.dicom_id.map(dmap).fillna(False).astype(bool)].copy()
    inst["cxr_path"] = inst.dicom_id.map(pmap)
    inst["fold_id"] = inst.episode_id.map(fold)
    inst["root"] = inst.episode_id.map(root_of); inst["asc"] = inst.episode_id.map(asc_of)
    inst = inst[inst.fold_id.notna()].reset_index(drop=True)

    # Fast path: read the preprocessed 224x224 tensors from the memmap cache instead of
    # decoding a JPG per image per epoch. Falls back to load_cxr if the cache is absent.
    cache_path = os.path.join(PC, "cxr_image_cache.npy")
    CACHE = crow_of = None
    if os.path.exists(cache_path):
        CACHE = np.load(cache_path, mmap_mode="r")
        cidx = pd.read_csv(os.path.join(PC, "cxr_image_cache_index.csv"))
        crow_of = dict(zip(cidx.dicom_id.astype(str), cidx.row))
        inst["crow"] = inst.dicom_id.map(crow_of)
        inst = inst[inst.crow.notna()].reset_index(drop=True); inst["crow"] = inst.crow.astype(int)
        log.info("using image cache %s", CACHE.shape)
    log.info("image rows %d | episodes %d | device %s | FT_BLOCKS=%d SMOKE=%s",
             len(inst), inst.episode_id.nunique(), dev, FT_BLOCKS, SMOKE)

    class DS(Dataset):
        def __init__(self, df, tmean, tstd, train):
            self.p = df.cxr_path.to_numpy(); self.eid = df.episode_id.to_numpy()
            self.crow = df.crow.to_numpy() if CACHE is not None else None
            self.Y = df[["root", "asc"]].to_numpy(np.float32)
            self.Z = np.nan_to_num((self.Y - tmean) / tstd, nan=0.0)
            self.mask = (~np.isnan(self.Y)).astype(np.float32); self.train = train
        def __len__(self): return len(self.p)
        def __getitem__(self, i):
            if CACHE is not None:
                x = torch.from_numpy(np.asarray(CACHE[self.crow[i]], dtype=np.float32))
                if self.train:
                    x = x + torch.randn_like(x) * 0.02                 # light augmentation
            else:
                x = load_cxr(self.p[i], cfg.data, is_train=self.train)  # (3,H,W)
            return x, torch.from_numpy(self.Z[i]), torch.from_numpy(self.mask[i]), i

    folds = sorted(inst.fold_id.unique())
    if SMOKE: folds = folds[:1]
    oof = {"root": {}, "asc": {}}                                    # episode_id -> list of image preds
    for k in folds:
        tr_df = inst[inst.fold_id != k]; te_df = inst[inst.fold_id == k].reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        vpats = set(rng.choice(tr_df.subject_id.unique(),
                               max(1, int(0.1 * tr_df.subject_id.nunique())), replace=False))
        val_df = tr_df[tr_df.subject_id.isin(vpats)]
        fit_df = tr_df[~tr_df.subject_id.isin(vpats)].reset_index(drop=True)
        tmean = np.nanmean(fit_df[["root", "asc"]].to_numpy(np.float32), 0)
        tstd = np.nanstd(fit_df[["root", "asc"]].to_numpy(np.float32), 0); tstd[tstd == 0] = 1.0

        enc = CXREncoder(model_name=cfg.model.cxr_model_name, freeze_blocks=12 - FT_BLOCKS).to(dev)
        head = torch.nn.Linear(enc.out_dim, 2).to(dev)
        params = [p for p in enc.parameters() if p.requires_grad] + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
        dl = DataLoader(DS(fit_df, tmean, tstd, True), batch_size=BATCH, shuffle=True,
                        num_workers=6, drop_last=True, pin_memory=True)
        vdl = DataLoader(DS(val_df, tmean, tstd, False), batch_size=64, num_workers=6)

        def run_val():
            enc.eval(); head.eval(); tot = w = 0.0
            with torch.no_grad():
                for x, z, m, _ in vdl:
                    p = head(enc(x.to(dev)))
                    tot += (((p - z.to(dev)) ** 2) * m.to(dev)).sum().item(); w += m.sum().item()
            enc.train(); head.train(); return tot / max(w, 1)

        best, best_state, bad, step, t0, stop = 1e9, None, 0, 0, time.time(), False
        enc.train(); head.train()
        while not stop:
            for x, z, m, _ in dl:
                x, z, m = x.to(dev), z.to(dev), m.to(dev)
                loss = (((head(enc(x)) - z) ** 2) * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
                if step % VAL_EVERY == 0:
                    vl = run_val()
                    if vl < best - 1e-4:
                        best, bad = vl, 0
                        best_state = ({kk: v.detach().cpu().clone() for kk, v in enc.state_dict().items()},
                                      {kk: v.detach().cpu().clone() for kk, v in head.state_dict().items()})
                    else:
                        bad += 1
                    log.info("  fold %d step %d val %.4f (best %.4f bad %d) %.0fs", k, step, vl, best, bad, time.time() - t0)
                if bad >= PATIENCE or step >= MAX_STEPS:
                    stop = True; break
        if best_state:
            enc.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
        # predict fold-k images, collect per episode
        enc.eval(); head.eval()
        pdl = DataLoader(DS(te_df, tmean, tstd, False), batch_size=64, num_workers=6)
        with torch.no_grad():
            base = 0
            for x, z, m, idx in pdl:
                P = (head(enc(x.to(dev))).cpu().numpy() * tstd + tmean)
                for r in range(len(idx)):
                    e = te_df.episode_id.iloc[int(idx[r])]
                    oof["root"].setdefault(e, []).append(P[r, 0]); oof["asc"].setdefault(e, []).append(P[r, 1])
        log.info("fold %d done (fit imgs %d, test imgs %d)", k, len(fit_df), len(te_df))

    # aggregate image preds -> episode, evaluate + save
    out_dir = os.path.join(ROOT, "outputs", "cxr_finetune_episode" + ("_smoke" if SMOKE else ""))
    os.makedirs(out_dir, exist_ok=True)
    rows, res = [], {"seed": SEED, "ft_blocks": FT_BLOCKS, "smoke": SMOKE, "sites": {}}
    sid_of = dict(zip(ep.episode_id.astype(str), ep.subject_id))
    for site, truth in (("root", root_of), ("asc", asc_of)):
        eids = list(oof[site].keys())
        p = np.array([np.mean(oof[site][e]) for e in eids]); d = np.array([truth[e] for e in eids])
        g = np.array([sid_of[e] for e in eids]); m = ~np.isnan(d) & ~np.isnan(p)
        if m.sum() > 10 and not SMOKE:
            y40 = (d[m] >= 4.0).astype(float)
            res["sites"][site] = {"n": int(m.sum()),
                                  "cxr_ft_r2": fmt(cluster_bootstrap_ci(d[m], p[m], g[m], r2, need_both_classes=False)),
                                  "cxr_ft_ge40": fmt(cluster_bootstrap_ci(y40, p[m], g[m], auroc))}
            log.info("[%s] CXR FINE-TUNED alone: R2 %s | ge40 %s", site,
                     res["sites"][site]["cxr_ft_r2"], res["sites"][site]["cxr_ft_ge40"])
        for e in eids:
            if not np.isnan(truth[e]):
                rows.append({"episode_id": e, "subject_id": int(sid_of[e]), "site": site,
                             "diam_true": truth[e], "pred_cxr_ft": float(np.mean(oof[site][e]))})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
