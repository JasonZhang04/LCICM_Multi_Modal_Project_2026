"""
Train the raw-waveform ECG arm (ECGResNet) with nested-in-fold OOF, so its per-episode
predicted diameter slots into the multimodal stack leakage-free.

For each immutable outer fold k: train ECGResNet on episodes in folds != k (with a small
inner validation split for early stopping), predict fold k. Targets [root, asc, hr] are
standardized on the training fold; predictions are inverted back to cm. HR is auxiliary.

Recipe (adapted from ECGAI-TAA): batch 64, Adam, 2000-minibatch warmup then flat 1e-3,
per-lead Gaussian noise augmentation, validate periodically and keep the best weights.
Reads the cached ecg_waveforms.npy so I/O is not the bottleneck.

Env: SMOKE=1 (1 fold, few steps, to check the pipeline on GPU); SEED, MAX_STEPS.
Out: outputs/ecg_waveform_episode/{oof_predictions.csv, results.json}
Run: sbatch scripts/slurm_ecg_waveform_episode.sh
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
MAX_STEPS = int(os.environ.get("MAX_STEPS", "200" if SMOKE else "12000"))
WARMUP = 200 if SMOKE else 2000
VAL_EVERY = 50 if SMOKE else 400
PATIENCE = 3 if SMOKE else 8
BATCH = 64
LR = 1e-3
TARGETS = ["root_cm", "asc_cm", "hr"]


def main():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from multimodal_aorta.models.ecg_resnet import ECGResNet
    from multimodal_aorta.training.bootstrap import cluster_bootstrap_ci, auroc, r2, fmt

    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coh = pd.read_csv(os.path.join(PC, "ecg_waveform_cohort.csv"))
    idx = pd.read_csv(os.path.join(PC, "ecg_waveform_index.csv"))
    row_of_study = dict(zip(idx.study_id, idx.row))
    coh = coh[coh.study_id.isin(row_of_study)].reset_index(drop=True)
    coh["wrow"] = coh.study_id.map(row_of_study).astype(int)
    waves = np.load(os.path.join(PC, "ecg_waveforms.npy"), mmap_mode="r")   # (N,12,5000) f16
    log.info("cohort %d episodes | waveforms %s | device %s | SMOKE=%s", len(coh), waves.shape, dev, SMOKE)

    class DS(Dataset):
        def __init__(self, df, tmean, tstd, train):
            self.wrow = df.wrow.to_numpy()
            self.Y = df[TARGETS].to_numpy(np.float32)
            self.Z = (self.Y - tmean) / tstd
            self.mask = (~np.isnan(self.Y)).astype(np.float32)
            self.Z = np.nan_to_num(self.Z, nan=0.0)
            self.train = train

        def __len__(self): return len(self.wrow)

        def __getitem__(self, i):
            x = np.asarray(waves[self.wrow[i]], dtype=np.float32)   # (12,5000)
            if self.train:
                x = x + np.random.randn(*x.shape).astype(np.float32) * 0.01   # per-lead noise
            return torch.from_numpy(x), torch.from_numpy(self.Z[i]), torch.from_numpy(self.mask[i])

    folds = sorted(coh.fold_id.unique())
    if SMOKE:
        folds = folds[:1]
    oof = {t: np.full(len(coh), np.nan) for t in ("root_cm", "asc_cm")}
    for k in folds:
        tr_df = coh[coh.fold_id != k]
        te_df = coh[coh.fold_id == k]
        # inner val split by patient for early stopping
        rng = np.random.default_rng(SEED)
        vpats = set(rng.choice(tr_df.subject_id.unique(),
                               max(1, int(0.1 * tr_df.subject_id.nunique())), replace=False))
        val_df = tr_df[tr_df.subject_id.isin(vpats)]
        fit_df = tr_df[~tr_df.subject_id.isin(vpats)]
        tmean = np.nanmean(fit_df[TARGETS].to_numpy(np.float32), 0)
        tstd = np.nanstd(fit_df[TARGETS].to_numpy(np.float32), 0); tstd[tstd == 0] = 1.0

        dl = DataLoader(DS(fit_df, tmean, tstd, True), batch_size=BATCH, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True)
        vdl = DataLoader(DS(val_df, tmean, tstd, False), batch_size=128, num_workers=4)
        model = ECGResNet().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

        def val_loss():
            model.eval(); tot = w = 0.0
            with torch.no_grad():
                for x, z, m in vdl:
                    p = model(x.to(dev))
                    l = (((p - z.to(dev)) ** 2) * m.to(dev))[:, :2].sum().item()  # diameters only
                    tot += l; w += m[:, :2].sum().item()
            model.train(); return tot / max(w, 1)

        best, best_state, bad, step = 1e9, None, 0, 0
        t0 = time.time(); stop = False
        while not stop:
            for x, z, m in dl:
                lr = LR * min(1.0, (step + 1) / WARMUP)
                for g in opt.param_groups: g["lr"] = lr
                x, z, m = x.to(dev), z.to(dev), m.to(dev)
                p = model(x)
                loss = (((p - z) ** 2) * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
                step += 1
                if step % VAL_EVERY == 0:
                    vl = val_loss()
                    if vl < best - 1e-4:
                        best, best_state, bad = vl, {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}, 0
                    else:
                        bad += 1
                    log.info("  fold %d step %d val %.4f (best %.4f, bad %d) %.0fs", k, step, vl, best, bad, time.time() - t0)
                if bad >= PATIENCE or step >= MAX_STEPS:
                    stop = True; break

        if best_state: model.load_state_dict(best_state)
        # predict fold k
        model.eval()
        pdl = DataLoader(DS(te_df, tmean, tstd, False), batch_size=128, num_workers=4)
        preds = []
        with torch.no_grad():
            for x, z, m in pdl:
                preds.append(model(x.to(dev)).cpu().numpy())
        P = np.concatenate(preds, 0) * tstd + tmean   # back to cm/bpm
        for j, t in enumerate(("root_cm", "asc_cm")):
            oof[t][te_df.index.to_numpy()] = P[:, j]
        log.info("fold %d done (train %d val %d test %d)", k, len(fit_df), len(val_df), len(te_df))

    # save OOF + standalone metrics
    out_dir = os.path.join(ROOT, "outputs", "ecg_waveform_episode" + ("_smoke" if SMOKE else ""))
    os.makedirs(out_dir, exist_ok=True)
    rows, res = [], {"seed": SEED, "smoke": SMOKE, "sites": {}}
    for site, col in (("root", "root_cm"), ("asc", "asc_cm")):
        d = coh[col].to_numpy(float); p = oof[col]; g = coh.subject_id.to_numpy(int)
        m = ~np.isnan(d) & ~np.isnan(p)
        if m.sum() > 10 and not SMOKE:
            y40 = (d[m] >= 4.0).astype(float)
            res["sites"][site] = {
                "n": int(m.sum()),
                "waveform_alone_r2": fmt(cluster_bootstrap_ci(d[m], p[m], g[m], r2, need_both_classes=False)),
                "waveform_alone_ge40": fmt(cluster_bootstrap_ci(y40, p[m], g[m], auroc)),
            }
            log.info("[%s] ECG-WAVEFORM alone: R2 %s | ge40 %s", site,
                     res["sites"][site]["waveform_alone_r2"], res["sites"][site]["waveform_alone_ge40"])
        for i in np.where(m)[0]:
            rows.append({"episode_id": coh.episode_id.iloc[i], "subject_id": int(g[i]),
                         "site": site, "diam_true": d[i], "pred_ecg_waveform": p[i]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    log.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
