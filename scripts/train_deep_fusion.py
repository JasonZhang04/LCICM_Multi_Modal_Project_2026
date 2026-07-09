"""
v3 Milestone 4 — deep early-fusion model on the triple cohort.

Frozen [PCLR(320) | RAD-DINO(768) | EHR(12)] -> V3FusionModel -> per-site CORAL
ordinal (>=4.0/>=4.5/>=5.0) + z-scored diameter regression. Stratified 5-fold CV
with out-of-fold predictions, an inner train/val split per fold for early
stopping, and the SAME metrics as the GBDT baseline for a head-to-head compare.

Run via SLURM: sbatch scripts/slurm_deep_fusion.sh
Outputs: console log + outputs/deep_fusion/results.json
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SEED = 42
EHR_NUM_IDX = [0, 2, 3, 4, 5, 6, 7]   # age,height,weight,bmi,bsa,sbp,dbp (rest are binary)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def assemble(cohort, pclr, raddino, ehr, ehr_cols):
    rows, ecg, cxr, eh = [], [], [], []
    ehr_dim = len(ehr_cols)
    for r in cohort.itertuples(index=False):
        sid = int(r.subject_id)
        if sid not in pclr or sid not in raddino:
            continue
        e = pclr[sid].numpy() if isinstance(pclr[sid], torch.Tensor) else np.asarray(pclr[sid])
        c = raddino[sid].numpy() if isinstance(raddino[sid], torch.Tensor) else np.asarray(raddino[sid])
        ecg.append(e.astype(np.float32)); cxr.append(c.astype(np.float32))
        eh.append(ehr.get(sid, np.full(ehr_dim, np.nan, np.float32)))
        rows.append(r)
    df = pd.DataFrame(rows).reset_index(drop=True)
    return df, np.vstack(ecg), np.vstack(cxr), np.vstack(eh).astype(np.float32)


def fit_ehr_scaler(ehr_train):
    med = np.nanmedian(ehr_train[:, EHR_NUM_IDX], axis=0)
    filled = ehr_train[:, EHR_NUM_IDX].copy()
    ii = np.where(np.isnan(filled)); filled[ii] = np.take(med, ii[1])
    return med, filled.mean(0), filled.std(0) + 1e-6


def apply_ehr_scaler(ehr, scaler):
    med, mean, std = scaler
    out = ehr.copy()
    block = out[:, EHR_NUM_IDX]
    ii = np.where(np.isnan(block)); block[ii] = np.take(med, ii[1])
    out[:, EHR_NUM_IDX] = (block - mean) / std
    return np.nan_to_num(out, nan=0.0)


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------

def train_fold(model, opt, tr, va, tensors, device, max_epochs=200, patience=20, lambda_reg=0.3):
    from multimodal_aorta.training.ordinal import coral_loss, masked_mse, cum_probs, binary_metrics
    ecg, cxr, ehr, cum, zdiam = tensors
    best_score, best_state, no_imp = -1e9, None, 0
    history = []   # per-epoch (train_loss, val_loss, val_auroc) for curves

    for epoch in range(max_epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=tr.device)]
        ep_loss, nb = 0.0, 0
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            ol, rg = model(ecg[b], cxr[b], ehr[b])
            loss = coral_loss(ol, cum[b]) + lambda_reg * masked_mse(rg, zdiam[b])
            loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        train_loss = ep_loss / max(nb, 1)

        # validation: loss + mean anyAD(>=4.0) AUROC over the two sites
        model.eval()
        with torch.no_grad():
            ol, rg = model(ecg[va], cxr[va], ehr[va])
            val_loss = float(coral_loss(ol, cum[va]).item()
                             + lambda_reg * masked_mse(rg, zdiam[va]).item())
        probs = cum_probs(ol.cpu().numpy())          # (Bv, 2, 3)
        cum_va = cum[va].cpu().numpy()
        aurocs = []
        for s in range(2):
            m = binary_metrics(cum_va[:, s, 0], probs[:, s, 0])
            if not np.isnan(m["auroc"]):
                aurocs.append(m["auroc"])
        score = float(np.mean(aurocs)) if aurocs else -1e9
        history.append((epoch + 1, train_loss, val_loss, score if score > -1e8 else float("nan")))
        if score > best_score:
            best_score, no_imp = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.splits import load_fold_assignments, load_fold_id_map
    from multimodal_aorta.data.targets import diam_to_grade, diam_to_cumulative
    from multimodal_aorta.data.ehr import load_ehr_features, FEATURE_COLS
    from multimodal_aorta.models.ecg_encoder import PCLREmbeddingEncoder
    from multimodal_aorta.models.cxr_encoder import CXREmbeddingEncoder
    from multimodal_aorta.models.fusion_v3 import V3FusionModel
    from multimodal_aorta.training.ordinal import (
        cum_probs, binary_metrics, reg_metrics, ordinal_qwk, grade_from_cum)
    from multimodal_aorta.training.oof_io import build_records, save_oof, write_manifest

    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(root, "outputs", "deep_fusion"); os.makedirs(out_dir, exist_ok=True)
    pc = os.path.join(root, "pretrained_checkpoints")
    fold_path = os.path.join(pc, "fold_assignments.csv")

    cohort = pd.read_csv(os.path.join(pc, "cohort_triple.csv"))
    pclr = PCLREmbeddingEncoder.load_embeddings(os.path.join(pc, "pclr_embeddings.pt"))
    raddino = CXREmbeddingEncoder.load_embeddings(os.path.join(pc, "raddino_embeddings.pt"))
    ehr = load_ehr_features(os.path.join(pc, "ehr_features.csv"))

    df, ECG, CXR, EHR = assemble(cohort, pclr, raddino, ehr, FEATURE_COLS)
    log.info("Assembled: ECG%s CXR%s EHR%s (n=%d)", ECG.shape, CXR.shape, EHR.shape, len(df))
    idx_of = {int(s): i for i, s in enumerate(df["subject_id"])}
    N = len(df)

    # Targets (cm). cum (N,2,3); diam (N,2); grade (N,2)
    diam = np.stack([df["target_root"].to_numpy(float), df["target_asc"].to_numpy(float)], axis=1)
    cum = np.stack([[diam_to_cumulative(diam[i, s]) for s in range(2)] for i in range(N)]).astype(np.float32)
    grade = np.stack([[diam_to_grade(diam[i, s]) for s in range(2)] for i in range(N)])

    folds = load_fold_assignments(fold_path, n_splits=5)

    # OOF prediction buffers
    oof_probs = np.full((N, 2, 3), np.nan, np.float32)
    oof_diam = np.full((N, 2), np.nan, np.float32)

    rng = np.random.default_rng(SEED)
    all_history = []   # (fold, epoch, train_loss, val_loss, val_auroc) for curves
    for fold, (train_ids, test_ids) in enumerate(folds):
        tr_all = np.array([idx_of[s] for s in train_ids if s in idx_of])
        te = np.array([idx_of[s] for s in test_ids if s in idx_of])
        # inner stratified-ish val (15%) for early stopping
        anyad = df["anyAD"].to_numpy()[tr_all]
        val_n = max(int(0.15 * len(tr_all)), 20)
        pos = tr_all[anyad == 1]; neg = tr_all[anyad == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_vp = max(int(val_n * len(pos) / len(tr_all)), 2)
        va = np.concatenate([pos[:n_vp], neg[:val_n - n_vp]])
        tr = np.setdiff1d(tr_all, va)

        # EHR scaling on inner-train; diam z-score on inner-train
        scaler = fit_ehr_scaler(EHR[tr])
        EHRs = apply_ehr_scaler(EHR, scaler)
        dmean = np.nanmean(diam[tr], axis=0); dstd = np.nanstd(diam[tr], axis=0) + 1e-6
        zdiam = (diam - dmean) / dstd

        tens = (torch.tensor(ECG, device=device), torch.tensor(CXR, device=device),
                torch.tensor(EHRs, device=device), torch.tensor(cum, device=device),
                torch.tensor(zdiam, dtype=torch.float32, device=device))
        trI = torch.tensor(tr, device=device); vaI = torch.tensor(va, device=device)

        model = V3FusionModel(ehr_dim=EHR.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        model, hist = train_fold(model, opt, trI, vaI, tens, device)
        for (ep, tl, vl, va_auc) in hist:
            all_history.append((fold + 1, ep, tl, vl, va_auc))

        teI = torch.tensor(te, dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            ol, rg = model(tens[0][teI], tens[1][teI], tens[2][teI])
        oof_probs[te] = cum_probs(ol.cpu().numpy())
        oof_diam[te] = (rg.cpu().numpy() * dstd + dmean)
        log.info("fold %d done (train %d / val %d / test %d)", fold + 1, len(tr), len(va), len(te))

    # ---- OOF metrics per site ----
    results = {"n_patients": int(N), "sites": {}}
    for s, site in enumerate(("root", "asc")):
        ge40 = binary_metrics(cum[:, s, 0], oof_probs[:, s, 0])
        ge45 = binary_metrics(cum[:, s, 1], oof_probs[:, s, 1])
        qwk = ordinal_qwk(grade[:, s].astype(float), grade_from_cum(oof_probs[:, s, :]).astype(float))
        rm = reg_metrics(diam[:, s], oof_diam[:, s])
        results["sites"][site] = {"anyAD_ge4.0": ge40, "moderate_ge4.5": ge45,
                                  "ordinal_qwk": qwk, "diam_regression": rm}
        log.info("=== SITE %s ===", site.upper())
        log.info("  anyAD>=4.0 AUROC=%.3f AUPRC=%.3f (pos=%d) | mod>=4.5 AUROC=%.3f (pos=%d)",
                 ge40["auroc"], ge40["auprc"], ge40["pos"], ge45["auroc"], ge45["pos"])
        log.info("  ordinal QWK=%.3f | diam MAE=%.3f R2=%.3f", qwk, rm["mae"], rm["r2"])

    # ---- save OOF predictions (for bootstrap-CI comparison) ----
    oof_store = {}
    for s, site in enumerate(("root", "asc")):
        oof_store[f"{site}_ge40_y"] = cum[:, s, 0]; oof_store[f"{site}_ge40_p"] = oof_probs[:, s, 0]
        oof_store[f"{site}_ge45_y"] = cum[:, s, 1]; oof_store[f"{site}_ge45_p"] = oof_probs[:, s, 1]
        oof_store[f"{site}_diam_y"] = diam[:, s];   oof_store[f"{site}_diam_p"] = oof_diam[:, s]
    np.savez(os.path.join(out_dir, "oof.npz"), **oof_store)

    # --- standardized per-patient OOF (M0) for cross-model paired comparison ---
    subj = [int(s) for s in df["subject_id"].tolist()]
    fold_map = load_fold_id_map(fold_path)
    fold_ids = [fold_map.get(s, -1) for s in subj]
    has_ehr = [int(s in ehr) for s in subj]
    records = []
    for site in ("root", "asc"):
        for ep in ("ge40", "ge45"):
            records += build_records(
                subject_ids=subj, fold_ids=fold_ids, model_name="deep_fusion",
                modality_set="ecg+cxr+ehr", site=site, endpoint=ep, target_type="binary",
                y_true=oof_store[f"{site}_{ep}_y"], pred_prob=oof_store[f"{site}_{ep}_p"],
                has_ehr=has_ehr)
        records += build_records(
            subject_ids=subj, fold_ids=fold_ids, model_name="deep_fusion",
            modality_set="ecg+cxr+ehr", site=site, endpoint="diam", target_type="regression",
            y_true=oof_store[f"{site}_diam_y"], pred_value=oof_store[f"{site}_diam_p"],
            has_ehr=has_ehr)
    save_oof(records, os.path.join(out_dir, "oof_predictions.csv"))
    write_manifest(
        os.path.join(out_dir, "run_manifest.json"), model_name="deep_fusion",
        seed=SEED, n_patients=int(N),
        cohort_csv=os.path.join(pc, "cohort_triple.csv"), fold_csv=fold_path)

    # ---- training/validation curves ----
    hist_df = pd.DataFrame(all_history, columns=["fold", "epoch", "train_loss", "val_loss", "val_auroc"])
    hist_df.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    _plot_curves(hist_df, os.path.join(out_dir, "training_curves.png"))

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s (+ oof.npz, training_curves.png)", os.path.join(out_dir, "results.json"))


def _plot_curves(hist_df, path):
    """Per-fold train/val loss + val AUROC curves (early-stop point marked)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folds = sorted(hist_df["fold"].unique())
    fig, axes = plt.subplots(1, len(folds), figsize=(4 * len(folds), 3.4), squeeze=False)
    for ax, fk in zip(axes[0], folds):
        h = hist_df[hist_df.fold == fk]
        ax.plot(h.epoch, h.train_loss, label="train loss", color="C0")
        ax.plot(h.epoch, h.val_loss, label="val loss", color="C1")
        ax.set_title(f"fold {fk}"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax2 = ax.twinx()
        ax2.plot(h.epoch, h.val_auroc, label="val AUROC", color="C2", ls="--")
        ax2.set_ylabel("val AUROC"); ax2.set_ylim(0.3, 1.0)
        best = h.loc[h.val_auroc.idxmax(), "epoch"] if h.val_auroc.notna().any() else None
        if best is not None:
            ax.axvline(best, color="grey", ls=":", lw=1)
    axes[0][0].legend(loc="upper right", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
