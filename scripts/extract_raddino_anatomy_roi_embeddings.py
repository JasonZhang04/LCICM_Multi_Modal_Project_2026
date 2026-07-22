"""
Extract RAD-DINO embeddings on the per-patient ANATOMY ROI (from anatomy_rois.csv).
Reads cohort_triple.csv for cxr_path (no cohort rebuild needed) and the per-subject
fractional box from anatomy_rois.csv, crops, runs frozen RAD-DINO, saves
pretrained_checkpoints/raddino_anatomy_roi_embeddings.pt.

ROI_MODE selects which structures form the box (columns already in anatomy_rois.csv):
  "union"          -> the saved union box (Aorta u Mediastinum u Heart, padded)
  "great_vessels"  -> Aorta u Mediastinum (padded), excludes lower heart
"""
import os, sys, logging
import numpy as np, pandas as pd, torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
PROJ = "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
PC = os.path.join(PROJ, "pretrained_checkpoints")
ROI_MODE = os.environ.get("ROI_MODE", "union")
OUT = os.path.join(PC, f"raddino_anatomy_roi_{ROI_MODE}_embeddings.pt")
BATCH = 16
PAD = 0.06


def box_for(row):
    """row is a dict (from DataFrame.to_dict('records'))."""
    if ROI_MODE == "union":
        return row["x0"], row["y0"], row["x1"], row["y1"]
    # great_vessels: union of Aorta + Mediastinum boxes, padded, fallback to union
    xs0, ys0, xs1, ys1 = [], [], [], []
    for s in ("Aorta", "Mediastinum"):
        b = (row[f"{s}_bx0"], row[f"{s}_by0"], row[f"{s}_bx1"], row[f"{s}_by1"])
        if not any(pd.isna(v) for v in b):
            xs0.append(b[0]); ys0.append(b[1]); xs1.append(b[2]); ys1.append(b[3])
    if not xs0:
        return row["x0"], row["y0"], row["x1"], row["y1"]
    return (max(0., min(xs0) - PAD), max(0., min(ys0) - PAD),
            min(1., max(xs1) + PAD), min(1., max(ys1) + PAD))


def main():
    sys.path.insert(0, os.path.join(PROJ, "src"))
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.preprocessing import load_cxr
    from multimodal_aorta.models.cxr_encoder import CXREncoder
    cfg = Config()

    rois = pd.read_csv(os.path.join(PC, "anatomy_rois.csv"))
    roi_of = {int(r["subject_id"]): box_for(r) for r in rois.to_dict("records")}
    cohort = pd.read_csv(os.path.join(PC, "cohort_triple.csv"))
    cohort = cohort[cohort.has_cxr].reset_index(drop=True)
    log.info("ROI_MODE=%s | %d CXR patients | %d ROIs", ROI_MODE, len(cohort), len(roi_of))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = CXREncoder(model_name=cfg.model.cxr_model_name, freeze_blocks=12).eval().to(device)

    emb = {}; nfail = 0
    with torch.no_grad():
        for start in range(0, len(cohort), BATCH):
            b = cohort.iloc[start:start + BATCH]
            imgs, sids = [], []
            for _, row in b.iterrows():
                sid = int(row.subject_id)
                roi = roi_of.get(sid, (0., 0., 1., 1.))
                try:
                    imgs.append(load_cxr(row.cxr_path, cfg.data, is_train=False, roi=roi))
                    sids.append(sid)
                except Exception as e:
                    log.warning("load fail %s: %s", sid, e); nfail += 1
            if not imgs:
                continue
            e = enc(torch.stack(imgs).to(device)).cpu()
            for sid, v in zip(sids, e):
                # .clone() is REQUIRED: the encoder returns a view into the full
                # (B, 197, 768) hidden state; without cloning, torch.save keeps that
                # whole storage per vector (250x file bloat on CPU where .cpu() no-ops).
                emb[sid] = v.detach().clone().float()
            if start % (BATCH * 5) == 0:
                log.info("  %d/%d (fail=%d)", start + len(b), len(cohort), nfail)
    torch.save(emb, OUT)
    log.info("Saved %d -> %s (fail=%d)", len(emb), OUT, nfail)


if __name__ == "__main__":
    main()
