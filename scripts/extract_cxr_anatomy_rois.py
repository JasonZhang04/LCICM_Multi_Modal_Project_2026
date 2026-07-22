"""
Anatomy-driven ROI: segment the aorta + mediastinum on each cohort CXR with a
pretrained ChestX-Det PSPNet (torchxrayvision), and derive a per-patient
fractional bounding box. Replaces the hard-coded (0.25,0.08,0.75,0.60) crop.

Output: anatomy_rois.csv  (subject_id, cxr_path, x0,y0,x1,y1 fractional box,
                           plus per-structure area fractions for QC)

Design choices:
 - Segment on a plain resize-to-512 (NO center crop) so the mask's fractional
   coordinates map directly onto the original image (which load_cxr crops with
   fractional coords).
 - ROI = union of {Aorta, Mediastinum, Heart} masks (the great-vessel silhouette),
   padded by `PAD` on each side, clamped to [0,1]. Falls back to the whole image
   if segmentation is empty/failed.
"""
import os, sys, logging
import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJ = "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
PC = os.path.join(PROJ, "pretrained_checkpoints")
OUT_CSV = os.path.join(PC, "anatomy_rois.csv")
PAD = 0.06                      # fractional padding around the union bbox
STRUCTS = ["Aorta", "Mediastinum", "Heart"]
THRESH = 0.5


def load_gray_512(path):
    img = Image.open(path).convert("L").resize((512, 512), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)   # (512,512) in [0,255]


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return xs.min() / mask.shape[1], ys.min() / mask.shape[0], \
           xs.max() / mask.shape[1], ys.max() / mask.shape[0]


def main():
    import torch
    import torchxrayvision as xrv

    cohort = pd.read_csv(os.path.join(PC, "cohort_triple.csv"))
    cohort = cohort[cohort["has_cxr"]].reset_index(drop=True)
    log.info("CXR patients: %d", len(cohort))

    model = xrv.baseline_models.chestx_det.PSPNet()
    model.eval()
    tgt = list(model.targets)
    log.info("Seg targets: %s", tgt)
    idx = {s: tgt.index(s) for s in STRUCTS if s in tgt}
    log.info("Using structures/indices: %s", idx)

    rows = []
    n_fallback = 0
    with torch.no_grad():
        for i, r in enumerate(cohort.itertuples(index=False), 1):
            sid = int(r.subject_id)
            try:
                g = load_gray_512(r.cxr_path)                    # (512,512) [0,255]
                x = xrv.datasets.normalize(g, 255)               # -> [-1024,1024]
                t = torch.from_numpy(x[None, None]).float()      # (1,1,512,512)
                pred = torch.sigmoid(model(t))[0].cpu().numpy()  # (14,512,512)
                union = np.zeros((512, 512), dtype=bool)
                areas, per_bbox = {}, {}
                for s, j in idx.items():
                    m = pred[j] > THRESH
                    areas[s] = float(m.mean())
                    per_bbox[s] = bbox_from_mask(m)
                    union |= m
                bb = bbox_from_mask(union)
                if bb is None:
                    n_fallback += 1
                    x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
                else:
                    x0, y0, x1, y1 = bb
                    x0 = max(0.0, x0 - PAD); y0 = max(0.0, y0 - PAD)
                    x1 = min(1.0, x1 + PAD); y1 = min(1.0, y1 + PAD)
            except Exception as e:
                log.warning("seg failed sid=%s: %s", sid, e)
                n_fallback += 1
                x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
                areas = {s: float("nan") for s in idx}
                per_bbox = {s: None for s in idx}
            rec = {"subject_id": sid, "cxr_path": r.cxr_path,
                   "x0": round(x0, 4), "y0": round(y0, 4),
                   "x1": round(x1, 4), "y1": round(y1, 4),
                   **{f"area_{s}": round(areas.get(s, float('nan')), 4) for s in STRUCTS}}
            for s in STRUCTS:
                b = per_bbox.get(s)
                for k, name in zip(range(4), ("bx0", "by0", "bx1", "by1")):
                    rec[f"{s}_{name}"] = round(b[k], 4) if b else float("nan")
            rows.append(rec)
            if i % 50 == 0:
                log.info("  %d/%d (fallbacks=%d)", i, len(cohort), n_fallback)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    # QC summary of box geometry
    log.info("Saved %s (n=%d, fallbacks=%d)", OUT_CSV, len(df), n_fallback)
    log.info("box width  frac: median=%.3f  IQR=[%.3f,%.3f]",
             (df.x1 - df.x0).median(), (df.x1 - df.x0).quantile(.25), (df.x1 - df.x0).quantile(.75))
    log.info("box height frac: median=%.3f  IQR=[%.3f,%.3f]",
             (df.y1 - df.y0).median(), (df.y1 - df.y0).quantile(.25), (df.y1 - df.y0).quantile(.75))
    log.info("box x-center median=%.3f  y-center median=%.3f",
             ((df.x0 + df.x1) / 2).median(), ((df.y0 + df.y1) / 2).median())


if __name__ == "__main__":
    main()
