"""
Combine the two CXR wins: extract anatomy-ROI RAD-DINO embeddings for EVERY
in-window CXR instance (not just the single best).

For each of the ~2,792 instances in cxr_instances.csv:
  1. Segment with ChestX-Det PSPNet (torchxrayvision) -> per-image box =
     (Aorta u Mediastinum u Heart) + 6% pad  (fallback: whole image).
  2. Crop to that box, run frozen RAD-DINO -> 768-dim CLS embedding.

Output: pretrained_checkpoints/raddino_multi_anatomy_embeddings.pt  {dicom_id: (768,)}
        pretrained_checkpoints/cxr_instance_anatomy_rois.csv         (per-instance box + QC)

Run on GPU via roi_venv (has torchxrayvision + transformers + GPU torch).
"""
import os, sys, logging
import numpy as np, pandas as pd, torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(PROJ, "pretrained_checkpoints")
OUT_EMB = os.path.join(PC, "raddino_multi_anatomy_embeddings.pt")
OUT_CSV = os.path.join(PC, "cxr_instance_anatomy_rois.csv")
STRUCTS = ["Aorta", "Mediastinum", "Heart"]
PAD = 0.06
THRESH = 0.5
BATCH = 16


def union_box(pred, idx):
    union = np.zeros((512, 512), dtype=bool)
    for s, j in idx.items():
        union |= pred[j] > THRESH
    ys, xs = np.where(union)
    if len(xs) == 0:
        return (0.0, 0.0, 1.0, 1.0), False
    x0 = max(0.0, xs.min() / 512 - PAD); y0 = max(0.0, ys.min() / 512 - PAD)
    x1 = min(1.0, xs.max() / 512 + PAD); y1 = min(1.0, ys.max() / 512 + PAD)
    return (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)), True


def main():
    sys.path.insert(0, os.path.join(PROJ, "src"))
    import torchxrayvision as xrv
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.preprocessing import load_cxr
    from multimodal_aorta.models.cxr_encoder import CXREncoder
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    seg = xrv.baseline_models.chestx_det.PSPNet().eval().to(device)
    idx = {s: list(seg.targets).index(s) for s in STRUCTS}
    enc = CXREncoder(model_name=cfg.model.cxr_model_name, freeze_blocks=12).eval().to(device)

    _INST = os.environ.get("CXR_INSTANCES", "cxr_instances.csv")  # episode rebuild: cxr_instances_episode.csv
    inst = pd.read_csv(os.path.join(PC, _INST))
    log.info("Instances: %d across %d patients", len(inst), inst.subject_id.nunique())

    emb, roi_rows, nfail, nfallback = {}, [], 0, 0
    with torch.no_grad():
        for start in range(0, len(inst), BATCH):
            b = inst.iloc[start:start + BATCH]
            # --- segment batch ---
            grays, valid = [], []
            for _, row in b.iterrows():
                try:
                    g = np.asarray(Image.open(row.cxr_path).convert("L").resize((512, 512), Image.BILINEAR), np.float32)
                    grays.append(xrv.datasets.normalize(g, 255)); valid.append(row)
                except Exception as e:
                    log.warning("seg-load fail %s: %s", row.dicom_id, e); nfail += 1
            if not grays:
                continue
            seg_in = torch.from_numpy(np.stack(grays)[:, None]).float().to(device)
            preds = torch.sigmoid(seg(seg_in)).cpu().numpy()          # (B,14,512,512)
            boxes = []
            for k, row in enumerate(valid):
                box, ok = union_box(preds[k], idx)
                if not ok:
                    nfallback += 1
                boxes.append(box)
                roi_rows.append({"subject_id": int(row.subject_id), "dicom_id": row.dicom_id,
                                 "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "seg_ok": ok})
            # --- crop + RAD-DINO batch ---
            imgs, dids = [], []
            for row, box in zip(valid, boxes):
                try:
                    imgs.append(load_cxr(row.cxr_path, cfg.data, is_train=False, roi=box)); dids.append(row.dicom_id)
                except Exception as e:
                    log.warning("cxr-load fail %s: %s", row.dicom_id, e); nfail += 1
            if imgs:
                e = enc(torch.stack(imgs).to(device)).cpu()
                for did, v in zip(dids, e):
                    emb[did] = v.float()
            if start % (BATCH * 10) == 0:
                log.info("  %d/%d (emb=%d fallback=%d fail=%d)", start + len(b), len(inst), len(emb), nfallback, nfail)

    torch.save(emb, OUT_EMB)
    pd.DataFrame(roi_rows).to_csv(OUT_CSV, index=False)
    log.info("Saved %d embeddings -> %s (fallback=%d fail=%d)", len(emb), OUT_EMB, nfallback, nfail)


if __name__ == "__main__":
    main()
