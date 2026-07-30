"""
REC #1: Engineered GEOMETRIC aortic features from CXR segmentation masks.

Rationale: a widened mediastinal / aortic silhouette IS how aortic dilation presents
on a frontal chest film. Instead of only a generic 768-dim RAD-DINO embedding, compute
the actual measurements radiologists use. These are low-dimensional (robust at n=522),
interpretable, and — crucially — ORTHOGONAL to both the ViT embedding and to EHR, which
is what late fusion needs in order to exceed the best single modality.

Segments each FRONTAL (PA/AP) in-window CXR with ChestX-Det PSPNet (torchxrayvision)
and derives, per image:

  thoracic_width           lung-to-lung outer extent (the normalizer)
  cardiothoracic_ratio     heart width / thoracic width          (classic CTR)
  mediastinal_ratio        max mediastinum width / thoracic width (MTR)
  med_upper_ratio          UPPER-third mediastinum width / thoracic width
                           <- the arch / ascending-aorta region: most relevant to AD
  med_upper_over_lower     upper vs lower mediastinal width (aortic vs cardiac share)
  aorta_width/height/area  aorta mask geometry (fraction of image / of thorax)
  aorta_knob_lateral       leftmost aorta extent relative to spine midline
                           (aortic knob prominence)
  aorta_centroid_offset    aorta centroid x relative to midline
  heart_area_ratio         heart area / thoracic area

Width/width ratios are invariant to the 512x512 resize aspect distortion, so they are the
primary features; raw fractional sizes are kept too (they carry absolute-size signal).

Output: pretrained_checkpoints/cxr_geometry_features.csv
        (subject_id, dicom_id, <features...>, seg_ok)
"""
import os, sys, logging
import numpy as np, pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(PROJ, "pretrained_checkpoints")
OUT = os.path.join(PC, os.environ.get("GEOMETRY_OUT", "cxr_geometry_features.csv"))
S = 512
THRESH = 0.5
BATCH = 16

FEATURES = [
    "thoracic_width", "cardiothoracic_ratio", "mediastinal_ratio", "med_upper_ratio",
    "med_mid_ratio", "med_lower_ratio", "med_upper_over_lower",
    "aorta_w_frac", "aorta_h_frac", "aorta_area_frac", "aorta_area_over_thorax",
    "aorta_knob_lateral", "aorta_centroid_offset", "aorta_top_y",
    "heart_w_frac", "heart_area_ratio", "med_area_ratio",
]


def _extent(mask):
    """(x0,x1,y0,y1,width,height,area_frac) in fractional units; None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (xs.min() / S, xs.max() / S, ys.min() / S, ys.max() / S,
            (xs.max() - xs.min()) / S, (ys.max() - ys.min()) / S, mask.mean())


def _row_width(mask, y0f, y1f):
    """Max horizontal width (fraction) of mask within a vertical band [y0f,y1f)."""
    y0, y1 = int(y0f * S), int(y1f * S)
    band = mask[y0:y1]
    if band.sum() == 0:
        return np.nan
    widths = []
    for r in band:
        xs = np.where(r)[0]
        if len(xs):
            widths.append((xs.max() - xs.min()) / S)
    return float(np.max(widths)) if widths else np.nan


def geometry(pred, idx):
    """Compute the geometric feature dict from sigmoid mask stack."""
    m = {k: pred[i] > THRESH for k, i in idx.items()}
    f = {k: np.nan for k in FEATURES}

    lungs = m.get("Left Lung", np.zeros((S, S), bool)) | m.get("Right Lung", np.zeros((S, S), bool))
    el = _extent(lungs)
    if el is None:
        return f, False
    thor_w = el[4]
    thor_area = lungs.mean()
    f["thoracic_width"] = thor_w
    if thor_w <= 0:
        return f, False

    eh = _extent(m.get("Heart", np.zeros((S, S), bool)))
    if eh:
        f["heart_w_frac"] = eh[4]
        f["cardiothoracic_ratio"] = eh[4] / thor_w
        f["heart_area_ratio"] = eh[6] / max(thor_area, 1e-6)

    med = m.get("Mediastinum", np.zeros((S, S), bool))
    em = _extent(med)
    if em:
        f["mediastinal_ratio"] = em[4] / thor_w
        f["med_area_ratio"] = em[6] / max(thor_area, 1e-6)
        # vertical bands over the mediastinum's own extent (upper = arch/ascending)
        y0, y1 = em[2], em[3]
        h = max(y1 - y0, 1e-6)
        up = _row_width(med, y0, y0 + h / 3)
        mid = _row_width(med, y0 + h / 3, y0 + 2 * h / 3)
        lo = _row_width(med, y0 + 2 * h / 3, y1)
        f["med_upper_ratio"] = up / thor_w if np.isfinite(up) else np.nan
        f["med_mid_ratio"] = mid / thor_w if np.isfinite(mid) else np.nan
        f["med_lower_ratio"] = lo / thor_w if np.isfinite(lo) else np.nan
        if np.isfinite(up) and np.isfinite(lo) and lo > 0:
            f["med_upper_over_lower"] = up / lo

    # midline from spine (fallback: thorax centre)
    esp = _extent(m.get("Spine", np.zeros((S, S), bool)))
    midline = ((esp[0] + esp[1]) / 2) if esp else ((el[0] + el[1]) / 2)

    ea = _extent(m.get("Aorta", np.zeros((S, S), bool)))
    if ea:
        f["aorta_w_frac"] = ea[4]
        f["aorta_h_frac"] = ea[5]
        f["aorta_area_frac"] = ea[6]
        f["aorta_area_over_thorax"] = ea[6] / max(thor_area, 1e-6)
        f["aorta_top_y"] = ea[2]
        # knob prominence: how far the aorta extends to one side of the spine midline,
        # normalised by thoracic width (size-invariant)
        f["aorta_knob_lateral"] = (midline - ea[0]) / thor_w
        aor = m["Aorta"]
        ys, xs = np.where(aor)
        f["aorta_centroid_offset"] = (xs.mean() / S - midline) / thor_w
    return f, True


def main():
    import torch, torchxrayvision as xrv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = xrv.baseline_models.chestx_det.PSPNet().eval().to(device)
    targets = list(seg.targets)
    want = ["Left Lung", "Right Lung", "Heart", "Aorta", "Mediastinum", "Spine"]
    idx = {s: targets.index(s) for s in want if s in targets}
    log.info("Device=%s | structures=%s", device, idx)

    _INST = os.environ.get("CXR_INSTANCES", "cxr_instances.csv")  # episode rebuild: cxr_instances_episode.csv
    inst = pd.read_csv(os.path.join(PC, _INST))
    inst = inst[inst.view_position.isin(["PA", "AP"])].reset_index(drop=True)

    # Resume: load rows already computed and skip those dicoms (multi-hour job at 60k).
    rows = []
    if os.path.exists(OUT):
        try:
            prev = pd.read_csv(OUT)
            rows = prev.to_dict("records")
            done = set(prev.dicom_id.astype(str))
            inst = inst[~inst.dicom_id.astype(str).isin(done)].reset_index(drop=True)
            log.info("resume: loaded %d existing rows; %d instances left", len(rows), len(inst))
        except Exception as e:  # noqa: BLE001
            log.warning("could not load %s (%s) — starting fresh", OUT, e)
    log.info("Frontal instances: %d to do across %d patients", len(inst), inst.subject_id.nunique())

    SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "4000"))

    def _checkpoint():
        tmp = OUT + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, OUT)

    nfail, nseg_bad, last_saved = 0, 0, len(rows)
    with torch.no_grad():
        for start in range(0, len(inst), BATCH):
            b = inst.iloc[start:start + BATCH]
            grays, keep = [], []
            for _, r in b.iterrows():
                try:
                    g = np.asarray(Image.open(r.cxr_path).convert("L").resize((S, S), Image.BILINEAR), np.float32)
                    grays.append(xrv.datasets.normalize(g, 255)); keep.append(r)
                except Exception as e:
                    log.warning("load fail %s: %s", r.dicom_id, e); nfail += 1
            if not grays:
                continue
            x = torch.from_numpy(np.stack(grays)[:, None]).float().to(device)
            preds = torch.sigmoid(seg(x)).cpu().numpy()
            for k, r in enumerate(keep):
                f, ok = geometry(preds[k], idx)
                if not ok:
                    nseg_bad += 1
                rows.append({"subject_id": int(r.subject_id), "dicom_id": r.dicom_id,
                             "seg_ok": int(ok), **f})
            if start % (BATCH * 20) == 0:
                log.info("  %d/%d (bad_seg=%d fail=%d)", start + len(b), len(inst), nseg_bad, nfail)
            if len(rows) - last_saved >= SAVE_EVERY:
                _checkpoint(); last_saved = len(rows)
                log.info("  checkpoint: %d rows saved", len(rows))

    _checkpoint()
    df = pd.DataFrame(rows)
    log.info("Saved %d rows -> %s (bad_seg=%d fail=%d)", len(df), OUT, nseg_bad, nfail)
    log.info("Feature coverage (non-null %%):\n%s",
             (df[FEATURES].notna().mean() * 100).round(1).to_string())
    log.info("Feature medians:\n%s", df[FEATURES].median().round(3).to_string())


if __name__ == "__main__":
    main()
