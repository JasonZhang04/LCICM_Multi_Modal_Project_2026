"""
REC #2: RAD-DINO PATCH-TOKEN pooling (spatially-aware CXR embedding).

Problem: the 768-dim CLS token is a single whole-image summary, which averages away
the *localized* aortic-knob / mediastinal-border signal. Anatomy-ROI cropping fixes
locality but throws away surrounding context (and over-cropping hurt in M4c).

This keeps BOTH: run RAD-DINO on the WHOLE image, take the 196 patch tokens (14x14
grid for ViT-B/16 @224), and pool them with the segmentation mask as weights:

    cls        (768)  global context               [current representation]
    aortapool  (768)  mask-weighted mean over the aorta u mediastinum patches
    heartpool  (768)  mask-weighted mean over the heart patches

The mask-weighted pool is a *soft* ROI: spatially selective without cropping.

Output: pretrained_checkpoints/raddino_patchpool_embeddings.pt
        {dicom_id: {"cls": (768,), "aortapool": (768,), "heartpool": (768,)}}
"""
import os, sys, logging
import numpy as np, pandas as pd, torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(PROJ, "pretrained_checkpoints")
OUT = os.path.join(PC, "raddino_patchpool_embeddings.pt")
SEG_S = 512
THRESH = 0.5
BATCH = 16


def mask_to_patch_weights(mask, g):
    """Downsample a (512,512) bool mask to a (g,g) weight grid summing to 1."""
    m = mask.astype(np.float32)
    # average-pool 512 -> g
    fac = m.shape[0] // g
    m = m[:fac * g, :fac * g].reshape(g, fac, g, fac).mean(axis=(1, 3))
    s = m.sum()
    if s <= 1e-6:
        return None
    return (m / s).reshape(-1)                      # (g*g,)


def main():
    sys.path.insert(0, os.path.join(PROJ, "src"))
    import torchxrayvision as xrv
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.preprocessing import load_cxr
    from multimodal_aorta.models.cxr_encoder import CXREncoder
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg = xrv.baseline_models.chestx_det.PSPNet().eval().to(device)
    tg = list(seg.targets)
    I_AO, I_MED, I_HRT = tg.index("Aorta"), tg.index("Mediastinum"), tg.index("Heart")

    enc = CXREncoder(model_name=cfg.model.cxr_model_name, freeze_blocks=12).eval().to(device)

    _INST = os.environ.get("CXR_INSTANCES", "cxr_instances.csv")  # episode rebuild: cxr_instances_episode.csv
    inst = pd.read_csv(os.path.join(PC, _INST))
    inst = inst[inst.view_position.isin(["PA", "AP"])].reset_index(drop=True)
    log.info("Frontal instances: %d | device=%s", len(inst), device)

    emb = {}; nfail = 0; nfallback = 0
    with torch.no_grad():
        for start in range(0, len(inst), BATCH):
            b = inst.iloc[start:start + BATCH]
            grays, imgs, dids = [], [], []
            for _, r in b.iterrows():
                try:
                    g = np.asarray(Image.open(r.cxr_path).convert("L").resize((SEG_S, SEG_S), Image.BILINEAR), np.float32)
                    im = load_cxr(r.cxr_path, cfg.data, is_train=False)       # whole image, 224
                except Exception as e:
                    log.warning("load fail %s: %s", r.dicom_id, e); nfail += 1; continue
                grays.append(xrv.datasets.normalize(g, 255)); imgs.append(im); dids.append(r.dicom_id)
            if not imgs:
                continue
            # --- segmentation masks ---
            sx = torch.from_numpy(np.stack(grays)[:, None]).float().to(device)
            sp = torch.sigmoid(seg(sx)).cpu().numpy()
            # --- RAD-DINO patch tokens ---
            px = torch.stack(imgs).to(device)
            out = enc.model(pixel_values=px).last_hidden_state          # (B, 1+N, 768)
            cls_tok = out[:, 0, :].cpu().numpy()
            patches = out[:, 1:, :].cpu().numpy()                       # (B, N, 768)
            N = patches.shape[1]; g = int(round(np.sqrt(N)))
            if g * g != N:
                log.error("patch grid not square: N=%d", N); return
            for k, did in enumerate(dids):
                rec = {"cls": torch.tensor(cls_tok[k])}
                ao = (sp[k, I_AO] > THRESH) | (sp[k, I_MED] > THRESH)
                hr = sp[k, I_HRT] > THRESH
                for name, mask in (("aortapool", ao), ("heartpool", hr)):
                    w = mask_to_patch_weights(mask, g)
                    if w is None:
                        nfallback += 1
                        rec[name] = torch.tensor(patches[k].mean(0))     # fallback: mean-pool
                    else:
                        rec[name] = torch.tensor((patches[k] * w[:, None]).sum(0))
                emb[did] = rec
            if start % (BATCH * 20) == 0:
                log.info("  %d/%d (emb=%d fallback=%d fail=%d)", start + len(b), len(inst),
                         len(emb), nfallback, nfail)
    torch.save(emb, OUT)
    log.info("Saved %d -> %s (mask-fallbacks=%d fail=%d)", len(emb), OUT, nfallback, nfail)


if __name__ == "__main__":
    main()
