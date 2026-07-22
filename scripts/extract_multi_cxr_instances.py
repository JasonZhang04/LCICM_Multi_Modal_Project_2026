"""
Multi-instance CXR extraction: enumerate ALL in-window (+/-180d) CXRs for each
triple-cohort patient and extract a frozen RAD-DINO embedding per image.

Rationale: the current pipeline keeps only the single best CXR/patient (522
images). Every in-window CXR is a valid training instance for the same patient-
level aortic label, so this yields ~2800 instances (~5x). Patient-level CV keeps
all of a patient's instances in one fold (no leakage); predictions are averaged
per patient at test time.

Outputs (pretrained_checkpoints/):
  cxr_instances.csv                 subject_id, dicom_id, view, days_off, cxr_path
  raddino_multi_embeddings.pt       {dicom_id: tensor(768,)}
"""
import os, sys, logging
import numpy as np, pandas as pd, torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(PROJ, "pretrained_checkpoints")
WINDOW = 180
BATCH = 64


def main():
    sys.path.insert(0, os.path.join(PROJ, "src"))
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.preprocessing import load_cxr
    from multimodal_aorta.models.cxr_encoder import CXREncoder
    cfg = Config()

    cohort = pd.read_csv(os.path.join(PC, "cohort_triple.csv"), parse_dates=["echo_date"])
    echo = dict(zip(cohort.subject_id.astype(int), cohort.echo_date))
    sids = set(echo)

    cache = pd.read_csv(os.path.join(PROJ, "data", "cxr_metadata_cache.csv"), parse_dates=["cxr_date"])
    cache = cache[cache.subject_id.isin(sids)].copy()
    cache["days_off"] = cache.apply(
        lambda r: abs((r.cxr_date - echo[int(r.subject_id)]).days)
        if pd.notna(r.cxr_date) and pd.notna(echo[int(r.subject_id)]) else np.nan, axis=1)
    inst = cache[cache.days_off <= WINDOW].dropna(subset=["cxr_path"]).reset_index(drop=True)
    inst = inst[["subject_id", "dicom_id", "view_position", "days_off", "cxr_path"]]
    inst.to_csv(os.path.join(PC, "cxr_instances.csv"), index=False)
    log.info("Enumerated %d in-window CXR instances across %d patients",
             len(inst), inst.subject_id.nunique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("RAD-DINO on %s", device)
    enc = CXREncoder(model_name=cfg.model.cxr_model_name, freeze_blocks=12).eval().to(device)

    emb = {}; nfail = 0
    with torch.no_grad():
        for start in range(0, len(inst), BATCH):
            b = inst.iloc[start:start + BATCH]
            imgs, dids = [], []
            for _, row in b.iterrows():
                try:
                    imgs.append(load_cxr(row.cxr_path, cfg.data, is_train=False))
                    dids.append(row.dicom_id)
                except Exception as e:
                    log.warning("load fail %s: %s", row.dicom_id, e); nfail += 1
            if not imgs:
                continue
            e = enc(torch.stack(imgs).to(device)).cpu()
            for did, v in zip(dids, e):
                emb[did] = v.float()
            if start % (BATCH * 5) == 0:
                log.info("  %d/%d (fail=%d)", start + len(b), len(inst), nfail)
    torch.save(emb, os.path.join(PC, "raddino_multi_embeddings.pt"))
    log.info("Saved %d embeddings -> raddino_multi_embeddings.pt (fail=%d)", len(emb), nfail)


if __name__ == "__main__":
    main()
