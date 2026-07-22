"""
M4c — extract RAD-DINO embeddings on an AORTA/MEDIASTINUM ROI crop.

M3 showed the whole-image RAD-DINO CLS token carries no aortic-size signal beyond
body size — plausibly because a global chest embedding dilutes the aortic
silhouette. This re-runs the frozen encoder on an upper-mediastinum ROI (the
aortic arch / knob + ascending-aorta contour region) so the embedding is focused
on the aorta.

Mirrors extract_raddino_embeddings.py exactly, except each image is cropped to
ROI (fractional box) before the encoder's resize. Whole-image extraction is left
untouched, so the two embedding sets are directly comparable.

Saves: pretrained_checkpoints/raddino_roi_embeddings.pt -> {subject_id: tensor(768,)}
Submit via SLURM: sbatch scripts/slurm_extract_raddino_roi.sh
"""

import os
import sys
import logging

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_PATH = "pretrained_checkpoints/raddino_roi_embeddings.pt"
BATCH = 32

# Upper-mediastinum ROI (fractional: left, upper, right, lower). Captures the
# aortic arch/knob and ascending-aorta contour while excluding lung periphery and
# the lower cardiac/abdominal field. Heuristic starting point (blueprint §10.3);
# anatomy-based segmentation would be a later refinement.
ROI = (0.25, 0.08, 0.75, 0.60)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_root, "src"))

    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.dataset import build_cohort
    from multimodal_aorta.data.preprocessing import load_cxr
    from multimodal_aorta.models.cxr_encoder import CXREncoder

    out_path = os.path.join(project_root, OUTPUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    log.info("Building aortic cohort to identify per-patient CXR paths...")
    cfg = Config()
    cohort = build_cohort(
        echo_labels_path=cfg.data.echo_labels_path,
        echo_structured_path=cfg.data.echo_structured_path,
        ecg_root=cfg.data.ecg_root,
        cxr_record_list=cfg.data.cxr_record_list,
        cxr_png_root=cfg.data.cxr_png_root,
        cxr_dicom_root=cfg.data.cxr_dicom_root,
        cxr_metadata_cache=cfg.data.cxr_metadata_cache,
        col_root=cfg.data.col_root,
        col_asc=cfg.data.col_asc,
        root_range=cfg.data.root_range,
        asc_range=cfg.data.asc_range,
        max_days_offset=cfg.data.max_days_offset,
        cxr_extra_png_root=cfg.data.cxr_extra_png_root,
        cxr_metadata_csv=cfg.data.cxr_metadata_csv,
    )

    cxr_cohort = cohort[cohort["has_cxr"]].reset_index(drop=True)
    log.info("Cohort patients with CXR: %d | ROI=%s", len(cxr_cohort), ROI)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading RAD-DINO (%s) on %s", cfg.model.cxr_model_name, device)
    encoder = CXREncoder(model_name=cfg.model.cxr_model_name, freeze_blocks=12)
    encoder.eval().to(device)

    embeddings = {}
    n_failed = 0
    with torch.no_grad():
        for start in range(0, len(cxr_cohort), BATCH):
            batch = cxr_cohort.iloc[start:start + BATCH]
            imgs, sids = [], []
            for _, row in batch.iterrows():
                try:
                    img = load_cxr(row["cxr_path"], cfg.data, is_train=False, roi=ROI)
                except Exception as e:  # noqa: BLE001
                    log.warning("CXR load failed for %s: %s", row["cxr_path"], e)
                    n_failed += 1
                    continue
                imgs.append(img)
                sids.append(int(row["subject_id"]))
            if not imgs:
                continue
            x = torch.stack(imgs).to(device)
            emb = encoder(x).cpu()
            for sid, e in zip(sids, emb):
                embeddings[sid] = e.float()
            log.info("  %d / %d processed  (%d failed)",
                     start + len(batch), len(cxr_cohort), n_failed)

    log.info("Done: %d ROI embeddings, %d failed", len(embeddings), n_failed)
    torch.save(embeddings, out_path)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
