"""
One-time script: extract RAD-DINO embeddings for the aortic cohort CXRs only.

Mirrors scripts/extract_pclr_embeddings.py on the imaging side. Uses
build_cohort() to find the exact best-view CXR for each cohort patient with a
chest X-ray (same temporal/view matching as the supervised pipeline), runs the
frozen RAD-DINO ViT once, and stores a 768-dim CLS embedding per patient.

CXR preprocessing is reused verbatim from the training pipeline
(preprocessing.load_cxr with is_train=False) so the stored embedding is exactly
what the supervised model would have seen — no train/extract skew.

Saves: pretrained_checkpoints/raddino_embeddings.pt  ->  {subject_id: tensor(768,)}

Submit via SLURM (recommended):
    sbatch scripts/slurm_extract_raddino.sh
"""

import os
import sys
import logging

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_PATH = "pretrained_checkpoints/raddino_embeddings.pt"
BATCH = 32


def main():
    # ── Add src/ to path so we can import the pipeline ────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, "src")
    sys.path.insert(0, src_dir)

    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.dataset import build_cohort
    from multimodal_aorta.data.preprocessing import load_cxr
    from multimodal_aorta.models.cxr_encoder import CXREncoder

    out_path = os.path.join(project_root, OUTPUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── 1. Build cohort to get the exact best-view CXR path per patient ───
    log.info("Building aortic cohort to identify per-patient CXR paths...")
    cfg = Config()
    cohort = build_cohort(
        echo_labels_path     = cfg.data.echo_labels_path,
        echo_structured_path = cfg.data.echo_structured_path,
        ecg_root             = cfg.data.ecg_root,
        cxr_record_list      = cfg.data.cxr_record_list,
        cxr_png_root         = cfg.data.cxr_png_root,
        cxr_dicom_root       = cfg.data.cxr_dicom_root,
        cxr_metadata_cache   = cfg.data.cxr_metadata_cache,
        col_root             = cfg.data.col_root,
        col_asc              = cfg.data.col_asc,
        root_range           = cfg.data.root_range,
        asc_range            = cfg.data.asc_range,
        max_days_offset      = cfg.data.max_days_offset,
        cxr_extra_png_root   = cfg.data.cxr_extra_png_root,
        cxr_metadata_csv     = cfg.data.cxr_metadata_csv,
    )

    # Keep only patients that have a CXR
    cxr_cohort = cohort[cohort["has_cxr"]].reset_index(drop=True)
    log.info("Cohort patients with CXR: %d (out of %d total)", len(cxr_cohort), len(cohort))

    # ── 2. Load frozen RAD-DINO ───────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading RAD-DINO (%s) on %s", cfg.model.cxr_model_name, device)
    encoder = CXREncoder(model_name=cfg.model.cxr_model_name, freeze_blocks=12)
    encoder.eval().to(device)

    # ── 3. Extract embeddings in batches ──────────────────────────────────
    embeddings = {}
    n_failed = 0

    with torch.no_grad():
        for start in range(0, len(cxr_cohort), BATCH):
            batch = cxr_cohort.iloc[start : start + BATCH]
            imgs, sids = [], []

            for _, row in batch.iterrows():
                try:
                    img = load_cxr(row["cxr_path"], cfg.data, is_train=False)  # (3, H, W)
                except Exception as e:
                    log.warning("CXR load failed for %s: %s", row["cxr_path"], e)
                    n_failed += 1
                    continue
                imgs.append(img)
                sids.append(int(row["subject_id"]))

            if not imgs:
                continue

            x = torch.stack(imgs).to(device)        # (B, 3, H, W)
            emb = encoder(x).cpu()                  # (B, 768)
            for sid, e in zip(sids, emb):
                embeddings[sid] = e.float()

            log.info("  %d / %d processed  (%d failed)",
                     start + len(batch), len(cxr_cohort), n_failed)

    log.info("Done: %d embeddings extracted, %d failed", len(embeddings), n_failed)

    # ── 4. Save ───────────────────────────────────────────────────────────
    torch.save(embeddings, out_path)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
