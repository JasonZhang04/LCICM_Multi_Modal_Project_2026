"""
One-time script: extract PCLR embeddings for the aortic cohort ECGs only.

Uses build_cohort() to find the exact ECG file for each of the ~2,902
cohort patients (same temporal matching as the supervised pipeline), then
runs the PCLR TF model to produce a 320-dim embedding per patient.

Saves: pretrained_checkpoints/pclr_embeddings.pt  →  {subject_id: tensor(320,)}

Runtime: ~3-5 min on CPU, ~1-2 min on GPU node.
Submit via SLURM (recommended):
    sbatch scripts/slurm_extract_pclr.sh
Or run interactively on a GPU node:
    srun --partition=a100 --gres=gpu:0 --cpus-per-task=4 --mem=16G --pty bash
    python scripts/extract_pclr_embeddings.py
"""

import os
import sys
import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PCLR_MODEL_PATH = "/scratch4/rsteven1/EHRBind/PCLR/PCLR.h5"
OUTPUT_PATH     = "pretrained_checkpoints/pclr_embeddings.pt"

# PCLR lead order — must match training convention
PCLR_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_SAMPLES = 4096


# ---------------------------------------------------------------------------
# PCLR preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_pclr(wfdb_path: str):
    """Load a WFDB ECG and return (4096, 12) float32, or None on failure."""
    try:
        import wfdb
        rec = wfdb.rdrecord(wfdb_path)
        sig = rec.p_signal        # (n_samples, n_leads)
        sig_names = [s.strip() for s in rec.sig_name]
        fs = rec.fs
    except Exception:
        return None

    if sig is None or sig.ndim != 2:
        return None

    if fs != 500:
        from scipy.signal import resample
        sig = resample(sig, int(round(sig.shape[0] * 500 / fs)), axis=0)

    out = np.zeros((ECG_SAMPLES, 12), dtype=np.float32)
    for i, lead in enumerate(PCLR_LEADS):
        lower = [s.lower() for s in sig_names]
        col = (sig_names.index(lead) if lead in sig_names
               else lower.index(lead.lower()) if lead.lower() in lower
               else None)
        if col is None:
            continue
        raw = sig[:, col].astype(np.float64)
        nan_mask = np.isnan(raw)
        if nan_mask.all():
            continue
        if nan_mask.any():
            raw[nan_mask] = 0.0
        out[:, i] = np.interp(
            np.linspace(0, 1, ECG_SAMPLES),
            np.linspace(0, 1, len(raw)),
            raw,
        ).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Add src/ to path so we can import the pipeline ────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, "src")
    sys.path.insert(0, src_dir)

    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.dataset import build_cohort

    os.makedirs(os.path.dirname(os.path.join(project_root, OUTPUT_PATH)), exist_ok=True)
    out_path = os.path.join(project_root, OUTPUT_PATH)

    # ── 1. Build cohort to get the exact ECG path per patient ─────────────
    # This replicates the temporal matching in the supervised pipeline so the
    # embedding corresponds to the same ECG used during training.
    log.info("Building aortic cohort to identify per-patient ECG paths...")
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

    # Keep only patients that have an ECG
    ecg_cohort = cohort[cohort["has_ecg"]].reset_index(drop=True)
    log.info("Cohort patients with ECG: %d (out of %d total)", len(ecg_cohort), len(cohort))

    # ── 2. Load PCLR model (TF, CPU) ──────────────────────────────────────
    log.info("Loading PCLR model from %s", PCLR_MODEL_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU, no conflict with GPU jobs
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    model = tf.keras.models.load_model(PCLR_MODEL_PATH, compile=False)
    log.info("PCLR model loaded. Processing %d patients...", len(ecg_cohort))

    # ── 3. Extract embeddings in batches ──────────────────────────────────
    BATCH = 64
    embeddings = {}
    n_failed = 0

    for start in range(0, len(ecg_cohort), BATCH):
        batch = ecg_cohort.iloc[start : start + BATCH]
        batch_ecgs, batch_sids = [], []

        for _, row in batch.iterrows():
            ecg = preprocess_for_pclr(row["ecg_path"])
            if ecg is None:
                n_failed += 1
                continue
            batch_ecgs.append(ecg)
            batch_sids.append(int(row["subject_id"]))

        if not batch_ecgs:
            continue

        preds = model.predict(np.stack(batch_ecgs), verbose=0)  # (B, 320)
        for sid, emb in zip(batch_sids, preds):
            embeddings[sid] = emb.astype(np.float32)

        log.info(
            "  %d / %d processed  (%d failed)",
            start + len(batch), len(ecg_cohort), n_failed,
        )

    log.info("Done: %d embeddings extracted, %d failed", len(embeddings), n_failed)

    # ── 4. Save as PyTorch .pt ─────────────────────────────────────────────
    import torch
    torch.save({sid: torch.from_numpy(arr) for sid, arr in embeddings.items()}, out_path)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
