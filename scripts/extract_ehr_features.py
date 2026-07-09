"""
One-time script: extract EHR (demographics + body-size) features for the aortic
cohort and cache them, mirroring the PCLR / RAD-DINO embedding-cache pattern.

For every subject in aortic_labels.csv we compute the echo date (earliest
structured-measurement datetime, same convention as build_cohort) and pull
age/sex/height/weight/bmi/bsa/bp from MIMIC-IV. Output is a per-subject CSV that
AortaDataset / the GBDT baseline look up by subject_id.

Saves: pretrained_checkpoints/ehr_features.csv

Runtime: ~1-2 min (CPU only, no GPU needed).
Run on a login node or:
    sbatch scripts/slurm_extract_ehr.sh    (if you prefer a job)
"""

import os
import sys
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# MIMIC-IV v3.1 hosp module (on cluster)
PATIENTS = "/scratch4/rsteven1/physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz"
OMR      = "/scratch4/rsteven1/physionet.org/files/mimiciv/3.1/hosp/omr.csv.gz"
OUTPUT   = "pretrained_checkpoints/ehr_features.csv"


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, "src")
    sys.path.insert(0, src_dir)

    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.ehr import build_ehr_features

    cfg = Config()
    out_path = os.path.join(project_root, OUTPUT)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- subjects + echo dates (earliest structured-measurement datetime) ---
    labels = pd.read_csv(cfg.data.echo_labels_path, usecols=["subject_id"])
    subjects = labels["subject_id"].drop_duplicates()
    log.info("Label subjects: %d", len(subjects))

    struct = pd.read_csv(
        cfg.data.echo_structured_path,
        usecols=["subject_id", "measurement_datetime"],
    )
    echo_dates = (
        struct.groupby("subject_id")["measurement_datetime"].min().reset_index()
        .rename(columns={"measurement_datetime": "echo_date"})
    )
    echo_dates["echo_date"] = pd.to_datetime(echo_dates["echo_date"], errors="coerce")
    echo_dates = subjects.to_frame().merge(echo_dates, on="subject_id", how="left")
    n_no_date = echo_dates["echo_date"].isna().sum()
    log.info("Echo dates: %d/%d subjects have a date (%d missing -> use most recent omr)",
             len(echo_dates) - n_no_date, len(echo_dates), n_no_date)

    # --- build + save ---
    feats = build_ehr_features(echo_dates, PATIENTS, OMR)
    feats.to_csv(out_path, index=False)
    log.info("Saved -> %s  (%d subjects, %d columns)", out_path, len(feats), feats.shape[1])

    # --- quick QC summary ---
    log.info("Coverage: %s",
             {c: int(feats[c].notna().sum()) for c in
              ["age", "sex", "height_cm", "weight_kg", "bmi", "bsa", "sbp"]})
    log.info("Numeric summary (cm/kg):\n%s",
             feats[["age", "height_cm", "weight_kg", "bmi", "bsa"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
