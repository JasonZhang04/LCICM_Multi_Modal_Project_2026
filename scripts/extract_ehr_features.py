"""
Extract EHR (demographics + body-size) features for the aortic cohort and cache them.

Two modes:
  --level episode  (default, episode-level rebuild)
      One row per (patient, echo study), keyed by episode_id, using only records
      AT OR BEFORE that episode's index date (causal). Reads
      pretrained_checkpoints/episodes.csv. Saves ehr_features_episode.csv.
  --level patient  (legacy)
      One row per subject at the earliest structured-measurement date, nearest-in-
      either-direction matching. Reads aortic_labels.csv. Saves ehr_features.csv.

age/sex/height/weight/bmi/bsa/bp come from MIMIC-IV hosp (patients + omr).

Runtime: a few min (CPU only). Reads omr.csv.gz (~large) via zcat.
    sbatch scripts/slurm_extract_ehr.sh
"""

import argparse
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


def run_episode(root, pc):
    from multimodal_aorta.data.episodes import load_episodes
    from multimodal_aorta.data.ehr import build_ehr_features_per_episode

    out_path = os.path.join(pc, "ehr_features_episode.csv")
    ep = load_episodes(pc, require_ecg=False)   # episode_id, subject_id, echo_dt, ...
    log.info("Episodes: %d / patients %d", len(ep), ep.subject_id.nunique())

    feats = build_ehr_features_per_episode(
        ep[["episode_id", "subject_id", "measurement_id", "echo_dt"]], PATIENTS, OMR)
    feats.to_csv(out_path, index=False)
    log.info("Saved -> %s  (%d episodes, %d columns)", out_path, len(feats), feats.shape[1])
    _qc(feats)


def run_patient(root, pc):
    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.ehr import build_ehr_features

    cfg = Config()
    out_path = os.path.join(root, "pretrained_checkpoints", "ehr_features.csv")
    labels = pd.read_csv(cfg.data.echo_labels_path, usecols=["subject_id"])
    subjects = labels["subject_id"].drop_duplicates()
    struct = pd.read_csv(cfg.data.echo_structured_path,
                         usecols=["subject_id", "measurement_datetime"])
    echo_dates = (struct.groupby("subject_id")["measurement_datetime"].min().reset_index()
                  .rename(columns={"measurement_datetime": "echo_date"}))
    echo_dates["echo_date"] = pd.to_datetime(echo_dates["echo_date"], errors="coerce")
    echo_dates = subjects.to_frame().merge(echo_dates, on="subject_id", how="left")
    feats = build_ehr_features(echo_dates, PATIENTS, OMR)
    feats.to_csv(out_path, index=False)
    log.info("Saved -> %s  (%d subjects, %d columns)", out_path, len(feats), feats.shape[1])
    _qc(feats)


def _qc(feats):
    log.info("Coverage: %s",
             {c: int(feats[c].notna().sum()) for c in
              ["age", "sex", "height_cm", "weight_kg", "bmi", "bsa", "sbp"] if c in feats})
    cols = [c for c in ["age", "height_cm", "weight_kg", "bmi", "bsa"] if c in feats]
    log.info("Numeric summary:\n%s", feats[cols].describe().round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["episode", "patient"], default="episode")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    pc = os.path.join(root, "pretrained_checkpoints")
    os.makedirs(pc, exist_ok=True)

    if args.level == "episode":
        run_episode(root, pc)
    else:
        run_patient(root, pc)


if __name__ == "__main__":
    main()
