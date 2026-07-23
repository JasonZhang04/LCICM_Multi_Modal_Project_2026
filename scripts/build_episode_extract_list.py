"""
Build the per-dicom CXR extraction list for the episode cohort.

The feature extractors (extract_raddino_patchpool.py, extract_cxr_geometry_features.py,
extract_multi_anatomy_roi_embeddings.py, ...) operate per UNIQUE dicom_id and expect
an instances CSV with columns:
    subject_id, dicom_id, view_position, days_off, cxr_path

episode_cxr_instances.csv (from build_episode_cohort.py) is an (episode, dicom_id)
mapping without paths/views. This script resolves the unique dicom_ids in that mapping
to their local JPG path (via the official cxr record list) and view (via the metadata
CSV), and writes `pretrained_checkpoints/cxr_instances_episode.csv` in the extractor
schema. Point the extractors at it with:  CXR_INSTANCES=cxr_instances_episode.csv

Rows whose JPG is not yet on disk are written too (cxr_path points where it WILL be),
but a `on_disk` column flags which are ready; extractors skip missing files anyway.

Run: python scripts/build_episode_extract_list.py
"""

import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJ = "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
PC = os.path.join(PROJ, "pretrained_checkpoints")
CXR_RECS = "/scratch4/rsteven1/MIMIC_CXR_GS/cxr-record-list.csv"
CXR_META = "/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/mimic-cxr-2.0.0-metadata.csv.gz"
JPG_ROOT = "/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/files"


def main():
    inst = pd.read_csv(os.path.join(PC, "episode_cxr_instances.csv"))
    inst["dicom_id"] = inst["dicom_id"].astype(str)
    # collapse (episode, dicom) -> unique dicom, keeping the smallest pre-index offset
    per_dicom = (inst.groupby("dicom_id")
                 .agg(subject_id=("subject_id", "first"),
                      days_off=("days_before_echo", "min")).reset_index())
    log.info("Unique dicoms to extract: %d (from %d episode-instance rows)", len(per_dicom), len(inst))

    rec = pd.read_csv(CXR_RECS, usecols=["dicom_id", "path"])
    rec["dicom_id"] = rec["dicom_id"].astype(str)
    per_dicom = per_dicom.merge(rec, on="dicom_id", how="left")
    n_unres = per_dicom["path"].isna().sum()
    if n_unres:
        log.warning("%d dicoms have no record-list path; dropping", n_unres)
        per_dicom = per_dicom[per_dicom["path"].notna()]

    jpg_rel = per_dicom["path"].str.replace(r"\.dcm$", ".jpg", regex=True)
    per_dicom["cxr_path"] = JPG_ROOT + "/" + jpg_rel.str.replace(r"^files/", "", regex=True)

    meta = pd.read_csv(CXR_META, usecols=["dicom_id", "ViewPosition"])
    meta["dicom_id"] = meta["dicom_id"].astype(str)
    per_dicom = per_dicom.merge(meta, on="dicom_id", how="left").rename(
        columns={"ViewPosition": "view_position"})

    per_dicom["on_disk"] = [os.path.exists(p) for p in per_dicom["cxr_path"]]
    out = per_dicom[["subject_id", "dicom_id", "view_position", "days_off", "cxr_path", "on_disk"]]
    out_path = os.path.join(PC, "cxr_instances_episode.csv")
    out.to_csv(out_path, index=False)
    log.info("Wrote %s (%d dicoms, %d on disk now, %d pending download)",
             out_path, len(out), int(out.on_disk.sum()), int((~out.on_disk).sum()))
    log.info("View mix: %s", out.view_position.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
