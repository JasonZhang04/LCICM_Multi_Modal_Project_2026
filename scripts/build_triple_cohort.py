"""
Build and cache the v3 triple-modality cohort (ECG + CXR + echo-label, ~522 pts).

Runs the same linkage as training, restricts to cohort_mode='triple', attaches
the ordinal/binary grade columns, and saves a single CSV that downstream baselines
(GBDT, deep fusion) load directly — so the ~1 min cohort build happens once.

Saves: pretrained_checkpoints/cohort_triple.csv

Run: python scripts/build_triple_cohort.py
"""

import os
import sys
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT = "pretrained_checkpoints/cohort_triple.csv"


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))

    from multimodal_aorta.configs.default_config import Config
    from multimodal_aorta.data.dataset import build_cohort
    from multimodal_aorta.data.targets import add_grade_columns, GRADE_NAMES
    from multimodal_aorta.data.splits import make_cv_folds

    cfg = Config()
    out_path = os.path.join(root, OUTPUT)

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
        cohort_mode          = "triple",
    )

    cohort = add_grade_columns(cohort)
    cohort.to_csv(out_path, index=False)
    log.info("Saved triple cohort -> %s (%d patients)", out_path, len(cohort))

    # --- Class distribution (the n=522 reality check) ---
    for site, col in [("ROOT", "grade_root"), ("ASCENDING", "grade_asc")]:
        vc = cohort[col].value_counts().sort_index()
        log.info("%s grade counts:", site)
        for g, name in enumerate(GRADE_NAMES):
            log.info("   %d %-9s : %d", g, name, int(vc.get(float(g), 0)))
    log.info("Patient-level anyAD (>=4.0 either site): %d / %d positive",
             int(cohort["anyAD"].sum()), len(cohort))

    # --- Fold sanity (positives per fold) ---
    make_cv_folds(cohort, stratify_col="anyAD", n_splits=5, seed=cfg.data.split_seed)


if __name__ == "__main__":
    main()
