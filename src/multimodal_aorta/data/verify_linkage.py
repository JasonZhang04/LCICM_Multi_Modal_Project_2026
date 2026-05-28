"""
Quick smoke-test for cohort linkage and splits.
Run with: python -m multimodal_aorta.data.verify_linkage
from the src/ directory.

Does NOT load any waveforms or images — only tests the tabular linkage.
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)

import pandas as pd

from multimodal_aorta.configs.default_config import Config
from multimodal_aorta.data.dataset import build_cohort
from multimodal_aorta.data.splits import make_splits


def main():
    cfg = Config()
    dc = cfg.data

    cohort = build_cohort(
        echo_labels_path=dc.echo_labels_path,
        echo_structured_path=dc.echo_structured_path,
        ecg_root=dc.ecg_root,
        cxr_record_list=dc.cxr_record_list,
        cxr_png_root=dc.cxr_png_root,
        cxr_dicom_root=dc.cxr_dicom_root,
        cxr_metadata_cache=dc.cxr_metadata_cache,
        col_root=dc.col_root,
        col_asc=dc.col_asc,
        root_range=dc.root_range,
        asc_range=dc.asc_range,
        max_days_offset=dc.max_days_offset,
        cxr_extra_png_root=dc.cxr_extra_png_root,
    )

    print("\n=== Cohort head ===")
    print(cohort.head(5).to_string())

    train_ids, val_ids, test_ids = make_splits(
        cohort,
        col_root="target_root",
        train_frac=dc.train_frac,
        val_frac=dc.val_frac,
        n_bins=dc.stratify_bins,
        seed=dc.split_seed,
    )

    print("\n=== Splits ===")
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")

    # Verify no leakage
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    print("No patient leakage detected across splits.")

    # Check label distribution per split
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        sub = cohort[cohort["subject_id"].isin(ids)]
        for col in ["target_root", "target_asc"]:
            vals = sub[col].dropna()
            print(
                f"  [{split_name}] {col}: n={len(vals)} "
                f"mean={vals.mean():.3f} std={vals.std():.3f}"
            )


if __name__ == "__main__":
    main()
