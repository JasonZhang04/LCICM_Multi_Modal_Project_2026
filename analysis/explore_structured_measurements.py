"""
explore_structured_measurements.py

Exploration of MIMIC-IV-Echo structured_measurement.csv.
Dataset is in long/EAV format: each row is one measurement observation.
  - 8 columns: subject_id, measurement_id, measurement_datetime, test_type,
                measurement, measurement_description, result, unit
  - 649 unique measurement types across ~28M rows and ~91K subjects

Usage:
    module load python/3.9.15
    python explore_structured_measurements.py
"""

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
STRUCTURED_MEAS_CSV = "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/data/echo/structured-measurement.csv"
ECHO_STUDY_LIST_CSV = "/home/czhan182/scr4_rsteven1/rsteven1/Echo_Data/echo-study-list.csv"

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKSIZE = 500_000

# Aortic measurements of interest
AORTIC_COLS = [
    "annulus_diam",    # Aorta - Annulus Diameter (cm)
    "aortic_sinus",    # Aorta - Sinus Size Description (categorical)
    "aortic_sinus2",   # Aorta - Sinus Atheroma
    "sinus_diam",      # Aorta - Sinus Diameter (cm)  <-- aortic root
    "ascending",       # Aorta - Ascending Aorta Size Description (categorical)
    "ascending_diam",  # Aorta - Ascending Aorta Diameter (cm)
    "arch_diam",       # Aorta - Arch Diameter (cm)
    "descending_diam", # Aorta - Descending Thoracic Aorta Diameter (cm)
]

# Clinical dilation thresholds for ascending aorta (cm)
# Source: AHA/ACC 2022 guidelines — normal ≤4.0 cm (M<3.9, F<3.6; 4.0 used as sex-agnostic cutoff)
ASCENDING_DILATION_THRESHOLDS = {
    "normal":               (0.0, 4.0),
    "mildly_dilated":       (4.0, 4.5),
    "moderately_dilated":   (4.5, 5.5),
    "severely_dilated":     (5.5, 99.0),
}

# Clinical dilation thresholds for aortic root / sinus of Valsalva (cm)
# Source: ASE guidelines + aligned with MIMIC-IV categorical labels (Nl <3.7, Mild 3.7-4.4, Mod 4.5-5.4, Severe >5.4)
SINUS_DILATION_THRESHOLDS = {
    "normal":               (0.0, 3.7),
    "mildly_dilated":       (3.7, 4.5),
    "moderately_dilated":   (4.5, 5.5),
    "severely_dilated":     (5.5, 99.0),
}

# Map measurement name → its threshold dict
THRESHOLDS_BY_MEASUREMENT = {
    "ascending_diam": ASCENDING_DILATION_THRESHOLDS,
    "sinus_diam":     SINUS_DILATION_THRESHOLDS,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_full(cols=None):
    """Load entire structured_measurement.csv (chunked). Optionally filter by measurement names."""
    chunks = []
    for chunk in pd.read_csv(STRUCTURED_MEAS_CSV, chunksize=CHUNKSIZE):
        if cols:
            chunk = chunk[chunk["measurement"].isin(cols)]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def apply_dilation_label(series: pd.Series, thresholds=ASCENDING_DILATION_THRESHOLDS) -> pd.Series:
    """Map a numeric diameter Series to a categorical dilation label."""
    labels = pd.Series("unknown", index=series.index)
    for label, (lo, hi) in thresholds.items():
        labels[(series >= lo) & (series < hi)] = label
    return labels


# ── Section 1: Schema overview ────────────────────────────────────────────────

def section_schema():
    print("=" * 70)
    print("SECTION 1 — SCHEMA OVERVIEW")
    print("=" * 70)
    sample = pd.read_csv(STRUCTURED_MEAS_CSV, nrows=5)
    print(f"Columns: {sample.columns.tolist()}")
    print(f"\nSample rows:")
    print(sample.to_string(index=False))


# ── Section 2: All unique measurements ───────────────────────────────────────

def section_all_measurements():
    print("\n" + "=" * 70)
    print("SECTION 2 — ALL UNIQUE MEASUREMENTS")
    print("=" * 70)
    chunks = []
    for chunk in pd.read_csv(
        STRUCTURED_MEAS_CSV,
        chunksize=CHUNKSIZE,
        usecols=["measurement", "measurement_description", "test_type", "unit"],
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    unique = (
        df[["measurement", "measurement_description", "test_type", "unit"]]
        .drop_duplicates()
        .sort_values("measurement")
    )
    print(f"Total rows: {len(df):,}")
    print(f"Total unique measurement types: {len(unique['measurement'].unique())}")
    print()
    print(unique.to_string(index=False))


# ── Section 3: Aortic measurements summary ───────────────────────────────────

def section_aortic_summary():
    print("\n" + "=" * 70)
    print("SECTION 3 — AORTIC MEASUREMENTS SUMMARY")
    print("=" * 70)
    df = load_full(cols=AORTIC_COLS)
    print(f"Total aortic-related rows: {len(df):,}")
    print(f"Unique subjects with any aortic measurement: {df['subject_id'].nunique():,}")
    print()

    for col in AORTIC_COLS:
        sub = df[df["measurement"] == col]
        if len(sub) == 0:
            print(f"[{col}] — no data found")
            continue
        desc = sub["measurement_description"].iloc[0]
        print(f"--- {col}  ({desc}) ---")
        print(f"  Rows: {len(sub):,} | Subjects: {sub['subject_id'].nunique():,}")

        numeric = pd.to_numeric(sub["result"], errors="coerce").dropna()
        numeric = numeric[numeric > 0]
        
        if len(numeric) > 0:
            print(
                f"  Numeric: {len(numeric):,} values | "
                f"Range: {numeric.min():.2f} – {numeric.max():.2f} cm | "
                f"Median: {numeric.median():.2f} cm"
            )
        else:
            print(f"  Categorical top values: {sub['result'].value_counts().head(8).to_dict()}")
        print()


# ── Section 4: Dilation label distribution ───────────────────────────────────

def section_dilation_labels():
    print("\n" + "=" * 70)
    print("SECTION 4 — DILATION LABEL DISTRIBUTION")
    print("=" * 70)
    df = load_full(cols=["ascending", "aortic_sinus", "ascending_diam", "sinus_diam"])

    # Categorical labels
    for col in ["ascending", "aortic_sinus"]:
        sub = df[df["measurement"] == col]
        print(f"\n{col} (categorical):")
        print(sub["result"].value_counts().to_string())

    # Numeric: ascending_diam (AHA/ACC thresholds, normal ≤4.0 cm)
    print("\n--- ascending_diam (AHA/ACC: normal ≤4.0 cm) ---")
    asc = df[df["measurement"] == "ascending_diam"].copy()
    asc["result_num"] = pd.to_numeric(asc["result"], errors="coerce")
    asc = asc[(asc["result_num"].notna()) & (asc["result_num"] > 0)]
    asc["label"] = apply_dilation_label(asc["result_num"], ASCENDING_DILATION_THRESHOLDS)
    print(asc["label"].value_counts().to_string())

    # Numeric: sinus_diam / aortic root (ASE thresholds, normal <3.7 cm)
    print("\n--- sinus_diam / aortic root (ASE: normal <3.7 cm) ---")
    sin = df[df["measurement"] == "sinus_diam"].copy()
    sin["result_num"] = pd.to_numeric(sin["result"], errors="coerce")
    sin = sin[(sin["result_num"].notna()) & (sin["result_num"] > 0)]
    sin["label"] = apply_dilation_label(sin["result_num"], SINUS_DILATION_THRESHOLDS)
    print(sin["label"].value_counts().to_string())


# ── Section 5: Overlap with DICOM dataset ────────────────────────────────────

def section_dicom_overlap():
    print("\n" + "=" * 70)
    print("SECTION 5 — OVERLAP WITH ECHO DICOM DATASET")
    print("=" * 70)
    echo_studies = pd.read_csv(ECHO_STUDY_LIST_CSV)
    dicom_subjects = set(echo_studies["subject_id"].unique())
    print(f"DICOM subjects total: {len(dicom_subjects):,}")

    df = load_full(cols=["ascending_diam", "sinus_diam"])
    overlap = df[df["subject_id"].isin(dicom_subjects)]
    print(f"DICOM subjects with aortic structured measurements: {overlap['subject_id'].nunique():,}")

    for col in ["ascending_diam", "sinus_diam"]:
        thresholds = THRESHOLDS_BY_MEASUREMENT[col]
        sub = overlap[overlap["measurement"] == col].copy()
        sub["result_num"] = pd.to_numeric(sub["result"], errors="coerce")
        sub = sub[(sub["result_num"].notna()) & (sub["result_num"] > 0)]
        sub["label"] = apply_dilation_label(sub["result_num"], thresholds)
        severity_order = list(thresholds.keys())
        sub["severity_rank"] = sub["label"].map({k: i for i, k in enumerate(severity_order)})
        per_subject = sub.loc[sub.groupby("subject_id")["severity_rank"].idxmax(), ["subject_id", "label"]]
        print(f"\n{col} — per-subject worst label (DICOM cohort only):")
        print(per_subject["label"].value_counts().to_string())
        print(f"  Any dilation: {per_subject[per_subject['label'].isin(['mildly_dilated','moderately_dilated','severely_dilated'])].shape[0]}")
        print(f"  Moderate+:    {per_subject[per_subject['label'].isin(['moderately_dilated','severely_dilated'])].shape[0]}")


# ── Section 6: Generate label table ──────────────────────────────────────────

def generate_label_table(output_path=None):
    """
    Build a per-subject label table with aortic diameters and dilation labels.
    Returns a DataFrame; optionally saves to CSV if output_path is provided.

    Columns: subject_id, ascending_diam_cm, ascending_label,
             sinus_diam_cm, sinus_label, any_dilation
    """
    echo_studies = pd.read_csv(ECHO_STUDY_LIST_CSV)
    dicom_subjects = set(echo_studies["subject_id"].unique())

    df = load_full(cols=["ascending_diam", "sinus_diam"])
    df = df[df["subject_id"].isin(dicom_subjects)]

    result_frames = {}
    for col in ["ascending_diam", "sinus_diam"]:
        thresholds = THRESHOLDS_BY_MEASUREMENT[col]
        rank_map = {k: i for i, k in enumerate(thresholds.keys())}
        sub = df[df["measurement"] == col].copy()
        sub["result_num"] = pd.to_numeric(sub["result"], errors="coerce")
        sub = sub[(sub["result_num"].notna()) & (sub["result_num"] > 0)]
        sub["label"] = apply_dilation_label(sub["result_num"], thresholds)
        sub["severity_rank"] = sub["label"].map(rank_map)
        worst = sub.loc[sub.groupby("subject_id")["severity_rank"].idxmax(),
                        ["subject_id", "result_num", "label"]]
        worst.columns = ["subject_id", f"{col}_cm", f"{col}_label"]
        result_frames[col] = worst.set_index("subject_id")

    label_df = pd.DataFrame({"subject_id": list(dicom_subjects)}).set_index("subject_id")
    for col_df in result_frames.values():
        label_df = label_df.join(col_df, how="left")

    label_df["any_dilation"] = (
        label_df.get("ascending_diam_label", pd.Series()).isin(
            ["mildly_dilated", "moderately_dilated", "severely_dilated"]
        ) |
        label_df.get("sinus_diam_label", pd.Series()).isin(
            ["mildly_dilated", "moderately_dilated", "severely_dilated"]
        )
    )

    label_df = label_df.reset_index()
    print(f"\nLabel table shape: {label_df.shape}")
    print(label_df.head(10).to_string(index=False))

    if output_path:
        label_df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

    return label_df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Run all sections by default; pass section numbers as args to run specific ones
    # e.g.  python explore_structured_measurements.py 1 3 5
    sections = {
        "1": section_schema,
        "2": section_all_measurements,
        "3": section_aortic_summary,
        "4": section_dilation_labels,
        "5": section_dicom_overlap,
        "6": lambda: generate_label_table(
            output_path="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/data/echo/aortic_labels.csv"
        ),
    }

    requested = sys.argv[1:] if len(sys.argv) > 1 else list(sections.keys())
    for key in requested:
        if key in sections:
            sections[key]()
        else:
            print(f"Unknown section '{key}'. Valid: {list(sections.keys())}")
