#!/usr/bin/env python3
"""
Step 2: Extract Aortic Dilation Labels from Echo Reports

Reads the JHU metadata (dbo.derived_cardiology_echo_procedures.csv),
extracts aortic dilation labels from the 'narrative' and 'impression'
columns, and produces a CSV mapping accession numbers to labels and
DICOM file paths.

The key columns in the JHU metadata are:
  - acc_num:     accession number (links to DICOM files)
  - narrative:   full echo report text (findings, measurements, etc.)
  - impression:  summary/conclusions section of the report

The label extractor searches both narrative and impression for:
  1. Explicit negative mentions ("aortic root normal", "no dilation")
  2. Numeric aortic root diameter (>= 4.0 cm = dilated)
  3. Severity keywords ("severely dilated aorta", etc.)
  4. General positive keywords ("aortic dilation", etc.)
  5. No mention -> default to normal (low confidence)

Usage:
    python scripts/02_extract_labels.py [--config configs/default.yaml]
"""

import os
import sys
import json
import argparse
import logging

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.label_extractor import AorticDilationLabeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_metadata_reports(metadata_dir: str) -> pd.DataFrame:
    """
    Load echo reports from the JHU metadata directory.

    The main file is dbo.derived_cardiology_echo_procedures.csv (~4 GB).
    We read it in chunks to avoid memory issues, and extract only the
    columns we need: acc_num, narrative, impression.

    We combine 'narrative' and 'impression' into a single 'report_text'
    field because:
      - 'narrative' contains the full report with detailed findings and
        numeric measurements (e.g., "Aorta at Sinuses Diameter 2.88 cm")
      - 'impression' contains the summary/conclusions
      - Searching both gives us the best chance of finding aortic dilation info

    Returns:
        DataFrame with columns [accession, report_text]
    """
    # This is the file that contains echo reports
    procedures_file = os.path.join(metadata_dir, "dbo.derived_cardiology_echo_procedures.csv")

    if not os.path.exists(procedures_file):
        logger.error(f"Procedures file not found: {procedures_file}")
        return pd.DataFrame(columns=["accession", "report_text"])

    file_size_gb = os.path.getsize(procedures_file) / (1024**3)
    logger.info(f"Reading {procedures_file} ({file_size_gb:.1f} GB)")
    logger.info("Reading in chunks to manage memory...")

    # Only read the columns we need — this saves a lot of memory
    # since the full CSV has 26 columns and is ~4 GB
    use_cols = ["acc_num", "narrative", "impression"]

    all_chunks = []
    chunk_size = 50000  # rows per chunk

    try:
        for i, chunk in enumerate(pd.read_csv(
            procedures_file,
            usecols=use_cols,
            chunksize=chunk_size,
            dtype=str,           # read everything as string to avoid type issues
            on_bad_lines="skip", # skip malformed rows
        )):
            # Drop rows where both narrative and impression are empty
            chunk = chunk.dropna(subset=["narrative", "impression"], how="all")

            # Drop rows without accession number (can't link to DICOM)
            chunk = chunk.dropna(subset=["acc_num"])

            all_chunks.append(chunk)

            if (i + 1) % 10 == 0:
                total_so_far = sum(len(c) for c in all_chunks)
                logger.info(f"  Read {(i+1) * chunk_size:,} rows, kept {total_so_far:,} with reports...")

    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        if not all_chunks:
            return pd.DataFrame(columns=["accession", "report_text"])

    # Combine all chunks
    df = pd.concat(all_chunks, ignore_index=True)
    logger.info(f"Total rows with reports: {len(df):,}")

    # Combine narrative + impression into one report_text field.
    # Some rows might have only one or the other, so we handle NaN.
    # We put narrative first since it has the detailed measurements,
    # then impression which has the summary.
    df["narrative"] = df["narrative"].fillna("")
    df["impression"] = df["impression"].fillna("")
    df["report_text"] = df["narrative"] + "\n\n" + df["impression"]

    # Clean up: strip whitespace and \r\n artifacts
    df["report_text"] = df["report_text"].str.replace(r"\r\n", " ", regex=True)
    df["report_text"] = df["report_text"].str.replace(r"\s+", " ", regex=True)
    df["report_text"] = df["report_text"].str.strip()

    # Rename acc_num to accession for consistency with rest of pipeline
    df = df.rename(columns={"acc_num": "accession"})

    # Drop duplicates — same accession might appear multiple times
    df = df.drop_duplicates(subset=["accession"])

    logger.info(f"Unique studies with reports: {len(df):,}")

    # Keep only accession and report_text
    return df[["accession", "report_text"]]


def link_accessions_to_dicoms(reports_df: pd.DataFrame, accession_index_path: str) -> pd.DataFrame:
    """
    Link accession numbers from reports to DICOM file paths.

    The accession index (built by 01_explore_metadata.py) maps each
    accession number to a list of DICOM file paths. One echo study
    can have many DICOM files (different views, clips). For now we
    pick the first one — this can be refined later to select the
    best view (e.g., parasternal long axis for aortic root).

    Args:
        reports_df: DataFrame with 'accession' column.
        accession_index_path: Path to JSON mapping accession -> [dicom_paths].

    Returns:
        DataFrame with added 'dicom_path' column.
    """
    if not os.path.exists(accession_index_path):
        logger.warning(
            f"Accession index not found at {accession_index_path}. "
            f"Run 01_explore_metadata.py first to build it."
        )
        reports_df["dicom_path"] = None
        return reports_df

    with open(accession_index_path) as f:
        acc_index = json.load(f)

    logger.info(f"Accession index has {len(acc_index)} entries")

    # For each accession, pick the first DICOM file
    # TODO: later, select the DICOM with the most frames or the
    # parasternal long-axis view for better aortic root visibility
    dicom_paths = []
    for acc in reports_df["accession"]:
        paths = acc_index.get(str(acc), [])
        if paths:
            dicom_paths.append(paths[0])
        else:
            dicom_paths.append(None)

    reports_df["dicom_path"] = dicom_paths

    n_linked = reports_df["dicom_path"].notna().sum()
    logger.info(f"Linked {n_linked}/{len(reports_df)} reports to DICOM files")

    return reports_df


def main():
    parser = argparse.ArgumentParser(description="Extract aortic dilation labels")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    label_cfg = config["labels"]

    # 1. Load echo reports from metadata
    logger.info("=" * 60)
    logger.info("STEP 1: Loading echo reports from metadata")
    logger.info("=" * 60)
    reports_df = load_metadata_reports(data_cfg["jhu_metadata_dir"])

    if reports_df.empty:
        logger.error("No reports to process. Exiting.")
        sys.exit(1)

    # 2. Extract labels
    # The labeler looks at each report and determines:
    #   - Is there aortic dilation? (binary: 0 = normal, 1 = dilated)
    #   - How confident are we? (high / medium / low)
    #   - What evidence triggered the label? (keyword or measurement)
    logger.info("=" * 60)
    logger.info("STEP 2: Extracting aortic dilation labels from reports")
    logger.info("=" * 60)
    labeler = AorticDilationLabeler(label_cfg)
    reports_list = reports_df.to_dict("records")
    label_results = labeler.batch_extract(reports_list)

    # 3. Merge labels back into the dataframe
    labels_df = pd.DataFrame(label_results)
    merged = reports_df.merge(labels_df, on="accession", how="left")

    # 4. Link to DICOM files using the accession index
    logger.info("=" * 60)
    logger.info("STEP 3: Linking accessions to DICOM files")
    logger.info("=" * 60)
    accession_index_path = os.path.join(
        os.path.dirname(data_cfg["jhu_echo_dir"]), "accession_index.json"
    )
    merged = link_accessions_to_dicoms(merged, accession_index_path)

    # 5. Save the final labels CSV
    # This CSV is what the training script (03_train.py) reads.
    # Each row = one echo study with: accession, label, confidence,
    # evidence, dicom_path, and the original report text.
    output_path = data_cfg["labels_cache"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info(f"Labels saved to {output_path}")

    # 6. Print summary
    logger.info("\n" + "=" * 60)
    logger.info("LABEL EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total reports processed: {len(merged)}")
    logger.info(f"\nLabel distribution:")
    logger.info(f"{merged['label'].value_counts().to_string()}")
    logger.info(f"\nConfidence distribution:")
    logger.info(f"{merged['confidence'].value_counts().to_string()}")
    logger.info(f"\nReports linked to DICOM: {merged['dicom_path'].notna().sum()}")

    # Show some examples so you can sanity-check the labels
    logger.info("\n--- Sample POSITIVE cases (aortic dilation detected) ---")
    positives = merged[merged["label"] == 1].head(5)
    for _, row in positives.iterrows():
        logger.info(f"  Accession: {row['accession']}")
        logger.info(f"  Evidence:  {row.get('evidence', 'N/A')}")
        logger.info(f"  Severity:  {row.get('severity', 'N/A')}")
        logger.info(f"  Confidence: {row.get('confidence', 'N/A')}")
        logger.info(f"  DICOM:     {row.get('dicom_path', 'N/A')}")
        logger.info("")

    logger.info("--- Sample NEGATIVE cases (normal aorta) ---")
    negatives = merged[(merged["label"] == 0) & (merged["confidence"] == "high")].head(3)
    for _, row in negatives.iterrows():
        logger.info(f"  Accession: {row['accession']}")
        logger.info(f"  Evidence:  {row.get('evidence', 'N/A')}")
        logger.info("")

    logger.info("Done! Review the summary above.")
    logger.info(f"Next step: python scripts/03_train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()