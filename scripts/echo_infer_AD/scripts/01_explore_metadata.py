#!/usr/bin/env python3
"""
Step 1: Explore JHU Echo Dataset

Scans the JHU echo dataset directory, builds an accession number index,
and explores the metadata files to understand what's available.

Run this first to understand the data structure before label extraction.

Usage:
    python scripts/01_explore_metadata.py
"""

import os
import sys
import json
import logging
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dicom_utils import (
    find_dicom_files,
    get_accession_number,
    get_dicom_metadata,
    build_accession_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def explore_directory_structure(root_dir: str, max_depth: int = 3):
    """Print the directory structure up to max_depth."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Directory structure of: {root_dir}")
    logger.info(f"{'='*60}")

    for depth, (dirpath, dirnames, filenames) in enumerate(os.walk(root_dir)):
        if depth >= max_depth:
            break
        level = dirpath.replace(root_dir, "").count(os.sep)
        indent = " " * 2 * level
        logger.info(f"{indent}{os.path.basename(dirpath)}/")
        if level < max_depth - 1:
            sub_indent = " " * 2 * (level + 1)
            # Show first few files
            for f in filenames[:5]:
                logger.info(f"{sub_indent}{f}")
            if len(filenames) > 5:
                logger.info(f"{sub_indent}... and {len(filenames) - 5} more files")


def explore_metadata_files(metadata_dir: str):
    """Explore the pmap metadata files."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Metadata files in: {metadata_dir}")
    logger.info(f"{'='*60}")

    if not os.path.exists(metadata_dir):
        logger.warning(f"Metadata directory not found: {metadata_dir}")
        return

    for fname in sorted(os.listdir(metadata_dir)):
        fpath = os.path.join(metadata_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        logger.info(f"\n  File: {fname} ({size_mb:.1f} MB)")

        # Try to read as CSV or similar
        try:
            if fname.endswith(".csv"):
                df = pd.read_csv(fpath, nrows=5)
                logger.info(f"  Columns: {list(df.columns)}")
                logger.info(f"  Shape (sample): {df.shape}")
                logger.info(f"  First row:\n{df.iloc[0].to_dict()}")
            elif fname.endswith((".xlsx", ".xls")):
                df = pd.read_excel(fpath, nrows=5)
                logger.info(f"  Columns: {list(df.columns)}")
                logger.info(f"  Shape (sample): {df.shape}")
            elif fname.endswith(".json"):
                with open(fpath) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    logger.info(f"  JSON array with {len(data)} items")
                    if data:
                        logger.info(f"  First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
                elif isinstance(data, dict):
                    logger.info(f"  JSON object with keys: {list(data.keys())[:10]}")
            elif fname.endswith(".txt"):
                with open(fpath) as f:
                    lines = f.readlines()[:10]
                logger.info(f"  Text file, {len(lines)} lines (showing first 10)")
                for line in lines:
                    logger.info(f"    {line.rstrip()}")
        except Exception as e:
            logger.info(f"  Could not parse: {e}")


def sample_dicom_metadata(dicom_dir: str, n_samples: int = 5):
    """Read metadata from a few sample DICOM files."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Sample DICOM metadata")
    logger.info(f"{'='*60}")

    dicom_files = find_dicom_files(dicom_dir)
    if not dicom_files:
        logger.warning("No DICOM files found")
        return

    # Sample evenly across the file list
    indices = [int(i * len(dicom_files) / n_samples) for i in range(n_samples)]
    for idx in indices:
        fpath = dicom_files[idx]
        meta = get_dicom_metadata(fpath)
        logger.info(f"\n  File: {os.path.basename(fpath)}")
        for k, v in meta.items():
            logger.info(f"    {k}: {v}")


def main():
    # --- EDIT THESE PATHS ---
    ECHO_DIR = "/data/rsteven1/echo_jhu_dataset"
    METADATA_DIR = "/data/rsteven1/echo_jhu_metadata"
    # -------------------------

    logger.info("="*60)
    logger.info("JHU Echo Dataset Explorer")
    logger.info("="*60)

    # 1. Check what's available
    if os.path.exists(ECHO_DIR):
        explore_directory_structure(ECHO_DIR)
        sample_dicom_metadata(ECHO_DIR, n_samples=3)
    else:
        logger.warning(f"Echo directory not found: {ECHO_DIR}")
        logger.info("Data may still be uploading. Check back later.")

    # 2. Explore metadata
    if os.path.exists(METADATA_DIR):
        explore_metadata_files(METADATA_DIR)
    else:
        logger.warning(f"Metadata directory not found: {METADATA_DIR}")

    # 3. Build accession index (can take a while for large datasets)
    if os.path.exists(ECHO_DIR):
        logger.info("\nBuilding accession index (this may take a while)...")
        logger.info("Press Ctrl+C to skip if the dataset is too large right now.")
        try:
            index = build_accession_index(ECHO_DIR)
            # Save index for later use
            output_path = output_path = "/home/czhan182/scr4_rsteven1/chenjia_echo_project/2026 Multi-Modal Project/scripts/echo_infer_AD/results/accession_index.json"
            with open(output_path, "w") as f:
                json.dump(index, f, indent=2)
            logger.info(f"Accession index saved to {output_path}")
        except KeyboardInterrupt:
            logger.info("Skipped accession index building.")

    logger.info("\nDone! Review the output above to understand the data structure.")


if __name__ == "__main__":
    main()