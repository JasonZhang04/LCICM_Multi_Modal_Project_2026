#!/usr/bin/env python3
"""
Review raw echo reports for specific accessions.

Lets you examine the full report text, LLM evidence, and label
for any accession number. Useful for validating LLM labels.

Usage:
    # Review the 6 suspected false positives:
    python scripts/review_labels.py --accessions 30148056 26032601 29939091 23830331 24281156 29948846

    # Review all positives:
    python scripts/review_labels.py --positives

    # Interactive mode — type accession numbers one at a time:
    python scripts/review_labels.py --interactive
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths — adjust if needed
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LABELS_FILE = os.path.join(RESULTS_DIR, "echo_jhu_labels_llm.csv")


def load_full_reports(metadata_dir: str, accessions: set) -> dict:
    """
    Load the full narrative + impression text for specific accessions.
    Reads the big CSV in chunks but only keeps the ones we want.
    """
    procedures_file = os.path.join(metadata_dir, "dbo.derived_cardiology_echo_procedures.csv")

    use_cols = ["acc_num", "narrative", "impression"]
    reports = {}

    for chunk in pd.read_csv(
        procedures_file,
        usecols=use_cols,
        chunksize=50000,
        dtype=str,
        on_bad_lines="skip",
    ):
        chunk = chunk.dropna(subset=["acc_num"])
        matched = chunk[chunk["acc_num"].isin(accessions)]

        for _, row in matched.iterrows():
            acc = row["acc_num"]
            reports[acc] = {
                "narrative": row.get("narrative", ""),
                "impression": row.get("impression", ""),
            }

        # Stop early if we found all of them
        if len(reports) >= len(accessions):
            break

    return reports


def display_report(acc: str, labels_df: pd.DataFrame, full_reports: dict):
    """Display a single report with its label info."""
    separator = "=" * 80

    # Get label info
    row = labels_df[labels_df["accession"].astype(str) == str(acc)]
    if row.empty:
        print(f"\n{separator}")
        print(f"Accession {acc}: NOT FOUND in labels CSV")
        print(separator)
        return

    row = row.iloc[0]

    print(f"\n{separator}")
    print(f"ACCESSION: {acc}")
    print(separator)
    print(f"  Label:       {'DILATED (1)' if row['label'] == 1 else 'NORMAL (0)'}")
    print(f"  Confidence:  {row.get('confidence', 'N/A')}")
    print(f"  Severity:    {row.get('severity', 'N/A')}")
    print(f"  Diameter:    {row.get('diameter_cm', 'N/A')} cm")
    print(f"  # DICOMs:    {row.get('num_dicom_files', 'N/A')}")
    print(f"  DICOM path:  {row.get('dicom_path', 'N/A')}")

    print(f"\n--- LLM EVIDENCE ---")
    print(f"  {row.get('evidence', 'N/A')}")

    # Get full report
    report = full_reports.get(str(acc))
    if report:
        print(f"\n--- FULL NARRATIVE ---")
        narrative = report.get("narrative", "")
        if narrative and str(narrative) != "nan":
            # Clean up and print with wrapping
            narrative = narrative.replace("\\r\\n", "\n").replace("  ", " ")
            print(narrative[:3000])
            if len(narrative) > 3000:
                print("\n[... truncated, full report is longer ...]")
        else:
            print("  (empty)")

        print(f"\n--- IMPRESSION ---")
        impression = report.get("impression", "")
        if impression and str(impression) != "nan":
            impression = impression.replace("\\r\\n", "\n").replace("  ", " ")
            print(impression[:2000])
        else:
            print("  (empty)")
    else:
        print(f"\n  (full report not found in metadata for this accession)")

    print(separator)


def main():
    parser = argparse.ArgumentParser(description="Review echo reports and labels")
    parser.add_argument("--accessions", nargs="+", default=None,
                        help="Specific accession numbers to review")
    parser.add_argument("--positives", action="store_true",
                        help="Review all positive (dilated) cases")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode — enter accessions one at a time")
    parser.add_argument("--metadata-dir", default="/data/rsteven1/echo_jhu_metadata",
                        help="Path to metadata directory")
    args = parser.parse_args()

    # Load labels
    print("Loading labels...")
    labels_df = pd.read_csv(LABELS_FILE)
    labels_df["accession"] = labels_df["accession"].astype(str)

    # Determine which accessions to review
    if args.accessions:
        review_accessions = set(args.accessions)
    elif args.positives:
        review_accessions = set(labels_df[labels_df["label"] == 1]["accession"].tolist())
    elif args.interactive:
        review_accessions = None  # handled below
    else:
        print("Specify --accessions, --positives, or --interactive")
        sys.exit(1)

    if args.interactive:
        # Interactive mode: load reports on demand
        print("Interactive mode. Type accession numbers to review, 'q' to quit.")
        print(f"Total studies: {len(labels_df)} ({labels_df['label'].sum()} positive)\n")

        while True:
            acc = input("Enter accession (or 'q' to quit): ").strip()
            if acc.lower() == "q":
                break

            print("Loading full report...")
            reports = load_full_reports(args.metadata_dir, {acc})
            display_report(acc, labels_df, reports)
    else:
        # Batch mode: load all needed reports at once
        print(f"Loading full reports for {len(review_accessions)} accessions...")
        full_reports = load_full_reports(args.metadata_dir, review_accessions)
        print(f"Found {len(full_reports)} reports in metadata\n")

        for acc in sorted(review_accessions):
            display_report(acc, labels_df, full_reports)

        print(f"\nReviewed {len(review_accessions)} accessions.")


if __name__ == "__main__":
    main()