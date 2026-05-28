"""
Selective parallel download of MIMIC-CXR-JPG images for the 1,927-patient cohort.

Only downloads images for patients in the echo ∩ CXR ∩ ECG intersection,
instead of the full 570 GB dataset. Total download: ~2.6 GB (17,946 files).

Usage:
    python download_cohort_cxr.py --user chenjiazhang --password YOUR_PASSWORD

    # Dry run (prints URLs without downloading):
    python download_cohort_cxr.py --user chenjiazhang --password YOUR_PASSWORD --dry-run

    # Adjust parallel workers (default 8):
    python download_cohort_cxr.py --user chenjiazhang --password YOUR_PASSWORD --workers 16
"""

import argparse
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import subprocess
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cxr_download.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
LABELS_CSV    = "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/data/echo/aortic_labels.csv"
CXR_INDEX_CSV = "/scratch4/rsteven1/MIMIC_CXR_GS/cxr-record-list.csv"
ECG_INDEX_CSV = "/scratch4/rsteven1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/record_list.csv"
DEST_ROOT     = "/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/files"
BASE_URL      = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files"


def build_file_list() -> list[tuple[str, str]]:
    """
    Returns list of (url, local_path) tuples for all cohort CXR images.
    Skips files that already exist locally (resume support).
    """
    labels = pd.read_csv(LABELS_CSV, usecols=["subject_id"])
    cxr    = pd.read_csv(CXR_INDEX_CSV)
    ecg    = pd.read_csv(ECG_INDEX_CSV, usecols=["subject_id"])

    cohort_ids = set(labels["subject_id"]) & set(cxr["subject_id"]) & set(ecg["subject_id"])
    cohort_cxr = cxr[cxr["subject_id"].isin(cohort_ids)].copy()

    log.info("Cohort: %d patients, %d CXR records", len(cohort_ids), len(cohort_cxr))

    tasks = []
    skipped = 0
    for row in cohort_cxr.itertuples(index=False):
        # Convert DICOM path → JPG path
        # path column looks like: files/p10/p10000032/s50414267/<dicom_id>.dcm
        jpg_rel = row.path.replace(".dcm", ".jpg")   # e.g. files/p10/.../xxx.jpg
        # Strip leading "files/" since BASE_URL already contains "files"
        rel_no_prefix = jpg_rel[len("files/"):]       # p10/p10000032/s50414267/xxx.jpg

        url        = f"{BASE_URL}/{rel_no_prefix}"
        local_path = os.path.join(DEST_ROOT, rel_no_prefix)

        if os.path.exists(local_path):
            skipped += 1
            continue
        tasks.append((url, local_path))

    log.info("Files to download: %d  (already present: %d)", len(tasks), skipped)
    return tasks


def download_one(args: tuple) -> tuple[str, bool, str]:
    """
    Download a single file using wget, which correctly handles PhysioNet's
    401-challenge → 200-response auth flow. Returns (url, success, message).
    """
    url, local_path, username, password = args
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        result = subprocess.run(
            [
                "wget", "-q",
                "--user", username,
                "--password", password,
                "-O", local_path,
                "--tries=3",
                "--timeout=60",
                url,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return url, True, "ok"
        else:
            # Clean up empty/partial file
            if os.path.exists(local_path):
                os.remove(local_path)
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            return url, False, err
    except Exception as e:
        return url, False, str(e)


def main():
    import getpass
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",     required=True,  help="PhysioNet username")
    parser.add_argument("--workers",  type=int, default=8, help="Parallel download threads")
    parser.add_argument("--dry-run",  action="store_true", help="Print URLs without downloading")
    args = parser.parse_args()
    # Prompt for password securely (not visible on screen or in shell history)
    args.password = getpass.getpass(f"PhysioNet password for {args.user}: ")

    tasks = build_file_list()
    if not tasks:
        log.info("All files already present. Nothing to download.")
        return

    if args.dry_run:
        for url, path in tasks[:20]:
            print(url)
        print(f"... ({len(tasks)} total URLs)")
        return

    # Test credentials with a single wget call before launching parallel workers
    test_url = f"{BASE_URL}/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
    test_path = "/tmp/physionet_auth_test.jpg"
    test_ok, _, test_msg = download_one((test_url, test_path, args.user, args.password))
    if os.path.exists(test_path):
        os.remove(test_path)
    if not test_ok:
        log.error("Credential test failed: %s — check --user and password", test_msg)
        sys.exit(1)
    log.info("Authentication OK")

    log.info("Starting download: %d files with %d parallel workers", len(tasks), args.workers)
    start = time.time()

    download_args = [(url, path, args.user, args.password) for url, path in tasks]

    n_done = n_ok = n_fail = 0
    report_every = max(1, len(tasks) // 20)   # log progress every 5%

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, a): a for a in download_args}
        for fut in as_completed(futures):
            url, ok, msg = fut.result()
            n_done += 1
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                log.warning("FAILED: %s — %s", url, msg)
                if "401" in msg:
                    log.error("Authentication error — aborting")
                    pool.shutdown(wait=False, cancel_futures=True)
                    sys.exit(1)

            if n_done % report_every == 0 or n_done == len(tasks):
                elapsed = time.time() - start
                rate = n_done / elapsed
                eta  = (len(tasks) - n_done) / rate if rate > 0 else 0
                log.info(
                    "Progress: %d/%d  (%.0f%%)  ok=%d fail=%d  "
                    "rate=%.1f files/s  ETA=%.0f min",
                    n_done, len(tasks), 100 * n_done / len(tasks),
                    n_ok, n_fail, rate, eta / 60,
                )

    elapsed = time.time() - start
    log.info(
        "Done: %d ok, %d failed in %.1f min",
        n_ok, n_fail, elapsed / 60,
    )
    if n_fail > 0:
        log.warning("%d files failed — re-run the script to retry (skips existing files)", n_fail)


if __name__ == "__main__":
    main()
