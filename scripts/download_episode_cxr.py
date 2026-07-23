"""
Manifest-driven download of MIMIC-CXR-JPG images for the episode-level cohort.

Unlike download_cohort_cxr.py (which derived its file list inline from the
DICOM-gated 1,927-patient cohort), this script downloads exactly the images
listed in a manifest CSV produced by build_episode_cohort.py. Keeping the
cohort definition and the download separate means the manifest is an auditable
artifact: the set of images backing the paper is a file, not a side effect.

Manifest columns: subject_id, study_id, dicom_id, jpg_rel
  jpg_rel looks like: files/p10/p10000032/s50414267/<dicom_id>.jpg

Credentials are read from, in order:
  1. --user + interactive password prompt (default, nothing hits the shell history)
  2. $PHYSIONET_USER / $PHYSIONET_PASS   (for non-interactive SLURM runs)

Usage:
    # interactive (login node, small test batch first)
    python scripts/download_episode_cxr.py --user <physionet_user> --limit 50

    # full run
    python scripts/download_episode_cxr.py --user <physionet_user> --workers 16

    # non-interactive / batch
    export PHYSIONET_USER=... PHYSIONET_PASS=...
    python scripts/download_episode_cxr.py --workers 16

Resume is automatic: files already present on disk are skipped, so re-running
after an interruption picks up where it left off.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

DEFAULT_MANIFEST = (
    "/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project/"
    "pretrained_checkpoints/cxr_download_manifest.csv"
)
DEST_ROOT = "/scratch4/rsteven1/MIMIC_CXR_JPG_cohort/files"
BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("cxr_episode_download.log", mode="a")],
)
log = logging.getLogger(__name__)


def build_tasks(manifest_path: str, limit: int | None) -> list[tuple[str, str]]:
    """Return (url, local_path) for every manifest row not already on disk."""
    man = pd.read_csv(manifest_path)
    log.info("Manifest: %d rows from %s", len(man), manifest_path)

    tasks, skipped = [], 0
    for rel in man["jpg_rel"]:
        rel_no_prefix = rel[len("files/"):] if rel.startswith("files/") else rel
        local_path = os.path.join(DEST_ROOT, rel_no_prefix)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            skipped += 1
            continue
        tasks.append((f"{BASE_URL}/{rel_no_prefix}", local_path))
        if limit and len(tasks) >= limit:
            break

    log.info("To download: %d  (already present: %d)", len(tasks), skipped)
    return tasks


def download_one(args: tuple) -> tuple[str, bool, str]:
    """wget handles PhysioNet's 401-challenge -> 200 auth flow correctly."""
    url, local_path, username, password = args
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        r = subprocess.run(
            ["wget", "-q", "--user", username, "--password", password,
             "-O", local_path, "--tries=3", "--timeout=60", url],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return url, True, "ok"
        if os.path.exists(local_path):
            os.remove(local_path)      # don't leave truncated files to fool the resume check
        err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown error"
        return url, False, err
    except Exception as e:  # noqa: BLE001
        return url, False, str(e)


def main() -> None:
    import getpass

    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--user", default=os.environ.get("PHYSIONET_USER"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Only fetch the first N missing files (use for a smoke test)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tasks = build_tasks(args.manifest, args.limit)
    if not tasks:
        log.info("Nothing to download — all manifest files already present.")
        return

    # Dry run is a manifest/URL check, so it deliberately runs before auth.
    if args.dry_run:
        for url, _ in tasks[:20]:
            print(url)
        print(f"... ({len(tasks)} total)")
        return

    if not args.user:
        p.error("PhysioNet username required via --user or $PHYSIONET_USER")
    password = os.environ.get("PHYSIONET_PASS") or getpass.getpass(
        f"PhysioNet password for {args.user}: ")

    # Verify credentials on one real file before spawning workers, so a bad
    # password fails in 2 seconds instead of 55,000 times in parallel.
    probe_url, probe_dest = tasks[0][0], "/tmp/physionet_auth_probe.jpg"
    _, ok, msg = download_one((probe_url, probe_dest, args.user, password))
    if os.path.exists(probe_dest):
        os.remove(probe_dest)
    if not ok:
        log.error("Credential check failed: %s", msg)
        sys.exit(1)
    log.info("Authentication OK — starting %d files on %d workers", len(tasks), args.workers)

    start, n_done, n_ok, n_fail = time.time(), 0, 0, 0
    failures = []
    report_every = max(1, len(tasks) // 40)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(download_one, (u, pth, args.user, password)) for u, pth in tasks]
        for fut in as_completed(futs):
            url, ok, msg = fut.result()
            n_done += 1
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                if len(failures) < 200:
                    failures.append(f"{url}\t{msg}")
            if n_done % report_every == 0:
                el = time.time() - start
                rate = n_done / el if el else 0
                eta = (len(tasks) - n_done) / rate / 60 if rate else 0
                log.info("%6d/%6d  ok=%d fail=%d  %.1f f/s  ETA %.0f min",
                         n_done, len(tasks), n_ok, n_fail, rate, eta)

    log.info("DONE in %.1f min — ok=%d fail=%d", (time.time() - start) / 60, n_ok, n_fail)
    if failures:
        with open("cxr_episode_download_failures.tsv", "w") as fh:
            fh.write("\n".join(failures) + "\n")
        log.warning("Wrote %d failures to cxr_episode_download_failures.tsv "
                    "(re-run the script to retry them)", len(failures))


if __name__ == "__main__":
    main()
