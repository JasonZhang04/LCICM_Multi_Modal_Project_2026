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

from __future__ import annotations   # allow 3.10 union hints under older interpreters

import argparse
import base64
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from urllib3.util.retry import Retry

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
# Retries are handled and reported via the failures file; silence urllib3's
# per-retry WARNING spam so transient connect timeouts don't drown the log.
logging.getLogger("urllib3").setLevel(logging.ERROR)


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


# The previous implementation shelled out to wget once per file: a fresh
# process, TCP connect, TLS handshake, and PhysioNet's 401-challenge round trip
# on every single image. For 1.5 MB files on a fast network that overhead
# dominated (~5 s/file), so throughput was ~0.1 files/s regardless of bandwidth.
#
# This version keeps one persistent requests.Session per worker thread. Two wins:
#   - HTTP keep-alive reuses one TCP+TLS connection across all of that thread's
#     files, so the handshake is paid once per worker, not once per file.
#   - The Authorization header is sent preemptively, skipping the 401 challenge
#     entirely (no wasted round trip to be told "auth required").
# The remaining per-file cost is essentially just the transfer, so more workers
# scale nearly linearly until PhysioNet or the NIC pushes back.

_AUTH_HEADER = ""                     # set once in main(), read by every worker
_thread_local = threading.local()
CHUNK = 1 << 16


def _session() -> requests.Session:
    """One keep-alive Session per thread, created lazily on first use."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["Authorization"] = _AUTH_HEADER
        # PhysioNet's WAF returns 403 to the default python-requests User-Agent
        # (verified: default UA -> 403, Wget UA -> 401). Present as wget so the
        # request reaches the auth layer at all.
        s.headers["User-Agent"] = "Wget/1.20.3"
        retry = Retry(total=3, backoff_factor=0.5,
                      status_forcelist=(500, 502, 503, 504),
                      allowed_methods=frozenset({"GET"}))
        s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))
        _thread_local.session = s
    return s


def download_one(args: tuple) -> tuple[str, bool, str]:
    """Stream one file to disk over this thread's persistent session."""
    url, local_path = args
    tmp_path = local_path + ".part"     # write to .part so an interrupted file never looks complete
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        with _session().get(url, stream=True, timeout=(10, 120)) as r:
            if r.status_code != 200:
                return url, False, f"HTTP {r.status_code}"
            with open(tmp_path, "wb") as fh:
                for chunk in r.iter_content(CHUNK):
                    fh.write(chunk)
        if os.path.getsize(tmp_path) == 0:
            os.remove(tmp_path)
            return url, False, "empty response"
        os.replace(tmp_path, local_path)   # atomic: the resume check only ever sees complete files
        return url, True, "ok"
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
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
    p.add_argument("--count-remaining", action="store_true",
                   help="Print how many manifest files are not yet on disk, then exit "
                        "(used by the slurm chain to decide whether to resubmit).")
    args = p.parse_args()

    if args.count_remaining:
        logging.disable(logging.CRITICAL)   # keep stdout to just the number
        print(len(build_tasks(args.manifest, None)))
        return

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

    global _AUTH_HEADER
    token = base64.b64encode(f"{args.user}:{password}".encode()).decode()
    _AUTH_HEADER = f"Basic {token}"

    # Verify credentials on one real file before spawning workers, so a bad
    # password fails immediately instead of 55,000 times in parallel.
    probe_url, probe_dest = tasks[0][0], "/tmp/physionet_auth_probe.jpg"
    _, ok, msg = download_one((probe_url, probe_dest))
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
        futs = [pool.submit(download_one, (u, pth)) for u, pth in tasks]
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
