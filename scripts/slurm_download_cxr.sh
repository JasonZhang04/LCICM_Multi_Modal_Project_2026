#!/bin/bash
#SBATCH --job-name=cxr_episode_dl
#SBATCH --partition=shared
#SBATCH --account=rsteven1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=logs/cxr_dl_%j.out
#SBATCH --error=logs/cxr_dl_%j.err
#
# Downloads the episode-level cohort's CXR images (~56k files, ~84 GB).
#
# Credentials must be exported before sbatch so they never land in the script
# or in the job's command line:
#     export PHYSIONET_USER=your_user
#     read -s PHYSIONET_PASS && export PHYSIONET_PASS
#     sbatch scripts/slurm_download_cxr.sh
#
# The job is resumable: files already on disk are skipped, so if it hits the
# time limit just resubmit it.

set -euo pipefail

PROJ="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
PY="/scratch4/rsteven1/your_env_name/bin/python3.10"

cd "$PROJ"
mkdir -p logs

# This cluster's Python SSL bundle is broken; wget/curl need the system one.
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

if [[ -z "${PHYSIONET_USER:-}" || -z "${PHYSIONET_PASS:-}" ]]; then
    echo "ERROR: export PHYSIONET_USER and PHYSIONET_PASS before sbatch" >&2
    exit 1
fi

# Tunable without editing the file:
#   DL_WORKERS  concurrent connections (default 12; PhysioNet throttles high
#               concurrency per IP, so raise gradually while watching for
#               ConnectTimeout failures)
#   DL_LIMIT    stop after N files — use for a bounded compute-node smoke test
WORKERS="${DL_WORKERS:-8}"
LIMIT_ARG=""
[[ -n "${DL_LIMIT:-}" ]] && LIMIT_ARG="--limit ${DL_LIMIT}"

echo "host        : $(hostname)"
echo "started     : $(date)"
echo "workers     : ${WORKERS}   limit: ${DL_LIMIT:-none}"
echo "manifest    : pretrained_checkpoints/cxr_download_manifest.csv"

# Network-bound work: threads sit on I/O, so worker count can exceed cores.
# Each worker keeps one persistent keep-alive connection to PhysioNet.
"$PY" scripts/download_episode_cxr.py --workers "$WORKERS" $LIMIT_ARG

echo "finished    : $(date)"
df -h /scratch4 | tail -1
