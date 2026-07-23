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

echo "host        : $(hostname)"
echo "started     : $(date)"
echo "manifest    : pretrained_checkpoints/cxr_download_manifest.csv"

# 16 wget threads on 8 cores: downloads are network-bound, not CPU-bound.
"$PY" scripts/download_episode_cxr.py --workers 16

echo "finished    : $(date)"
df -h /scratch4 | tail -1
