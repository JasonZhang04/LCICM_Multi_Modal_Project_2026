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
# Self-resubmitting download of the episode-cohort CXR images (~56k files, ~84 GB).
# PhysioNet's HTTPS endpoint throttles to ~145 KB/s from our single shared IP, so
# the full set is ~7 days — well past the 24h wall. This script downloads until
# ~1h before the wall, then automatically resubmits itself (resumably) until every
# manifest file is on disk. Guards prevent a runaway: it stops on auth failure, on
# completion, or after DL_MAXITER iterations.
#
# Launch it ONCE from your shell (credentials are read from your environment and
# forwarded to each link in the chain):
#     export PHYSIONET_USER=your_user
#     read -s -p "PhysioNet password: " PHYSIONET_PASS && export PHYSIONET_PASS && echo
#     sbatch scripts/slurm_download_cxr.sh
#
# Tunables (env):
#     DL_WORKERS   concurrent connections (default 8; PhysioNet fails above ~8-10)
#     DL_LIMIT     stop after N files, NO self-resubmit (bounded smoke test)
#     DL_MAXITER   safety cap on chain length (default 20; ~7 days needs ~7)

set -euo pipefail

PROJ="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
PY="/scratch4/rsteven1/your_env_name/bin/python3.10"
cd "$PROJ"
mkdir -p logs

# This cluster's Python SSL bundle is broken; the requests session needs the system one.
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

if [[ -z "${PHYSIONET_USER:-}" || -z "${PHYSIONET_PASS:-}" ]]; then
    echo "ERROR: export PHYSIONET_USER and PHYSIONET_PASS before sbatch" >&2
    exit 1
fi

WORKERS="${DL_WORKERS:-8}"
ITER="${DL_ITER:-1}"
MAXITER="${DL_MAXITER:-20}"
LIMIT_ARG=""
[[ -n "${DL_LIMIT:-}" ]] && LIMIT_ARG="--limit ${DL_LIMIT}"

echo "===== chain iter ${ITER}/${MAXITER}  host $(hostname)  $(date) ====="
echo "workers=${WORKERS}  limit=${DL_LIMIT:-none}"

# Download until ~1h before the 24h wall, then hand off to a fresh job. `timeout`
# returns 124 when it stops the run at the wall; the download script exits 1 only
# on a credential failure and 0 when the batch is exhausted.
set +e
timeout 23h "$PY" scripts/download_episode_cxr.py --workers "$WORKERS" $LIMIT_ARG
RC=$?
set -e
echo "download step returned ${RC}  $(date)"

# A bounded smoke test never chains.
if [[ -n "${DL_LIMIT:-}" ]]; then
    echo "DL_LIMIT set — bounded test, not resubmitting."
    exit 0
fi

# Credential failure: stop the chain rather than loop forever on a bad password.
if [[ ${RC} -eq 1 ]]; then
    echo "download exited 1 (credential/auth failure) — chain stopped." >&2
    exit 1
fi

REMAIN="$("$PY" scripts/download_episode_cxr.py --count-remaining)"
echo "files still missing: ${REMAIN}"

if [[ "${REMAIN}" -gt 0 && "${ITER}" -lt "${MAXITER}" ]]; then
    NEXT=$((ITER + 1))
    echo "resubmitting chain link ${NEXT} ..."
    # --export=ALL forwards this job's environment (incl. PHYSIONET_PASS) to the next.
    sbatch --export=ALL,DL_ITER=${NEXT} scripts/slurm_download_cxr.sh
elif [[ "${REMAIN}" -eq 0 ]]; then
    echo "ALL FILES DOWNLOADED — chain complete. $(date)"
    df -h /scratch4 | tail -1
else
    echo "hit DL_MAXITER=${MAXITER} with ${REMAIN} files left — resubmit manually to continue." >&2
fi
