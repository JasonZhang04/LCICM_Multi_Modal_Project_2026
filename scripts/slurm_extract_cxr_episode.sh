#!/bin/bash -l
#SBATCH --job-name=extract_cxr_ep
#SBATCH --partition=a100
#SBATCH --account=rsteven1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract_cxr_ep_%j.out
#SBATCH --error=logs/extract_cxr_ep_%j.err
#
# CXR feature extraction for the EPISODE cohort (~60k frontal images).
#   sbatch scripts/slurm_extract_cxr_episode.sh patchpool   # RAD-DINO cls/aortapool/heartpool
#   sbatch scripts/slurm_extract_cxr_episode.sh geometry    # 17 geometry features
#
# Both are resumable (periodic atomic checkpoints, skip-done on restart), so this
# runs the script until ~30min before the wall, then resubmits itself if it timed
# out (max EXTRACT_MAXITER links). roi_venv = base env (torch/RAD-DINO) + torchxrayvision.

set -euo pipefail

TARGET="${1:?usage: sbatch slurm_extract_cxr_episode.sh patchpool|geometry}"
case "$TARGET" in
    patchpool) SCRIPT=scripts/extract_raddino_patchpool.py ;;
    geometry)  SCRIPT=scripts/extract_cxr_geometry_features.py ;;
    *) echo "unknown target '$TARGET' (want patchpool|geometry)" >&2; exit 2 ;;
esac

VENV=/scratch4/rsteven1/chenjia_echo_project/roi_venv
PROJ="/scratch4/rsteven1/chenjia_echo_project/2026 Multi-Modal Project"
ITER="${EXTRACT_ITER:-1}"
MAXITER="${EXTRACT_MAXITER:-4}"

cd "$PROJ"
mkdir -p logs
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HOME=/scratch4/rsteven1/chenjia_echo_project/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CXR_INSTANCES=cxr_instances_episode.csv
# Write to episode-specific outputs so the v7 (522-cohort) artifacts stay intact.
export PATCHPOOL_OUT=raddino_patchpool_embeddings_episode.pt
export GEOMETRY_OUT=cxr_geometry_features_episode.csv
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

echo "===== extract ${TARGET} iter ${ITER}/${MAXITER}  node ${SLURMD_NODENAME}  $(date) ====="
echo "gpu: ${CUDA_VISIBLE_DEVICES:-none}  instances: ${CXR_INSTANCES}  script: ${SCRIPT}"

set +e
timeout 11.5h "$VENV/bin/python" -u "$SCRIPT"
RC=$?
set -e
echo "extract step returned ${RC}  $(date)"

# 124 = hit the internal timeout with work left -> resume in a fresh job.
if [[ ${RC} -eq 124 && ${ITER} -lt ${MAXITER} ]]; then
    NEXT=$((ITER + 1))
    echo "timed out — resubmitting ${TARGET} link ${NEXT} (resumes from checkpoint)"
    sbatch --export=ALL,EXTRACT_ITER=${NEXT} scripts/slurm_extract_cxr_episode.sh "${TARGET}"
elif [[ ${RC} -eq 0 ]]; then
    echo "${TARGET} extraction COMPLETE."
else
    echo "extract returned ${RC} (not a clean finish) — check logs before rerunning." >&2
fi
