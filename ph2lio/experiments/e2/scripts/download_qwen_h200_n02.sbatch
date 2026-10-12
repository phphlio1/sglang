#!/usr/bin/env bash
# One-off: seed the Qwen3.8-27B-FP8 checkpoint (~27 GB; download permission
# granted 2026-10-09) onto hgx-h200-02's node-local /mnt so the Qwen arm can
# run there in parallel with the GLM arm on hgx-h200-01 (both arms queued
# independently instead of 38 h sequential). Same flat-dir convention as the
# -01 copy from job 2624. `hf download` resumes partial downloads, so
# re-submitting after a timeout is safe.
#SBATCH --job-name=e2-qwen-dl-n02
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/shared/loipp/logs/e2_qwen_dl_n02_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"

# Compute nodes carry the submitter's HOME=/home/loipp, but /home is
# login-local — the node's own /home is unwritable by us and hf caches
# under $HOME at startup. Point HOME at /shared.
export HOME="${BENCH}/home"
mkdir -p "${HOME}"

HF_CLI="${BENCH}/venv-aiperf/bin/hf"
DEST="/mnt/hf_hub_cache/Qwen3.8-27B-FP8"

mkdir -p "${DEST}" /shared/loipp/logs
[[ -x "${HF_CLI}" ]] || { echo "ERROR: ${HF_CLI} not found (venv-aiperf)"; exit 2; }

echo "[$(date +%H:%M:%S)] download Qwen/Qwen3.8-27B-FP8 -> ${DEST} (hgx-h200-02)"
"${HF_CLI}" download Qwen/Qwen3.8-27B-FP8 --local-dir "${DEST}"
echo "[$(date +%H:%M:%S)] done: $(du -sh "${DEST}" | cut -f1)"
