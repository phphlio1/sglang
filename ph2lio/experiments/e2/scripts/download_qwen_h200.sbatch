#!/usr/bin/env bash
# One-off: download the Qwen3.8-27B-FP8 checkpoint (~27 GB; permission
# granted 2026-10-09) as a flat local dir into the node-local
# /mnt/hf_hub_cache (team convention). Pinned to hgx-h200-01 — /mnt is
# node-local and the GLM weights already live on that node, so both e2
# arms run co-located. `hf download` resumes partial downloads, so
# re-submitting after a timeout is safe.
#
# The pre-existing models--Qwen--Qwen3.8-27B-FP8 cache entry is a 366 MB
# stub owned by another user (not writable by us) — hence our own flat dir.
#SBATCH --job-name=e2-qwen-dl
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/shared/loipp/logs/e2_qwen_dl_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
# Compute nodes carry the submitter's HOME=/home/loipp from the login
# node, but /home is login-local there and the node's own /home is not
# writable by us — hf (and torch, if anything imports it) cache under
# $HOME at startup and die with EACCES. Point HOME at /shared.
export HOME="${BENCH}/home"
mkdir -p "${HOME}"

HF_CLI="${BENCH}/venv-aiperf/bin/hf"
DEST="/mnt/hf_hub_cache/Qwen3.8-27B-FP8"

mkdir -p "${DEST}" /shared/loipp/logs
[[ -x "${HF_CLI}" ]] || { echo "ERROR: ${HF_CLI} not found (venv-aiperf)"; exit 2; }

echo "[$(date +%H:%M:%S)] download Qwen/Qwen3.8-27B-FP8 -> ${DEST}"
"${HF_CLI}" download Qwen/Qwen3.8-27B-FP8 --local-dir "${DEST}"
echo "[$(date +%H:%M:%S)] done: $(du -sh "${DEST}" | cut -f1)"
