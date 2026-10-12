#!/usr/bin/env bash
# E2 smoke — Qwen3.8-27B-FP8 arm (hgx-h200-02, parallel with the GLM arm on
# -01): one 5-min dfs-weight run over 16 dataset entries. Validates the RTX
# config boots as 8 dp-size replicas (EAGLE, replayssm-spec, hicache) and the
# harness wiring (health, eval-log, /metrics) before the 16 h sweep.
# Requires the node-local weights from download_qwen_h200_n02.sbatch.
#SBATCH --job-name=e2-qwen-smoke
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-02
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# All 192 CPUs so the 8 replicas are not squeezed onto NUMA node 0 (the
# node grants 64-core jobs node-0-only affinity).
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --output=/shared/loipp/logs/e2_qwen_smoke_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"

# Compute nodes inherit HOME=/home/loipp from the submitter, but /home is
# login-local — on the node that path doesn't exist and its parent is
# root-owned, so torch dynamo/inductor's cache bootstrap (makedirs under
# $HOME at import time) dies with EACCES before the server ever boots.
# --- node-local execution mirror (built by mirror_from_login.sh + ----------
# --- mirror_node_side.sh, run from the login node) ----------------------------
# venvs, the uv base interpreter, the sglang import tree, and the HOME
# compile caches live on this node's /mnt after the mirror job: imports and
# JIT-cache verification stop touching the ~1.4 MB/s NFS and survive
# teammates' page-cache eviction. Falls back to the /shared paths (slow)
# whenever the mirror is absent.
NODE_LOCAL="${NODE_LOCAL:-/mnt/loipp}"
if [ -x "${NODE_LOCAL}/venv-server/bin/python3" ]; then
  export SERVER_PYTHON="${NODE_LOCAL}/venv-server/bin/python3"
  export AIPERF="${NODE_LOCAL}/venv-aiperf/bin/aiperf"
  export CUDA_HOME="${NODE_LOCAL}/venv-server/lib/python3.11/site-packages/nvidia/cu13"
  export HOME="${NODE_LOCAL}/home"
else
  echo "=== node-local mirror absent on $(hostname); running from /shared (slow NFS) ==="
  export HOME="${BENCH}/home"
fi
mkdir -p "${HOME}"

# Same knobs as run_e2_h200_qwen.sbatch (smoke overrides duration/entries
# itself via SMOKE_DURATION / NUM_ENTRIES defaults in run_e2.sh).
export SGLANG_URL="http://127.0.0.1:30000"
export MODEL="Qwen/Qwen3.8-27B-FP8"
export TOKENIZER="/mnt/hf_hub_cache/Qwen3.8-27B-FP8"
export OUTPUT_DIR="${E2_DIR}/results/8x-h200/qwen/agentx"
export SERVER_SCRIPT="${E2_DIR}/scripts/start_server_h200_qwen.sh"
# 8 replicas import the NFS venv independently at boot, and the first boot
# also JIT-compiles the flashinfer GDN kernels per replica (2751: ranks
# DP2-DP4 took ~9.3 min EACH, serially; 8 ranks ~= 150 min to ready vs the
# 5400s timeout that killed it). GDN JIT is cached under /shared HOME after
# 2751, so later boots are faster, but keep the window wide regardless.
export HEALTH_TIMEOUT=9000
export FORCE_RESTART=1
export GRACE_PERIOD=360
export SEED=42
export BOOST_ADAPTIVE_GAMMA=1
export HF_HOME="${BENCH}/hf"
# Separate mmap cache from the -01 jobs: both arms' clients may build it
# from the dataset concurrently on their first run.
export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache_n02"

cd "${E2_DIR}"
bash scripts/run_e2.sh smoke
