#!/usr/bin/env bash
# E2 smoke — GLM-5.3 arm: one 5-min dfs-weight run over 16 dataset entries.
# Validates the production-mapped config boots on 8 ranks (deepep, dsa,
# EAGLE, hicache) and the harness wiring (health, eval-log 8-writer,
# /metrics) before the 22 h sweep. Submit when an hgx node is free; after
# it starts, confirm the node matches the one holding
# /mnt/hf_hub_cache/GLM-5.3-W4AFP8 (sacct) — /mnt is node-local.
#SBATCH --job-name=e2-glm53-smoke
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# The server pins GPUs 4-7 to NUMA node 1 (--numa-node 0 0 0 0 1 1 1 1).
# Each NUMA node has 96 CPUs (2x48c/96t), so any request <= 96 lands on one
# node only and the node-1 binding is skipped at boot (2623, with 64). 160
# > 96 forces Slurm to span both nodes (measured demand is only ~16-20
# cores total).
#SBATCH --cpus-per-task=160
# GLM's 8 ranks allocate ~1.23 TB of host hicache/indexer pools (8 x 128 GB
# KV + 8 x 25.8 GB DSA); measured peak RSS 146.5 GB/rank (sacct 2992) ~= 1.17
# TB across 8 ranks, so the default per-CPU grant is not enough. 1600G covers
# it and leaves ~430 GB for teammates (CR_Core_Memory makes --mem=0 exclude
# every other job from the node). The uvicorn worker deaths were NOT cgroup
# OOM (disproven by smoke 2794) — root cause was the uvicorn worker
# healthcheck SIGKILL, fixed and verified by smoke 2925.
#SBATCH --mem=1600G
#SBATCH --time=04:30:00
#SBATCH --output=/shared/loipp/logs/e2_glm53_smoke_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"

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

# --- GLM worker-death probe (uvicorn workers lose stderr at spawn) ---------
# Every python process re-opens fds 1/2 onto PROBE_WORKER_LOG via the
# sitecustomize hook, so worker tracebacks survive the multiprocessing
# spawn fd-close that made earlier deaths invisible (SIGKILL leaves no
# traceback at all — the 600 s healthcheck fix in
# start_server_h200_glm53.sh is still unverified). Health polling is HTTP
# (run_e2.sh server_healthy -> curl /health), so redirecting the server log
# breaks nothing; side effect: server_*.log stays empty — ALL server output
# lands in the probe log on the node instead. Truncated here so stale lines
# from earlier jobs can never pollute the death count. Disable at submit
# with PROBE_SITE= (empty), e.g. once the worker fix is verified.
PROBE_SITE="${PROBE_SITE:-/shared/loipp/probe_site}"
if [ -n "${PROBE_SITE}" ]; then
  export PYTHONPATH="${PROBE_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
  export PROBE_WORKER_LOG="${PROBE_WORKER_LOG:-/tmp/glm_workers_probe.log}"
  : > "${PROBE_WORKER_LOG}"
  echo "probe on: all python output -> ${PROBE_WORKER_LOG}"
fi

# Same knobs as run_e2_h200_glm53.sbatch (smoke overrides duration/entries
# itself via SMOKE_DURATION / NUM_ENTRIES defaults in run_e2.sh).
export SGLANG_URL="http://127.0.0.1:30000"
export MODEL="zai-org/GLM-5.3"
export TOKENIZER="/mnt/hf_hub_cache/GLM-5.3-W4AFP8"
export OUTPUT_DIR="${E2_DIR}/results/glm53/agentx"
export SERVER_SCRIPT="${E2_DIR}/scripts/start_server_h200_glm53.sh"
# Cold boot on the NFS venv measured > 60 min (2623: ~50 min of imports
# + arg resolution before schedulers spawned, still loading at the 3600 s
# cutoff). First health window must cover it; later boots reuse the
# /shared HOME compile caches and are much faster.
# GLM boot measured > 2 h: DeepGEMM warmup + prefill CUDA-graph JIT took to
# 05:16 (smoke 2758), decode graphs + hicache still pending at the 7200 s
# cutoff (06:18) — no crash, purely slow. 12000 s covers the observed ~2.5-3 h
# path with margin; the /shared HOME JIT caches cut later boots.
export HEALTH_TIMEOUT=12000
export FORCE_RESTART=1
export GRACE_PERIOD=360
export SEED=42
export BOOST_ADAPTIVE_GAMMA=1
# Import-cache pre-read DISABLED: the smoke is a single boot, where the
# ~2.5h cold pass costs more than the ~1h40m of boot it saves (net loss,
# and under the 4h30 wall it risks killing the bench mid-run). Boot cold
# instead (~2h40m) — see run_e2_h200_glm53.sbatch for the full arithmetic.
# Overrideable: submit with WARM_CACHE_PATHS=<paths> (via --export) to re-enable.
export WARM_CACHE_PATHS="${WARM_CACHE_PATHS:-}"
export HF_HOME="${BENCH}/hf"
export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache"

cd "${E2_DIR}"
bash scripts/run_e2.sh smoke
