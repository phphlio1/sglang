#!/usr/bin/env bash
# E2 8x H200 sweep — Qwen3.8-27B-FP8, RESUME variant.
# Identical to run_e2_h200_qwen.sbatch except the cell loop: a cell is skipped
# iff one of its run dirs already holds e2_metrics.json, so an scancelled sweep
# continues from the first unfinished cell on resubmit. Only submit this AFTER
# run_e2_h200_qwen.sbatch (no resume) was interrupted — never run both at once.
#SBATCH --job-name=e2-qwen-h200-r
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-02
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# All 192 CPUs so the 8 replicas are not squeezed onto NUMA node 0 (the
# node grants 64-core jobs node-0-only affinity).
#SBATCH --cpus-per-task=192
# Jobs on this account must not exceed 4 h. One cell = boot (~2 h with the
# per-rank load-weight stagger) + warmup + 30 min bench + grace ~= 3.2 h, so
# each submission completes ~1 cell; the wall kills the next cell's boot
# mid-flight and the resume loop below re-runs it on resubmit — completed
# cells (e2_metrics.json present) are never lost.
#SBATCH --time=04:00:00
#SBATCH --output=/shared/loipp/logs/e2_h200_qwen_resume_%j.log

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

export SGLANG_URL="http://127.0.0.1:30000"
export MODEL="Qwen/Qwen3.8-27B-FP8"
# TOKENIZER points at the local checkpoint dir so the aiperf client never
# touches the network.
export TOKENIZER="/mnt/hf_hub_cache/Qwen3.8-27B-FP8"
export OUTPUT_DIR="${E2_DIR}/results/8x-h200/qwen/agentx"
export SERVER_SCRIPT="${E2_DIR}/scripts/start_server_h200_qwen.sh"
# Per-cell reboot: 8 replicas each import the NFS venv and (first time)
# JIT+graph-capture ~9 min serially (2751); /shared HOME caches cut later
# boots. Keep the window generous — timeout only fails if actually exceeded.
export HEALTH_TIMEOUT=9000
# Campaign regime (mentor, 2026-10-08): fresh boot per cell (cold radix
# cache); client drain capped at 360 s (user-set; requests in flight
# longer than that at close are cut from metrics).
export FORCE_RESTART=1
export GRACE_PERIOD=360
export SEED=42
export DURATION=1800
# Boost arm on this hardware: adaptive gamma (the RTX static 0.09 was
# tuned for the 1x-RTX regime; gamma-Ada self-tunes from the admission
# wait tail, gamma0 = clock default). Veto back to the RTX arm:
# BOOST_ADAPTIVE_GAMMA=0 + BOOST_GAMMA=0.09.
export BOOST_ADAPTIVE_GAMMA=1

# Import-cache pre-read DISABLED on this arm: the warm-vs-cold experiment
# (2026-10-11) showed the Qwen boot path is unaffected (load-weight starts
# stay staggered ~8.2-9.2 min/rank and DP0 still ~31 min from args to load
# weight, identical warm vs cold — the serializer is not NFS data reads),
# while a cold pass costs ~2.2 h on this node — pure loss under the 4 h cap.
# The GLM arm keeps its warm pass (it does help there; single tp8 process).
# Overrideable: submit with WARM_CACHE_PATHS=<paths> (via --export) to re-enable.
export WARM_CACHE_PATHS="${WARM_CACHE_PATHS:-}"
export HF_HOME="${BENCH}/hf"
# Separate mmap cache from the -01 jobs.
export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache_n02"

cd "${E2_DIR}"

# Resume-capable sweep (see run_e2_h200_glm53.sbatch for the rationale; the
# loop lives here because run_e2.sh must not be edited while jobs run it).
SWEEP_CONCS=(32 64 128)
for p in fcfs-match lpm hrrn dfs-weight boost; do
  for c in "${SWEEP_CONCS[@]}"; do
    # Mirrors run_label in run_e2.sh; BOOST_ADAPTIVE_GAMMA=1 is pinned above.
    label="c${c}"
    [[ "${p}" == "boost" ]] && label="c${c}_gada_${BOOST_CLOCK:-wall}_k${BOOST_BIN_SIZE:-256}"
    if ls "${OUTPUT_DIR}/${p}/${label}_"*/e2_metrics.json >/dev/null 2>&1; then
      echo "=== E2 resume: skip completed ${p} c${c} ==="
      continue
    fi
    echo "=== E2 sweep point: policy=${p} concurrency=${c} ==="
    bash scripts/run_e2.sh policy "${p}" "${c}" \
      || echo "=== E2 cell failed: ${p} c${c} (will retry on resubmit) ==="
  done
done
