#!/usr/bin/env bash
# E2 boost campaign — Phase 4: production verification on the dynamo arm.
# Prod-exact c64 cells: dfs-weight (current prod policy) vs boost at the
# phase-1 sweet-spot gamma vs boost at gamma=15. DURATION=1800, seed 42.
# These are the focus cells: raw server_metrics_export.jsonl stays UNCOMPRESSED
# (no rolling gzip here) for extract_metrics_timeseries.py + focus reports.
#SBATCH --job-name=e2boost-p4-prodverify
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# Must stay >96: the server binds GPUs 4-7 to NUMA node 1 (--numa-node 0 0 0 0
# 1 1 1 1); <=96 CPUs lands the whole allocation on one NUMA node and the
# binding is skipped at boot (smoke 2623 died at 64). Measured demand is
# ~16-20 cores — the count is purely topological. 128 since 2026-10-19.
#SBATCH --cpus-per-task=128
#SBATCH --mem=1600G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/loipp/logs/e2boost_p4_prodverify_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"
BOOST_SWEEP_GAMMA="${BOOST_SWEEP_GAMMA:-5.0}"   # phase-1 winner

NODE_LOCAL="${NODE_LOCAL:-/mnt/loipp}"
if [ -x "${NODE_LOCAL}/venv-dynamo/bin/python3" ]; then
  export AIPERF="${NODE_LOCAL}/venv-aiperf/bin/aiperf"
  export HOME="${NODE_LOCAL}/home"
else
  echo "=== node-local mirror absent on $(hostname); running from /shared (slow NFS) ==="
  export HOME="${BENCH}/home"
fi
mkdir -p "${HOME}"

export SERVER_SCRIPT="${E2_DIR}/scripts/start_dynamo_stack.sh"
export SGLANG_URL="http://127.0.0.1:8888"
export MODEL="zai-org/GLM-5.3"
export TOKENIZER="/mnt/hf_hub_cache/GLM-5.3-W4AFP8"
export RESULTS_ROOT="${E2_DIR}/results/glm53/agentx_dynamo"
export OUTPUT_DIR="${RESULTS_ROOT}/policies_v2/phase4_prodverify"
export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache"
export READY_ON_MODEL=1
export SERVER_PROC_PATTERN='dynamo\.sglang'
export SERVER_METRICS_URLS="127.0.0.1:8081"
export SGLANG_ALLOW_PTRACE=1
export HEALTH_TIMEOUT=3600
export FORCE_RESTART=1
export GRACE_PERIOD=360
export SEED=42
export DURATION=1800
export STOP_SERVER=always
export HF_HOME="${BENCH}/hf"
export BOOST_ADAPTIVE_GAMMA=0
export BOOST_CLOCK=wall
export BOOST_BIN_SIZE=256
export IN_BATCH_PREFIX_CACHING_MAX_PIN_S=60

cd "${E2_DIR}"

run_cell() {  # $1 = policy, $2 = boost gamma (empty for non-boost)
  local p="$1" g="$2" label
  if [[ "${p}" == "boost" ]]; then
    label="c64_g${g}_wall_k256"
  else
    label="c64"
  fi
  if ls "${OUTPUT_DIR}/${p}/${label}_"*/e2_metrics.json >/dev/null 2>&1; then
    echo "=== E2 resume: skip completed ${p} ${label} ==="
    return 0
  fi
  echo "=== E2 phase4 prod-verify cell: policy=${p} c64 ${g:+gamma=${g} }==="
  if [[ "${p}" == "boost" ]]; then
    BOOST_GAMMA="${g}" bash scripts/run_e2.sh policy boost 64 \
      || echo "=== E2 cell failed: ${p} ${label} (will retry on resubmit) ==="
  else
    bash scripts/run_e2.sh policy "${p}" 64 \
      || echo "=== E2 cell failed: ${p} ${label} (will retry on resubmit) ==="
  fi
}

run_cell dfs-weight ""
run_cell boost "${BOOST_SWEEP_GAMMA}"
if [[ "${BOOST_SWEEP_GAMMA}" != "15.0" ]]; then
  run_cell boost "15.0"
fi
echo "=== PHASE4 DONE (keep raw exports; run focus analysis next) ==="
