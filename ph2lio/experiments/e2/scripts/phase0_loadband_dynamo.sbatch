#!/usr/bin/env bash
# E2 boost campaign — Phase 0: load-band recalibration on the dynamo arm.
# 3 short probe cells (DURATION=600) with dfs-weight at c16/c32/c64 to find
# the concurrency band where #queue-req is persistently > 0 but not drowned
# (the 5 old c64 cells showed 73k-132k pending = saturated). The chosen band
# feeds phase 1's gamma sweep. Gate (evaluated after the cells, from the
# engine log + schedule_eval queue_wait mass): highest c with a persistent
# but modest queue.
#
# Prereqs: mirror_from_login.sh hgx-h200-01 re-run after the boost merge
# (tag boost-h200-merged); smoke_dynamo_stack.sbatch green.
#SBATCH --job-name=e2boost-p0-loadband
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1600G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/loipp/logs/e2boost_p0_loadband_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/lopp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"

# --- node-local execution mirror (see run_e2_h200_glm53.sbatch for the why) ---
NODE_LOCAL="${NODE_LOCAL:-/mnt/loipp}"
if [ -x "${NODE_LOCAL}/venv-dynamo/bin/python3" ]; then
  export AIPERF="${NODE_LOCAL}/venv-aiperf/bin/aiperf"
  export HOME="${NODE_LOCAL}/home"
else
  echo "=== node-local mirror absent on $(hostname); running from /shared (slow NFS) ==="
  export HOME="${BENCH}/home"
fi
mkdir -p "${HOME}"

# --- dynamo arm knobs (profile_cell.sh dynamo arm) ---------------------------
export SERVER_SCRIPT="${E2_DIR}/scripts/start_dynamo_stack.sh"
export SGLANG_URL="http://127.0.0.1:8888"
export MODEL="zai-org/GLM-5.3"
export TOKENIZER="/mnt/hf_hub_cache/GLM-5.3-W4AFP8"
export RESULTS_ROOT="${E2_DIR}/results/8x-h200/glm53/agentx_dynamo"
export OUTPUT_DIR="${RESULTS_ROOT}/policies_v2/phase0_loadband"
export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache"
export READY_ON_MODEL=1
export SERVER_PROC_PATTERN='dynamo\.sglang'
export SERVER_METRICS_URLS="127.0.0.1:8081"
export SGLANG_ALLOW_PTRACE=1
export HEALTH_TIMEOUT=3600
export FORCE_RESTART=1
export GRACE_PERIOD=360
export SEED=42
export DURATION=600          # probe cells only — not comparable to 1800s cells
export STOP_SERVER=always
export HF_HOME="${BENCH}/hf"

cd "${E2_DIR}"

gzip_cell_exports() {  # rolling disk policy (plan §H.3): completed probe cells
  # keep e2_metrics*.json + profile_export + schedule_eval; the GB-scale
  # server_metrics_export.jsonl gzips ~8-12x. Phase-0 probes never need
  # timeseries extraction.
  find "${OUTPUT_DIR}" -name e2_metrics.json | while read -r m; do
    d="$(dirname "${m}")"
    [[ -f "${d}/server_metrics_export.jsonl" ]] && gzip -f "${d}/server_metrics_export.jsonl"
  done
}

for c in 16 32 64; do
  if ls "${OUTPUT_DIR}/dfs-weight/c${c}_"*/e2_metrics.json >/dev/null 2>&1; then
    echo "=== E2 resume: skip completed dfs-weight c${c} ==="
    continue
  fi
  echo "=== E2 phase0 probe: policy=dfs-weight concurrency=${c} duration=600 ==="
  bash scripts/run_e2.sh policy dfs-weight "${c}" \
    || echo "=== E2 cell failed: dfs-weight c${c} (will retry on resubmit) ==="
  gzip_cell_exports
done
echo "=== PHASE0 DONE (evaluate the queue-depth gate before phase 1) ==="
