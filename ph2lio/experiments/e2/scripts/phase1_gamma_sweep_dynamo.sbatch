#!/usr/bin/env bash
# E2 boost campaign — Phase 1: static gamma sweep on the dynamo arm
# (future_plan_e2_policies.md §4.1). The RTX sweet spot was gamma=5.0; on
# 8xH200 TP8 the prefill throughput is ~8x higher so the optimum may shift.
# Sweep wall-clock gamma {1,3,5,7,10,15} at the concurrency chosen by the
# phase-0 gate (PHASE1_CONC below), k=256, adaptive OFF (static arms must be
# static), pin=60 (prod parity + comparability with the H200 dfs-weight
# cells; the RTX sweep's pin=0 was an RTX-regime choice).
#
# 6 cells x ~55-70 min ≈ 7 h → expect 2 submissions (4 h wall); the resume
# loop skips completed cells on resubmit.
#
# Prereqs: phase0_loadband_dynamo.sbatch DONE + its gate evaluated;
# PHASE1_CONC set to the chosen band concurrency.
#SBATCH --job-name=e2boost-p1-gamma
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1600G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/loipp/logs/e2boost_p1_gamma_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/lopp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"
PHASE1_CONC="${PHASE1_CONC:-16}"   # from the phase-0 gate; submit with --export

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
export RESULTS_ROOT="${E2_DIR}/results/8x-h200/glm53/agentx_dynamo"
export OUTPUT_DIR="${RESULTS_ROOT}/policies_v2/phase1_gamma"
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
# Static boost arm: adaptive OFF, wall clock, k=256, pin=60 (see header).
export BOOST_ADAPTIVE_GAMMA=0
export BOOST_CLOCK=wall
export BOOST_BIN_SIZE=256
export IN_BATCH_PREFIX_CACHING_MAX_PIN_S=60

cd "${E2_DIR}"

# run_label in run_e2.sh: static boost = c<conc>_g<gamma>_<clock>_k<bin>
GAMMAS=(1.0 3.0 5.0 7.0 10.0 15.0)

gzip_cell_exports() {
  find "${OUTPUT_DIR}" -name e2_metrics.json | while read -r m; do
    d="$(dirname "${m}")"
    [[ -f "${d}/server_metrics_export.jsonl" ]] && gzip -f "${d}/server_metrics_export.jsonl"
  done
}

for g in "${GAMMAS[@]}"; do
  label="c${PHASE1_CONC}_g${g}_wall_k256"
  if ls "${OUTPUT_DIR}/boost/${label}_"*/e2_metrics.json >/dev/null 2>&1; then
    echo "=== E2 resume: skip completed boost ${label} ==="
    continue
  fi
  echo "=== E2 phase1 gamma cell: boost gamma=${g} c${PHASE1_CONC} ==="
  BOOST_GAMMA="${g}" bash scripts/run_e2.sh policy boost "${PHASE1_CONC}" \
    || echo "=== E2 cell failed: boost ${label} (will retry on resubmit) ==="
  gzip_cell_exports
done
echo "=== PHASE1 DONE (pick the sweet-spot gamma before phase 2) ==="
