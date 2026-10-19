#!/usr/bin/env bash
# E2 boost campaign — Phase 2: policy comparison on the dynamo arm.
# 7 policies x c{16,32,64}, DURATION=1800, seed 42, fresh boot per cell:
#   dfs-weight (prod policy, first), fcfs-match, lpm, hrrn,
#   shortest-prefill-first, lof, boost at the phase-1 sweet-spot gamma
#   (BOOST_SWEEP_GAMMA, default 5.0) + boost at gamma=15 (near-FCFS bound).
# pin=60 default everywhere (prod parity).
#
# 21+3 cells ≈ 24 h → expect ~6-7 submissions (4 h wall); resume loop skips
# completed cells. dfs-weight first: prod-config reference cells.
#
# Prereqs: phase1 DONE + sweet-spot gamma chosen (pass --export=BOOST_SWEEP_GAMMA=<g>
# if it differs from 5.0).
#SBATCH --job-name=e2boost-p2-policies
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
#SBATCH --output=/shared/loipp/logs/e2boost_p2_policies_%j.log

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
export RESULTS_ROOT="${E2_DIR}/results/8x-h200/glm53/agentx_dynamo"
export OUTPUT_DIR="${RESULTS_ROOT}/policies_v2/phase2_policies"
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

run_cell() {  # $1 = policy, $2 = conc, $3 = boost gamma (empty for non-boost)
  local p="$1" c="$2" g="$3" label
  if [[ "${p}" == "boost" ]]; then
    label="c${c}_g${g}_wall_k256"
  else
    label="c${c}"
  fi
  if ls "${OUTPUT_DIR}/${p}/${label}_"*/e2_metrics.json >/dev/null 2>&1; then
    echo "=== E2 resume: skip completed ${p} ${label} ==="
    return 0
  fi
  echo "=== E2 phase2 cell: policy=${p} c${c} ${g:+gamma=${g} }==="
  if [[ "${p}" == "boost" ]]; then
    BOOST_GAMMA="${g}" bash scripts/run_e2.sh policy boost "${c}" \
      || echo "=== E2 cell failed: ${p} ${label} (will retry on resubmit) ==="
  else
    bash scripts/run_e2.sh policy "${p}" "${c}" \
      || echo "=== E2 cell failed: ${p} ${label} (will retry on resubmit) ==="
  fi
  # Rolling disk policy (plan §H.3): completed cells keep e2_metrics* +
  # profile_export + schedule_eval raw; the GB-scale server_metrics_export
  # gzips ~8-12x. If a cell later needs timeseries extraction
  # (extract_metrics_timeseries.py), gunzip it first (srun, then re-gzip).
  find "${OUTPUT_DIR}" -name e2_metrics.json | while read -r m; do
    d="$(dirname "${m}")"
    [[ -f "${d}/server_metrics_export.jsonl" ]] && gzip -f "${d}/server_metrics_export.jsonl"
  done
}

# dfs-weight first (prod reference), boost arms last.
for c in 16 32 64; do run_cell dfs-weight "${c}" ""; done
for p in fcfs-match lpm hrrn shortest-prefill-first lof; do
  for c in 16 32 64; do run_cell "${p}" "${c}" ""; done
done
for c in 16 32 64; do run_cell boost "${c}" "${BOOST_SWEEP_GAMMA}"; done
if [[ "${BOOST_SWEEP_GAMMA}" != "15.0" ]]; then
  for c in 16 32 64; do run_cell boost "${c}" "15.0"; done
fi
echo "=== PHASE2 DONE ==="
