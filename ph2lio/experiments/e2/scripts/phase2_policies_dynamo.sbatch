#!/usr/bin/env bash
# E2 boost campaign — Phase 2: policy comparison on the dynamo arm.
# 7 policies x c{16,32,64}, DURATION=1800, fresh boot per cell:
#   dfs-weight (prod policy, first), fcfs-match, lpm, hrrn,
#   shortest-prefill-first, lof, boost at the phase-1 sweet-spot gamma
#   (BOOST_SWEEP_GAMMA, default 5.0) + boost at gamma=15 (near-FCFS bound).
# pin=60 default everywhere (prod parity).
#
# Paired multi-seed (2026-10-19 RTX audit, doc e2_policies.md / plan §5.5):
# seed variance (60-89%) dwarfs the ~15% policy gaps, so single-seed
# rankings are indicative only — H200 comparisons should use 3+ seeds per
# arm. Seed-MAJOR order: each seed block completes all arms, so a partial
# campaign still yields balanced paired comparisons (same seed = same
# request mix); seed 42 goes first and alone yields the full ranking.
# Default SEEDS_LIST="42" keeps the original single-seed scope; submit
# --export=SEEDS_LIST="42 43 44" (or 5 seeds) for the fair sweep
# (cells scale linearly: 24 cells/seed).
#
# 24 cells/seed x ~55-70 min -> ~24 h/seed; expect ~6-7 submissions (4 h
# wall); resume loop skips completed (policy, conc, seed) cells.
#
# Prereqs: phase1 DONE + sweet-spot gamma chosen (pass --export=BOOST_SWEEP_GAMMA=<g>
# if it differs from 5.0). CONCS may need retargeting after the phase-0
# gate if the load band lands at c40/c48 instead.
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
SEEDS_LIST="${SEEDS_LIST:-42}"   # paired seeds, seed-major (see header)
CONCS="${CONCS:-16 32 64}"       # revisit after the phase-0 gate (band may be 40/48)

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

seed_done() {  # $1 = policy, $2 = label, $3 = seed: completed cell with this seed?
  local d
  for d in "${OUTPUT_DIR}/$1/$2"_*/; do
    [[ -f "${d}e2_metrics.json" && -f "${d}run_meta.json" ]] || continue
    grep -q "\"seed\": $3," "${d}run_meta.json" && return 0
  done
  return 1
}

run_cell() {  # $1 = policy, $2 = conc, $3 = boost gamma (empty for non-boost), $4 = seed
  local p="$1" c="$2" g="$3" seed="$4" label
  if [[ "${p}" == "boost" ]]; then
    label="c${c}_g${g}_wall_k256"
  else
    label="c${c}"
  fi
  if seed_done "${p}" "${label}" "${seed}"; then
    echo "=== E2 resume: skip completed ${p} ${label} seed=${seed} ==="
    return 0
  fi
  echo "=== E2 phase2 cell: policy=${p} c${c} ${g:+gamma=${g} }seed=${seed} ==="
  if [[ "${p}" == "boost" ]]; then
    BOOST_GAMMA="${g}" SEED="${seed}" bash scripts/run_e2.sh policy boost "${c}" \
      || echo "=== E2 cell failed: ${p} ${label} seed=${seed} (will retry on resubmit) ==="
  else
    SEED="${seed}" bash scripts/run_e2.sh policy "${p}" "${c}" \
      || echo "=== E2 cell failed: ${p} ${label} seed=${seed} (will retry on resubmit) ==="
  fi
  # Rolling disk policy (plan §H.3): completed cells keep e2_metrics* +
  # profile_export + schedule_eval raw; the GB-scale server_metrics_export
  # gzips ~8-12x. If a cell later needs timeseries extraction
  # (extract_metrics_timeseries.py), gunzip it first (srun, then re-gzip).
  find "${OUTPUT_DIR}" -name e2_metrics.json | while read -r m; do
    d="$(dirname "${m}")"
    # `if` (not `[[ ]] && gzip`): a false guard as the loop's last status
    # returns 1 and, under set -euo pipefail (find | while), kills the whole
    # driver after the last cell — job 4391 died exactly this way.
    if [[ -f "${d}/server_metrics_export.jsonl" ]]; then gzip -f "${d}/server_metrics_export.jsonl"; fi
  done
}

# Seed-major (paired): seed 42 block first — it alone completes the full
# 24-cell ranking; later seeds only refine it. dfs-weight first within each
# seed (prod reference), boost arms last.
read -ra SEEDS <<< "${SEEDS_LIST}"
read -ra CONC_LIST <<< "${CONCS}"
for seed in "${SEEDS[@]}"; do
  for c in "${CONC_LIST[@]}"; do run_cell dfs-weight "${c}" "" "${seed}"; done
  for p in fcfs-match lpm hrrn shortest-prefill-first lof; do
    for c in "${CONC_LIST[@]}"; do run_cell "${p}" "${c}" "" "${seed}"; done
  done
  for c in "${CONC_LIST[@]}"; do run_cell boost "${c}" "${BOOST_SWEEP_GAMMA}" "${seed}"; done
  if [[ "${BOOST_SWEEP_GAMMA}" != "15.0" ]]; then
    for c in "${CONC_LIST[@]}"; do run_cell boost "${c}" "15.0" "${seed}"; done
  fi
done
echo "=== PHASE2 DONE ==="
