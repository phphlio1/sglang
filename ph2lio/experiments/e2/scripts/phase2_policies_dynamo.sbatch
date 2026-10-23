#!/usr/bin/env bash
# E2 boost campaign — Phase 2: 6-arm policy comparison on the dynamo arm.
# 6 policies x c{64}, one seed per job (Seeds block), DURATION=1800, fresh
# boot per cell, pin=60 (prod parity):
#   dfs-weight (prod policy, baseline, first), fcfs (plain, matches the RTX
#   tables), lpm,
#   hrrn, boost static gamma=0.25 (RTX 10-arm sweep representative: top-2 in
#   both regimes, best light-regime arm), boost adaptive (gamma-Ada Eq. 9
#   fit, jump floor J=BOOST_ADAPTIVE_MAX_JUMP_S).
#
# J derivation (2026-10-23, 5 valid phase-1 cells, n=419 cold giants >=50k
# tok with matched<=10%): prefill p50=19.8 p95=46.3 p99=65.7 s at ~4.9-5.3k
# uncached tok/s, stable across gamma arms (measured on node 1 while it
# still had 8 GPUs; node 2 is the same 8xH200 platform, so the rig
# carries). Default J=50 = 1.08x the p95
# giant prefill; the RTX J-grid verdict was J in [1,2]x one giant prefill
# wins and <1x loses (j60 beat j15 at overload; RTX giants 30-60 s). At ~5k
# tok/s the floor pins gamma~0.105 under overload (RTX-j60-like lean) while
# static gamma=0.25 maps to J~17.5 s (RTX j15-zone) -- the two boost arms
# span the lean range. Mini-sweep {half,one,double} stays available via
# --export=ALL,BOOST_ADAPTIVE_MAX_JUMP_S=25|50|100.
#
# Seeds: the RTX campaign pair -- 77188 (overload regime there) + 42302
# (light regime there) -- same trace seeds so the request mix matches the
# RTX tables. The regime label is rig x conc dependent: on H200 at c64
# both seeds may sit inside the load band; the pairing still controls the
# mix. RTX numbers are NOT directly comparable (pin=0 + c16 there vs
# pin=60 + c64 here).
#
# ONE SEED PER JOB (user flow 2026-10-23): default SEEDS_LIST="77188";
# queue one sbatch per seed (--export=ALL,SEEDS_LIST=42302 for the second,
# more seeds the same way -- 3 seeds = 3 jobs) -- the node-pinned jobs
# serialize in the queue and each carries a full 6-arm pass for its seed.
# N_RUNS (default 1) = target completed runs per (policy, label, seed)
# cell; a cell skips only at N_RUNS completed runs. Repeat passes of the
# SAME seed submit the same job with --export=ALL,N_RUNS=2, then 3 (each
# pass adds exactly one run per cell; the seed fixes the request mix, not
# the realization, so repeats still vary -- the RTX n=2 protocol worked
# this way).
#
# Seed-MAJOR order: each seed block completes all 6 arms, so a partial
# campaign still yields balanced paired comparisons (same seed = same
# request mix); seed 77188 goes first.
# Per job (1 seed): 6 arms x 1 conc = 6 cells x ~55-70 min -> ~5.5-7 h vs
# the 12 h wall (temporary, was 4 h), ~2x headroom; a failed cell retries
# on resubmit (resume skips a cell at N_RUNS completed runs). c48 only if
# time permits: --export=ALL,CONCS="48 64" reruns resume-safely
# (completed c64 cells skip).
#
# Prereqs: phase-1 DONE (J derivation above); the node-local mirror
# (/mnt/loipp, ~21g) and the 373g model under /mnt/hf_hub_cache present on
# the target node -- the driver fatals early otherwise; CONCS may need
# retargeting after the phase-0 gate.
#SBATCH --job-name=e2boost-p2-policies
#SBATCH --partition=hardware-hcm
# Node 2 since 2026-10-23: node 1 was downgraded 8 -> 4 GPUs, so 8-GPU
# cells moved to hgx-h200-02 (same HGX H200 platform).
#SBATCH --nodelist=hgx-h200-02
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# Must stay >96: the server binds GPUs 4-7 to NUMA node 1 (--numa-node 0 0 0 0
# 1 1 1 1); <=96 CPUs lands the whole allocation on one NUMA node and the
# binding is skipped at boot (smoke 2623 died at 64). Measured demand is
# ~16-20 cores — the count is purely topological. 128 since 2026-10-19.
#SBATCH --cpus-per-task=128
#SBATCH --mem=1600G
# 12 h wall, temporary (user request 2026-10-23; was 4 h).
#SBATCH --time=12:00:00
#SBATCH --output=/shared/loipp/logs/e2boost_p2_policies_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"
BOOST_SWEEP_GAMMA="${BOOST_SWEEP_GAMMA:-0.25}"  # RTX representative static gamma
BOOST_ADAPTIVE_MAX_JUMP_S="${BOOST_ADAPTIVE_MAX_JUMP_S:-50}"  # 1.08x p95 giant prefill (header)
SEEDS_LIST="${SEEDS_LIST:-77188}"  # one seed per job; 42302 + later seeds via --export=ALL,SEEDS_LIST=<seed>
N_RUNS="${N_RUNS:-1}"              # target completed runs/cell; same-seed passes: --export=ALL,N_RUNS=2,3,...
CONCS="${CONCS:-64}"       # c48 only if time permits (--export=ALL,CONCS="48 64")

NODE_LOCAL="${NODE_LOCAL:-/mnt/loipp}"
if [ ! -x "${NODE_LOCAL}/venv-dynamo/bin/python3" ]; then
  # A bare-/shared boot reads the venvs over NFS at ~0.1-1.4 MB/s -- one
  # stack boot takes hours (start_dynamo_stack.sh header), so the mirror
  # is a hard prerequisite. Build it from the login node first:
  #   bash mirror_from_login.sh <this-node>
  echo "=== FATAL: ${NODE_LOCAL} mirror absent on $(hostname); run mirror_from_login.sh first ==="
  exit 1
fi
export AIPERF="${NODE_LOCAL}/venv-aiperf/bin/aiperf"
export HOME="${NODE_LOCAL}/home"
mkdir -p "${HOME}"

export SERVER_SCRIPT="${E2_DIR}/scripts/start_dynamo_stack.sh"
export SGLANG_URL="http://127.0.0.1:8888"
export MODEL="zai-org/GLM-5.3"
export TOKENIZER="/mnt/hf_hub_cache/GLM-5.3-W4AFP8"
# The engine (GLM_MODEL_PATH defaults to TOKENIZER in start_dynamo_stack.sh)
# and the aiperf client both read these 373g of node-local weights; there is
# no shared master (hf/hub carries only the dataset). Copy node->node before
# running on a node that lacks them.
if [ ! -d "${TOKENIZER}" ]; then
  echo "=== FATAL: model path ${TOKENIZER} absent on $(hostname) ==="
  exit 1
fi
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

runs_done() {  # $1 = policy, $2 = label, $3 = seed: count completed runs of this cell
  local d n=0
  for d in "${OUTPUT_DIR}/$1/$2"_*/; do
    [[ -f "${d}e2_metrics.json" && -f "${d}run_meta.json" ]] || continue
    # `if` (not `grep && n=...`): keep every statement's status 0 under
    # set -e (same hazard class as the gzip loop below).
    if grep -q "\"seed\": $3," "${d}run_meta.json"; then n=$((n + 1)); fi
  done
  echo "${n}"
}

run_cell() {  # $1 = policy, $2 = conc, $3 = boost mode ("" | static gamma | "ada"), $4 = seed
  local p="$1" c="$2" mode="$3" seed="$4" label n
  case "${mode}" in
    ada) label="c${c}_gada_${BOOST_CLOCK}_k${BOOST_BIN_SIZE}_j${BOOST_ADAPTIVE_MAX_JUMP_S}" ;;
    "")  label="c${c}" ;;
    *)   label="c${c}_g${mode}_${BOOST_CLOCK}_k${BOOST_BIN_SIZE}" ;;
  esac
  n="$(runs_done "${p}" "${label}" "${seed}")"
  if [[ "${n}" -ge "${N_RUNS}" ]]; then
    echo "=== E2 resume: skip ${p} ${label} seed=${seed} (${n}/${N_RUNS} runs done) ==="
    return 0
  fi
  if [[ "${n}" -gt 0 ]]; then
    echo "=== E2 repeat pass ${n} -> ${N_RUNS}: ${p} ${label} seed=${seed} ==="
  fi
  echo "=== E2 phase2 cell: policy=${p} c=${c} ${mode:+mode=${mode} }seed=${seed} ==="
  if [[ "${p}" == "boost" && "${mode}" == "ada" ]]; then
    BOOST_ADAPTIVE_GAMMA=1 BOOST_ADAPTIVE_MAX_JUMP_S="${BOOST_ADAPTIVE_MAX_JUMP_S}" \
      SEED="${seed}" bash scripts/run_e2.sh policy boost "${c}" \
      || echo "=== E2 cell failed: ${p} ${label} seed=${seed} (will retry on resubmit) ==="
  elif [[ "${p}" == "boost" ]]; then
    BOOST_GAMMA="${mode}" SEED="${seed}" bash scripts/run_e2.sh policy boost "${c}" \
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

# Seed-major (paired): seed 77188 block first. dfs-weight first within each
# seed (prod baseline), classic policies next, boost arms last.
read -ra SEEDS <<< "${SEEDS_LIST}"
read -ra CONC_LIST <<< "${CONCS}"
for seed in "${SEEDS[@]}"; do
  for c in "${CONC_LIST[@]}"; do run_cell dfs-weight "${c}" "" "${seed}"; done
  for p in fcfs lpm hrrn; do
    for c in "${CONC_LIST[@]}"; do run_cell "${p}" "${c}" "" "${seed}"; done
  done
  for c in "${CONC_LIST[@]}"; do run_cell boost "${c}" "${BOOST_SWEEP_GAMMA}" "${seed}"; done
  for c in "${CONC_LIST[@]}"; do run_cell boost "${c}" "ada" "${seed}"; done
done
echo "=== PHASE2 DONE ==="
