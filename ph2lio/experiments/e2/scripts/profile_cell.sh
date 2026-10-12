#!/usr/bin/env bash
# E2 one-off profiling cell: py-spy scheduler flames + a 30-step torch
# profiler window + 1 s GPU sampling around ONE (arm, policy, concurrency)
# point, submitted SEPARATELY from the sweeps. py-spy sampling and the torch
# profiler perturb the measured process, so this cell must never run inside
# a sweep — the policy comparison stays clean (runs there are untouched).
#
# Cell choice rule: the worst output-tok/s cell of the sweep per
# e2_metrics_v2.json, tie-break ITL p99. Expected default: glm53 at the
# highest concurrency on the losing policy (that arm carries the extra
# degrees of freedom: DP-attention sync, deepep, tokenizer workers, and has
# the richest corroborating metrics).
#
# Usage (arm must match the node the weights live on — override the default
# #SBATCH nodelist below per arm: glm53 -> hgx-h200-01, qwen -> hgx-h200-02):
#   sbatch [--nodelist=<node>] scripts/profile_cell.sh <arm> <conc> <policy>
#     arm: glm53 | qwen      policy: fcfs-match lpm hrrn dfs-weight boost
#
# Env knobs (all optional):
#   PROBE_DURATION_S       aiperf duration, default 900 (scenario minimum;
#                          the probe windows below fit well inside it)
#   STEADY_STATE_DELAY_S   wait after aiperf starts before attaching,
#                          default 120
#   PYSPY_RATE             py-spy sample rate Hz, default 25
#   PYSPY_DURATION_S       py-spy record duration, default 120
#   PROFILE_STEPS          torch profiler step budget, default 30 (traces
#                          land under <run_dir>/torch_traces; GBs on /shared)
#   EXTRA_PYSPY_PATTERNS   extra pgrep -f patterns to flame, space-separated
#                          (e.g. 'sglang::detokenizer'); default none
#
# Pre-flight, once on a booted server before spending a real cell:
#   /shared/loipp/venv-server/bin/py-spy dump --pid <one scheduler pid>
#   curl -X POST $SGLANG_URL/start_profile -d '{"output_dir": "<scratch>",
#        "num_steps": 3, "activities": ["CPU","GPU"]}'
# If the body output_dir is ignored by the server build, export
# SGLANG_TORCH_PROFILER_DIR in this script's env only.

#SBATCH --job-name=e2-profile-cell
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# All 192 CPUs: 64-cpu jobs get node-0-only affinity and the server's NUMA
# binding silently skips the node-1 half (see the smoke 2623 log).
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --output=/shared/loipp/logs/e2_profile_cell_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"

ARM="${1:?usage: profile_cell.sh <glm53|qwen> <conc> <policy>}"
CONC="${2:?usage: profile_cell.sh <glm53|qwen> <conc> <policy>}"
POLICY="${3:?usage: profile_cell.sh <glm53|qwen> <conc> <policy>}"

case "${ARM}" in
  glm53)
    export SERVER_SCRIPT="${E2_DIR}/scripts/start_server_h200_glm53.sh"
    export MODEL="zai-org/GLM-5.3"
    export TOKENIZER="/mnt/hf_hub_cache/GLM-5.3-W4AFP8"
    export RESULTS_ROOT="${E2_DIR}/results/8x-h200/glm53/agentx"
    export HEALTH_TIMEOUT=7200    # cold NFS boot measured > 60 min (2623)
    export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache"
    ;;
  qwen)
    export SERVER_SCRIPT="${E2_DIR}/scripts/start_server_h200_qwen.sh"
    export MODEL="Qwen/Qwen3.8-27B-FP8"
    export TOKENIZER="/mnt/hf_hub_cache/Qwen3.8-27B-FP8"
    export RESULTS_ROOT="${E2_DIR}/results/8x-h200/qwen/agentx"
    export HEALTH_TIMEOUT=5400
    export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache_n02"
    ;;
  *) echo "ERROR: arm must be glm53 or qwen (got '${ARM}')" >&2; exit 2 ;;
esac

# run_e2.sh knobs: identical regime to the sweep cells, except a shorter
# duration and finer GPU sampling; results land under profiling/, never in
# the sweep result tree.
export SGLANG_URL="http://127.0.0.1:30000"
export AIPERF="${BENCH}/venv-aiperf/bin/aiperf"
export OUTPUT_DIR="${RESULTS_ROOT}/profiling"
export DURATION="${PROBE_DURATION_S:-900}"
export FORCE_RESTART=1
export GRACE_PERIOD=360
export SEED=42
export BOOST_ADAPTIVE_GAMMA=1
export GPU_SAMPLER_INTERVAL=1
export HF_HOME="${BENCH}/hf"

# Compute nodes get /home/loipp from the submitter but /home is login-local:
# torch dynamo's cache bootstrap dies with EACCES at import otherwise.
export HOME="${BENCH}/home"
mkdir -p "${HOME}"

PYSPY="${BENCH}/venv-server/bin/py-spy"
PYSPY_RATE="${PYSPY_RATE:-25}"
PYSPY_DURATION_S="${PYSPY_DURATION_S:-120}"
PROFILE_STEPS="${PROFILE_STEPS:-30}"
STEADY_STATE_DELAY_S="${STEADY_STATE_DELAY_S:-120}"
PROBE_WAIT_TIMEOUT_S="${PROBE_WAIT_TIMEOUT_S:-300}"

PROBE_PIDS=()

cleanup_probes() {
  local pid
  for pid in "${PROBE_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_probes EXIT

server_healthy() { curl -sf "${SGLANG_URL}/health" >/dev/null 2>&1; }

echo "=== E2 profiling cell: arm=${ARM} policy=${POLICY} conc=${CONC} duration=${DURATION}s ==="
cd "${E2_DIR}"
bash scripts/run_e2.sh policy "${POLICY}" "${CONC}" &
RUN_PID=$!

# Attach only after the run reaches steady state: server healthy, aiperf
# launched (command.txt is its first artifact), then a fixed delay. If the
# run dies or never starts aiperf, bail out of the probes but still let the
# run finish so its own failure is visible.
until server_healthy; do
  kill -0 "${RUN_PID}" 2>/dev/null || { echo "run_e2 exited before the server was healthy"; wait "${RUN_PID}" || true; exit 1; }
  sleep 10
done
run_dir="$(ls -td "${OUTPUT_DIR}/${POLICY}"/c${CONC}_* 2>/dev/null | head -1 || true)"
if [[ -z "${run_dir}" ]]; then
  echo "ERROR: no run dir under ${OUTPUT_DIR}/${POLICY}; aborting probes" >&2
  wait "${RUN_PID}" || true
  exit 1
fi
waited=0
until [[ -f "${run_dir}/command.txt" ]]; do
  kill -0 "${RUN_PID}" 2>/dev/null || break
  (( waited += 5 )); (( waited >= PROBE_WAIT_TIMEOUT_S )) && break
  sleep 5
done
if [[ -f "${run_dir}/command.txt" ]]; then
  echo "probing run dir: ${run_dir} (steady state in ${STEADY_STATE_DELAY_S}s)"
  sleep "${STEADY_STATE_DELAY_S}"
else
  echo "WARN: aiperf did not start within ${PROBE_WAIT_TIMEOUT_S}s; probes skipped" >&2
  wait "${RUN_PID}" || true
  exit 1
fi

probe_dir="${run_dir}/profiling"
mkdir -p "${probe_dir}"

# py-spy: one flame per scheduler process (8 on either arm). Attach
# failures (ptrace scope) are warned per pid, not fatal.
pyspy_patterns=('sglang::scheduler')
read -r -a extra_patterns <<< "${EXTRA_PYSPY_PATTERNS:-}" || true
pyspy_patterns+=("${extra_patterns[@]:-}")
for pattern in "${pyspy_patterns[@]}"; do
  for pid in $(pgrep -f "${pattern}" || true); do
    # shellcheck disable=SC2024
    "${PYSPY}" record -r "${PYSPY_RATE}" -d "${PYSPY_DURATION_S}" \
      -o "${probe_dir}/pyspy_${pattern//::/_}_${pid}.svg" --pid "${pid}" \
      >"${probe_dir}/pyspy_${pid}.log" 2>&1 &
    PROBE_PIDS+=("$!")
    echo "py-spy record: pattern=${pattern} pid=${pid} rate=${PYSPY_RATE} duration=${PYSPY_DURATION_S}s"
  done
done

# torch profiler: one window, auto-stops after the step budget.
if ! curl -sf -X POST "${SGLANG_URL}/start_profile" \
    -d "{\"output_dir\": \"${probe_dir}/torch_traces\", \"num_steps\": ${PROFILE_STEPS}, \"activities\": [\"CPU\", \"GPU\"]}" \
    >"${probe_dir}/start_profile_response.json" 2>"${probe_dir}/start_profile_err.log"; then
  echo "WARN: /start_profile failed; see ${probe_dir}/start_profile_err.log" >&2
else
  echo "torch profiler window started: ${PROFILE_STEPS} steps -> ${probe_dir}/torch_traces"
fi

# py-spy self-terminates after its duration; reap and report status per pid.
for pid in "${PROBE_PIDS[@]:-}"; do
  if wait "${pid}"; then
    echo "py-spy pid ${pid}: done"
  else
    echo "WARN: py-spy pid ${pid} exited nonzero (ptrace denied?)" >&2
  fi
done

# The boot log lives in the sweep tree (start_server writes there); link it
# next to the run so bottleneck_report.py finds it without copying GBs.
newest_server_log="$(ls -t "${RESULTS_ROOT}/${POLICY}"/server_*.log 2>/dev/null | head -1 || true)"
if [[ -n "${newest_server_log}" ]]; then
  ln -sf "${newest_server_log}" "${run_dir}/server_$(basename "${newest_server_log}")"
fi

wait "${RUN_PID}"
echo "=== PROFILING CELL DONE: ${run_dir} ==="
