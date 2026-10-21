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
# #SBATCH nodelist below per arm: glm53/dynamo -> hgx-h200-01, qwen -> hgx-h200-02):
#   sbatch [--nodelist=<node>] scripts/profile_cell.sh <arm> <conc> <policy>
#     arm: glm53 | qwen | dynamo   policy: fcfs-match lpm hrrn dfs-weight boost
#
# The dynamo arm is the FULL-STACK cell (dynamo frontend --router-mode=kv +
# engine worker under `python -m dynamo.sglang`, prod config verbatim — see
# start_dynamo_stack.sh): the engine-only glm53/qwen cells never exercised
# the frontend/router path, so their numbers cannot back the mentor
# deliverable's full-stack bottleneck claims.
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
#   NSYS_PROFILE           1 = wrap the engine in `nsys launch` (daemon mode,
#                          collection OFF until the window below) and collect
#                          one CUDA+NVTX window around the py-spy attach —
#                          memcpy/kernel/graph-exec evidence per rank; default
#                          0 = engine command byte-identical, no nsys at all
#   NSYS_DELAY_S           delay from aiperf start to the probe window; steers
#                          BOTH nsys and py-spy so the windows coincide;
#                          default = STEADY_STATE_DELAY_S (120)
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
# The server pins GPUs 4-7 to NUMA node 1 on the glm53 arm. Each NUMA node
# has 96 CPUs (2x48c/96t), so any request <= 96 lands on one node only and
# the node-1 binding is skipped at boot (2623, with 64). 160 > 96 forces
# Slurm to span both nodes while leaving 32 CPUs for teammates.
#SBATCH --cpus-per-task=160
# Same server pools as the sweep (~1.17 TB measured, see
# run_e2_h200_glm53.sbatch): the default per-CPU grant is below that.
#SBATCH --mem=1600G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/loipp/logs/e2_profile_cell_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"

ARM="${1:?usage: profile_cell.sh <glm53|qwen|dynamo> <conc> <policy>}"
CONC="${2:?usage: profile_cell.sh <glm53|qwen|dynamo> <conc> <policy>}"
POLICY="${3:?usage: profile_cell.sh <glm53|qwen|dynamo> <conc> <policy>}"

case "${ARM}" in
  glm53)
    export SERVER_SCRIPT="${E2_DIR}/scripts/start_server_h200_glm53.sh"
    export SGLANG_URL="http://127.0.0.1:30000"
    export MODEL="zai-org/GLM-5.3"
    export TOKENIZER="/mnt/hf_hub_cache/GLM-5.3-W4AFP8"
    export RESULTS_ROOT="${E2_DIR}/results/glm53/agentx"
    export HEALTH_TIMEOUT=7200    # cold NFS boot measured > 60 min (2623)
    export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache"
    ;;
  qwen)
    export SERVER_SCRIPT="${E2_DIR}/scripts/start_server_h200_qwen.sh"
    export SGLANG_URL="http://127.0.0.1:30000"
    export MODEL="Qwen/Qwen3.8-27B-FP8"
    export TOKENIZER="/mnt/hf_hub_cache/Qwen3.8-27B-FP8"
    export RESULTS_ROOT="${E2_DIR}/results/qwen/agentx"
    export HEALTH_TIMEOUT=5400
    export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache_n02"
    ;;
  dynamo)
    # Full stack: dynamo frontend (:8888) + engine worker under dynamo.sglang
    # (start_dynamo_stack.sh). run_e2.sh knobs specific to this stack:
    #   READY_ON_MODEL        dynamo /health is 200 with zero workers — real
    #                         readiness = served model on /v1/models
    #   SERVER_PROC_PATTERN   boot fail-fast pattern for `python -m
    #                         dynamo.sglang` (not sglang.launch_server)
    #   SERVER_METRICS_URLS   engine sglang:* registry lives on DYN_SYSTEM_PORT
    #                         8081 (smoke 3307), NOT on the frontend URL.
    #                         aiperf ALSO auto-scrapes <SGLANG_URL>/metrics
    #                         (8888) on its own — cell 3320's export holds
    #                         BOTH endpoints (9588 records each), so the
    #                         dynamo_frontend_*/router metrics come for free;
    #                         listing 8888 here would duplicate the scrape.
    #   SGLANG_ALLOW_PTRACE   schedulers call prctl(PR_SET_PTRACER_ANY)
    #                         (branch scheduler.py) so sibling py-spy attach
    #                         works under yama ptrace_scope=1 — without it
    #                         every flame is lost (all of cell 3065's)
    export SERVER_SCRIPT="${E2_DIR}/scripts/start_dynamo_stack.sh"
    export SGLANG_URL="http://127.0.0.1:8888"
    export MODEL="zai-org/GLM-5.3"
    export TOKENIZER="/mnt/hf_hub_cache/GLM-5.3-W4AFP8"
    export RESULTS_ROOT="${E2_DIR}/results/glm53/agentx_dynamo"
    export HEALTH_TIMEOUT=12000   # mirror boot measured 281 s to registration
    export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache"
    export READY_ON_MODEL=1
    export SERVER_PROC_PATTERN='dynamo\.sglang'
    export SERVER_METRICS_URLS="127.0.0.1:8081"
    export SGLANG_ALLOW_PTRACE=1
    ;;
  *) echo "ERROR: arm must be glm53, qwen or dynamo (got '${ARM}')" >&2; exit 2 ;;
esac

# run_e2.sh knobs: identical regime to the sweep cells, except a shorter
# duration and finer GPU sampling; results land under profiling/, never in
# the sweep result tree. SGLANG_URL is exported per-arm above (dynamo = the
# frontend :8888, engine-only arms = the sglang HTTP server :30000).
export OUTPUT_DIR="${RESULTS_ROOT}/profiling"
# --- node-local execution mirror (built by mirror_from_login.sh + ----------
# --- mirror_node_side.sh, run from the login node) ----------------------------
# venvs, the uv base interpreter, the sglang import tree, and the HOME
# compile caches live on this node's /mnt after the mirror job: imports and
# JIT-cache verification stop touching the ~1.4 MB/s NFS and survive
# teammates' page-cache eviction. Falls back to the /shared paths (slow)
# whenever the mirror is absent.
NODE_LOCAL="${NODE_LOCAL:-/mnt/loipp}"
if [ -x "${NODE_LOCAL}/venv-server/bin/python3" ] || [ -x "${NODE_LOCAL}/venv-dynamo/bin/python3" ]; then
  export AIPERF="${NODE_LOCAL}/venv-aiperf/bin/aiperf"
  # CUDA_HOME differs per stack: the engine-only server runs from venv-server
  # (py3.11), the dynamo stack from venv-dynamo (py3.12) — each venv carries
  # its own cu13 wheels and start_dynamo_stack.sh only fills CUDA_HOME when
  # it is unset, so an inherited wrong-venv path would win there.
  if [[ "${ARM}" == "dynamo" ]]; then
    export CUDA_HOME="${NODE_LOCAL}/venv-dynamo/lib/python3.12/site-packages/nvidia/cu13"
  else
    export SERVER_PYTHON="${NODE_LOCAL}/venv-server/bin/python3"
    export CUDA_HOME="${NODE_LOCAL}/venv-server/lib/python3.11/site-packages/nvidia/cu13"
  fi
  PYSPY="${NODE_LOCAL}/venv-server/bin/py-spy"
  export HOME="${NODE_LOCAL}/home"
else
  echo "=== node-local mirror absent on $(hostname); running from /shared (slow NFS) ==="
  export HOME="${BENCH}/home"
  PYSPY="${BENCH}/venv-server/bin/py-spy"
fi
mkdir -p "${HOME}"

export DURATION="${PROBE_DURATION_S:-900}"
export FORCE_RESTART=1
export GRACE_PERIOD=360
export SEED=42
export BOOST_ADAPTIVE_GAMMA=1
export GPU_SAMPLER_INTERVAL=1
export HF_HOME="${BENCH}/hf"

PYSPY_RATE="${PYSPY_RATE:-25}"
PYSPY_DURATION_S="${PYSPY_DURATION_S:-120}"
PROFILE_STEPS="${PROFILE_STEPS:-30}"
STEADY_STATE_DELAY_S="${STEADY_STATE_DELAY_S:-120}"
# NSys window (NSYS_PROFILE=1 cells): delay from aiperf start to the probe
# window, steering BOTH nsys and py-spy so the GPU window coincides with the
# python window. Defaults to the steady-state delay (the original probe
# placement). Exported for the engine wrap (session name) and run_meta.
NSYS_DELAY_S="${NSYS_DELAY_S:-${STEADY_STATE_DELAY_S}}"
if [[ "${NSYS_PROFILE:-0}" == "1" ]]; then
  export NSYS_SESSION_NAME="e2nsys-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
  export NSYS_DELAY_S
fi
# 900 covers a cold-mirror boot end to end: frontend :8888 answers /health
# within seconds (before any worker registers), so with the readiness gate
# below the clock starts at true readiness and aiperf's command.txt lands
# within seconds — but keep headroom anyway (3436's cold-JIT boot was 433 s
# from frontend-up to aiperf start and lost every probe at the old 300 s).
PROBE_WAIT_TIMEOUT_S="${PROBE_WAIT_TIMEOUT_S:-900}"

PROBE_PIDS=()

cleanup_probes() {
  # Finalize the nsys report FIRST (needs the session alive); plain `nsys
  # stop` keeps the engine running. Never `nsys shutdown` here — its default
  # --kill=sigterm would take the engine down with it.
  if [[ "${NSYS_COLLECTING:-0}" == "1" ]]; then
    timeout 900 nsys stop --session="${NSYS_SESSION_NAME:-}" \
      >>"${probe_dir:-/tmp}/nsys/nsys_stop.log" 2>&1 || true
    NSYS_COLLECTING=0
  fi
  local pid
  for pid in "${PROBE_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_probes EXIT

# Same readiness gate as run_e2.sh: the dynamo frontend answers /health 200
# with ZERO workers (READY_ON_MODEL), so gate on the served model too —
# otherwise the PROBE_WAIT clock starts at frontend-up and the whole engine
# boot (5–7+ min cold) counts against it (3436 skipped all probes this way).
# --max-time: a half-dead frontend makes a bare curl block forever (3437).
server_healthy() {
  curl -sf --max-time 10 "${SGLANG_URL}/health" >/dev/null 2>&1 || return 1
  [[ "${READY_ON_MODEL:-0}" == "1" ]] || return 0
  local m; m="$(mktemp)"
  curl -sf --max-time 10 "${SGLANG_URL}/v1/models" -o "${m}" 2>/dev/null \
    && grep -qE "\"id\": *\"${MODEL}\"" "${m}"
  local rc=$?
  rm -f "${m}"
  return "${rc}"
}

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
  echo "probing run dir: ${run_dir} (probe window in ${NSYS_DELAY_S}s)"
  sleep "${NSYS_DELAY_S}"
else
  echo "WARN: aiperf did not start within ${PROBE_WAIT_TIMEOUT_S}s; probes skipped" >&2
  wait "${RUN_PID}" || true
  exit 1
fi

probe_dir="${run_dir}/profiling"
mkdir -p "${probe_dir}"

# NSys collection window: start BEFORE the py-spy attach so the GPU window
# fully covers the python window. The report goes node-local first (/mnt is
# /dev/md0; /shared is NFS and `nsys stop` BLOCKS on the finalize) and is
# copied into the run dir at cell end, outside any measurement.
# --sample/--cpuctxsw none: without them `nsys start` defaults CPU sampling
# to process-tree — a duplicate of py-spy plus report bloat.
if [[ "${NSYS_PROFILE:-0}" == "1" ]]; then
  mkdir -p "${probe_dir}/nsys"
  if [[ -d "${NODE_LOCAL}" ]]; then
    NSYS_REPORT_DIR_LOCAL="${NODE_LOCAL}/nsys/${NSYS_SESSION_NAME}"
  else
    NSYS_REPORT_DIR_LOCAL="${probe_dir}/nsys"
  fi
  mkdir -p "${NSYS_REPORT_DIR_LOCAL}"
  NSYS_STAMP="$(date +%Y%m%d_%H%M%S)"
  NSYS_REPORT_LOCAL="${NSYS_REPORT_DIR_LOCAL}/engine_${NSYS_STAMP}.nsys-rep"
  if timeout 120 nsys start --session="${NSYS_SESSION_NAME}" \
      -o "${NSYS_REPORT_LOCAL}" --force-overwrite=true \
      --sample=none --cpuctxsw=none \
      >"${probe_dir}/nsys/nsys_start.log" 2>&1; then
    NSYS_COLLECTING=1
    # wall-clock anchors: phase-B joins nsys ns-since-collection-start
    # (≈ the nsys_start anchor) with schedule_eval arrival times.
    printf '%s nsys_start %s\n' "$(date +%s.%N)" "${NSYS_REPORT_LOCAL}" \
      >>"${probe_dir}/nsys/nsys_window.log"
    echo "nsys window started: session=${NSYS_SESSION_NAME} -> ${NSYS_REPORT_LOCAL}"
  else
    echo "WARN: nsys start failed; see ${probe_dir}/nsys/nsys_start.log" >&2
  fi
fi

# py-spy: one flame per scheduler process (8 on either arm). Attach
# failures (ptrace scope) are warned per pid, not fatal.
pyspy_patterns=('sglang::scheduler')
if [[ -n "${EXTRA_PYSPY_PATTERNS:-}" ]]; then
  # shellcheck disable=SC2206
  pyspy_patterns+=(${EXTRA_PYSPY_PATTERNS})
fi
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

# torch profiler: one window, auto-stops after the step budget. -S keeps the
# error message visible (a bare -sf swallows it; 3065 lost the reason).
if ! curl -sfS --max-time 120 -X POST "${SGLANG_URL}/start_profile" \
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

# End the nsys window right after the py-spy reap: `nsys stop` keeps the
# engine running and blocks only until the node-local report is finalized.
if [[ "${NSYS_COLLECTING:-0}" == "1" ]]; then
  printf '%s nsys_stop\n' "$(date +%s.%N)" >>"${probe_dir}/nsys/nsys_window.log"
  if timeout 900 nsys stop --session="${NSYS_SESSION_NAME}" \
      >"${probe_dir}/nsys/nsys_stop.log" 2>&1; then
    NSYS_COLLECTING=0
    echo "nsys report ready: ${NSYS_REPORT_LOCAL} ($(du -h "${NSYS_REPORT_LOCAL}" 2>/dev/null | cut -f1))"
  else
    echo "WARN: nsys stop failed or timed out; see ${probe_dir}/nsys/nsys_stop.log" >&2
  fi
fi

# The boot log lives in the sweep tree (start_server writes there); link it
# next to the run so bottleneck_report.py finds it without copying GBs.
newest_server_log="$(ls -t "${RESULTS_ROOT}/${POLICY}"/server_*.log 2>/dev/null | head -1 || true)"
if [[ -n "${newest_server_log}" ]]; then
  ln -sf "${newest_server_log}" "${run_dir}/server_$(basename "${newest_server_log}")"
fi
# The dynamo stack instead writes engine/frontend/nats logs under
# ${RESULTS_ROOT}/logs/ (start_dynamo_stack.sh); link the engine boot log
# the same way (it is the one with the scheduler/boot evidence).
if [[ "${ARM}" == "dynamo" ]]; then
  for component in engine frontend; do
    newest_log="$(ls -t "${RESULTS_ROOT}"/logs/${component}_*.log 2>/dev/null | head -1 || true)"
    [[ -n "${newest_log}" ]] && \
      ln -sf "${newest_log}" "${run_dir}/${component}_$(basename "${newest_log}")"
  done
fi

wait "${RUN_PID}"

# Copy the node-local nsys report into the run dir now that the benchmark is
# over — the NFS copy runs outside any measurement window. If the cell failed
# above, the report stays node-local at the path the nsys echo already logged.
if [[ -n "${NSYS_REPORT_LOCAL:-}" && -f "${NSYS_REPORT_LOCAL}" ]]; then
  if cp -v "${NSYS_REPORT_LOCAL}" "${probe_dir}/nsys/"; then
    echo "nsys report archived: ${probe_dir}/nsys/$(basename "${NSYS_REPORT_LOCAL}")"
  else
    echo "WARN: nsys report copy failed; node-local copy kept at ${NSYS_REPORT_LOCAL}" >&2
  fi
fi
echo "=== PROFILING CELL DONE: ${run_dir} ==="
