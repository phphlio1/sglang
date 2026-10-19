#!/usr/bin/env bash
# =============================================================================
# E2 unified experiment runner — single entrypoint
# =============================================================================
# AIPerf inferencex-agentx-mvp benchmark across SGLang schedule policies.
#
# Modes:
#   run_e2.sh smoke                     pipeline smoke test (NOT a valid benchmark)
#   run_e2.sh policy <policy> [conc]    one official (policy, concurrency) point
#   run_e2.sh sweep [conc...]           all 6 policies (fcfs fcfs-match lpm
#                                       hrrn dfs-weight boost), each at the
#                                       given concurrencies (default: $CONCURRENCY)
#   run_e2.sh array                     Slurm array mode: SLURM_ARRAY_TASK_ID
#                                       0..5 maps to the 6 policies
#
# Server lifecycle is managed here: the server is started via the
# user-maintained scripts/start_server.sh (all serving config lives there;
# policy is its $1). A running server is reused only when its FULL policy
# spec matches -- schedule_policy, and for boost also boost_gamma /
# boost_bin_size, plus the schedule-eval log path; anything else is
# restarted. A server this script started is stopped when the run finishes
# (STOP_SERVER=never to keep it, always to stop any).
#
# Per-request schedule-eval log: every policy-mode and smoke run asks the
# server for --log-schedule-eval-path (per policy+gamma+k, reset on server
# boot, appended across reused-server runs). The run directory gets a
# snapshot plus the line offset that isolates this run's records.
#
# Slurm: the SBATCH block below is inert under plain bash. Submit with e.g.
#   sbatch --array=0-5 scripts/run_e2.sh array     # one policy per array task
#   sbatch scripts/run_e2.sh policy fcfs           # single point
# Every knob is an env var, so --export=ALL,CONCURRENCY=8 also works. The
# benchmark client and the server share the allocated node/GPU (single-GPU
# experiment design; see docs/agentx_experiment.md).
#
#SBATCH --job-name=e2-agentx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/e2_%A_%a.log

set -euo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ===== User-editable variables (env-overridable) ============================
SGLANG_URL="${SGLANG_URL:-http://127.0.0.1:30000}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
CONCURRENCY="${CONCURRENCY:-4}"           # session-tree concurrency (default point)
DURATION="${DURATION:-1800}"              # s; scenario minimum is 900
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-${E2_DIR}/results/1x-rtx6000pro/agentx}"
SERVER_SCRIPT="${SERVER_SCRIPT:-${E2_DIR}/scripts/start_server.sh}"
METRICS_SCRIPT="${METRICS_SCRIPT:-${E2_DIR}/analysis/agentx_metrics.py}"
AIPERF="${AIPERF:-${E2_DIR}/.venv/bin/aiperf}"

SMOKE_POLICY="${SMOKE_POLICY:-dfs-weight}"  # server policy for smoke mode
SMOKE_DURATION="${SMOKE_DURATION:-300}"     # s; below scenario min -> --unsafe-override
NUM_ENTRIES="${NUM_ENTRIES:-16}"            # dataset subset, smoke only
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"    # s; model load + first-run JIT can be slow
STOP_SERVER="${STOP_SERVER:-auto}"          # auto | always | never
# Readiness gate for stacks whose /health lies: the dynamo frontend returns
# 200 on /health even with ZERO workers registered (measured; see
# smoke_dynamo_stack.sbatch) — the served model only appears on /v1/models
# after the engine finished booting and registered through discovery.
# READY_ON_MODEL=1 makes server_healthy also require "${MODEL}" on
# /v1/models (dynamo cells set this; the plain sglang HTTP server doesn't
# need it — its /health is honest).
READY_ON_MODEL="${READY_ON_MODEL:-0}"
# Boot fail-fast pgrep pattern (server_start): the engine-only server boots
# via sglang.launch_server; the dynamo stack boots via `python -m
# dynamo.sglang`. Same job, different process spelling — env-overridable so
# the dynamo cells don't die at the 60 s fail-fast check.
SERVER_PROC_PATTERN="${SERVER_PROC_PATTERN:-sglang\\.launch_server}"
# Extra Prometheus scrape endpoints appended to aiperf's --server-metrics
# (space-separated, e.g. "127.0.0.1:8081"). aiperf always also scrapes
# <SGLANG_URL>/metrics; under the dynamo frontend the engine's sglang:*
# multiprocess registry serves on DYN_SYSTEM_PORT (start_dynamo_stack.sh,
# confirmed by smoke 3307) instead, so dynamo cells set this to 8081.
SERVER_METRICS_URLS="${SERVER_METRICS_URLS:-}"
# Boost arm hyperparameters (larger gamma = more FCFS-like). BOOST_GAMMA empty
# = the server's clock-mode default (1e-4 tokens / 0.1 1/s wall).
BOOST_GAMMA="${BOOST_GAMMA:-}"
BOOST_BIN_SIZE="${BOOST_BIN_SIZE:-256}"
# gamma-Ada arm (Task 8): 1 = server adapts boost_gamma online from the
# admission wait tail (gamma0 = BOOST_GAMMA or the clock-mode default); arms
# log to a distinct eval file and run label so they never mix with
# static-gamma cells.
BOOST_ADAPTIVE_GAMMA="${BOOST_ADAPTIVE_GAMMA:-0}"
# Boost clock (wall-clock fix): wall (default) pins the effective gamma and
# the adaptive fixed point independent of cache state; tokens is the
# paper-exact counted-token clock (the pre-fix behavior).
BOOST_CLOCK="${BOOST_CLOCK:-wall}"
# In-batch dedup follower-pin hold bound (seconds), every cache-aware policy.
IN_BATCH_PREFIX_CACHING_MAX_PIN_S="${IN_BATCH_PREFIX_CACHING_MAX_PIN_S:-60}"
# Exported so start_server.sh sees the resolved values (not just inherited
# ones), keeping the two scripts' defaults from drifting apart.
export BOOST_GAMMA BOOST_BIN_SIZE BOOST_ADAPTIVE_GAMMA BOOST_CLOCK IN_BATCH_PREFIX_CACHING_MAX_PIN_S
# Exported so the run_meta python block records the resolved values.
export GRACE_PERIOD FORCE_RESTART
# Dataset subsetting for policy mode (Task-4 targeted regime): empty = the
# full dataset (official matrix points); N = first N entries in scan order
# (same subset for every policy/seed, so arms stay comparable).
NUM_DATASET_ENTRIES="${NUM_DATASET_ENTRIES:-}"
# Trajectory resume window (Task-4 targeted regime): empty = scenario default
# [0.0, 1.0]; set to restrict where sessions resume within each trajectory.
# Early turns carry seconds-scale think-times (late turns carry minutes-scale
# agent-work gaps), so a small window is the arrival-density lever at c8.
TRAJECTORY_START_MIN_RATIO="${TRAJECTORY_START_MIN_RATIO:-}"
TRAJECTORY_START_MAX_RATIO="${TRAJECTORY_START_MAX_RATIO:-}"
# Client drain after --benchmark-duration: stop issuing new requests, then
# wait up to this many seconds for in-flight requests to return (their
# responses count in metrics). inf = wait for all; fixes the in-flight-at-
# close cut (the 306.3 s / 152.6 s prefill giants lost at lpm/c16).
GRACE_PERIOD="${GRACE_PERIOD:-inf}"
# 1 = always stop and boot a fresh server, even when the running server's
# full policy spec matches: every cell starts from a cold radix cache (the
# c16-sweep regime). 0 keeps the reuse rule (a warm server carries earlier
# runs' cache state).
FORCE_RESTART="${FORCE_RESTART:-0}"
# GPU-side time series — the one signal the 3 Hz /metrics scrapes lack
# (util/power/mem per GPU). One nvidia-smi query per interval during the
# benchmark+drain window only, written to <run_dir>/gpu_samples.csv.
# 0 disables. Runs outside the serving path and identically for every
# policy, so the policy comparison stays clean.
GPU_SAMPLER_INTERVAL="${GPU_SAMPLER_INTERVAL:-5}"
# Space-separated dirs to pre-read into the node page cache before a server
# boot; empty = off. On the H200 arms the venv and the sglang tree live on
# NFS and the 8 dp replicas import them concurrently: the contending cold
# reads serialize on NFS bandwidth (measured on qwen cell 2: rank
# weight-load starts staggered ~9.2 min apart while each rank's own GPU work
# is only ~5 min, so boot is ~2 h), and each cell's ~0.5-1.2 TB of hicache
# host pools evicts the venv pages again, so every cell re-reads. One
# sequential cat (~12 GB) replaces 8 contending importers.
WARM_CACHE_PATHS="${WARM_CACHE_PATHS:-}"
# ============================================================================

AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800  # first run reconstructs the dataset cache
AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_CONFIGURATION_TIMEOUT AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT

SCENARIO="inferencex-agentx-mvp"
DATASET="semianalysis_cc_traces_weka_062126_256k"
POLICIES=(fcfs-match lpm hrrn dfs-weight boost)

log()  { printf '[e2 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '[e2 ERROR] %s\n' "$*" >&2; exit 2; }

valid_policy() {
  case "$1" in fcfs|fcfs-match|lpm|hrrn|dfs-weight|boost) return 0 ;; *) return 1 ;; esac
}

# --- Server lifecycle --------------------------------------------------------

server_healthy() {
  # --max-time is load-bearing: a half-dead frontend (SIGTERM'd, socket still
  # open, never answering) makes a bare curl block forever — cell 3437 hung
  # ~100 min in server_stop's poll on exactly that until it was scancelled.
  curl -sf --max-time 10 "${SGLANG_URL}/health" >/dev/null 2>&1 || return 1
  [[ "${READY_ON_MODEL}" == "1" ]] || return 0
  # File-first grep (never `curl | grep -q` under pipefail: grep -q closes
  # the pipe, curl dies SIGPIPE, pipefail fails a healthy poll — same
  # artifact as smoke 3307 step 5a).
  local m; m="$(mktemp)"
  # Tolerate both spellings: sglang HTTP emits "id": "x" (with space), the
  # dynamo frontend emits "id":"x" (no space; measured in smoke 3307).
  curl -sf --max-time 10 "${SGLANG_URL}/v1/models" -o "${m}" 2>/dev/null \
    && grep -qE "\"id\": *\"${MODEL}\"" "${m}"
  local rc=$?
  rm -f "${m}"
  return "${rc}"
}

server_info_json() {
  # /server_info is the current endpoint; /get_server_info still answers.
  curl -sf "${SGLANG_URL}/server_info" 2>/dev/null \
    || curl -sf "${SGLANG_URL}/get_server_info" 2>/dev/null
}

server_policy() {
  server_info_json | python3 -c 'import json,sys; print(json.load(sys.stdin)["schedule_policy"])'
}

# The schedule-eval log path a run of this policy expects: stable per
# (policy, clock, gamma, k) so a reused server keeps appending to one file,
# which the per-run line offset then slices. The clock is part of the name:
# a gamma value means different things under tokens vs wall.
requested_eval_path() {  # $1 = policy
  if [[ "$1" == "boost" && "${BOOST_ADAPTIVE_GAMMA}" == "1" ]]; then
    printf '%s/schedule_eval_ada_%s_k%s.jsonl' "${OUTPUT_DIR}/boost" "${BOOST_CLOCK}" "${BOOST_BIN_SIZE}"
  elif [[ "$1" == "boost" ]]; then
    printf '%s/schedule_eval_%s_g%s_k%s.jsonl' "${OUTPUT_DIR}/boost" "${BOOST_CLOCK}" "${BOOST_GAMMA:-def}" "${BOOST_BIN_SIZE}"
  else
    printf '%s/%s/schedule_eval.jsonl' "${OUTPUT_DIR}" "$1"
  fi
}

policy_params_match() {  # $1 = policy; 0 when the running server's full spec matches
  local info
  info="$(server_info_json)" || return 1
  python3 - "$1" "${BOOST_GAMMA}" "${BOOST_BIN_SIZE}" "${BOOST_CLOCK}" "$(requested_eval_path "$1")" "${info}" <<'PY'
import json, sys

policy, gamma_s, k_s, clock, want_path, info_s = sys.argv[1:7]
srv = json.loads(info_s)
ok = (
    srv.get("schedule_policy") == policy
    and (srv.get("log_schedule_eval_path") or "") == want_path
)
if policy == "boost":
    # Two boost runs with different (clock, gamma, k) must not silently reuse
    # the first server: the triple is the reuse key; an empty gamma arg means
    # the clock-mode default (1e-4 tokens / 0.1 1/s wall).
    default_gamma = 0.1 if clock == "wall" else 1e-4
    want_gamma = float(gamma_s) if gamma_s else default_gamma
    ok = (
        ok
        and srv.get("boost_gamma") is not None
        and float(srv["boost_gamma"]) == want_gamma
        and srv.get("boost_bin_size") is not None
        and int(srv["boost_bin_size"]) == int(k_s)
        and srv.get("boost_clock") == clock
    )
sys.exit(0 if ok else 1)
PY
}

schedule_eval_lines() {  # $1 = eval log path; line count, 0 when absent
  [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || printf '0'
}

server_stop() {
  bash "${SERVER_SCRIPT}" --stop >/dev/null 2>&1 || true
  local i
  for i in $(seq 1 30); do server_healthy || return 0; sleep 2; done
  die "server did not stop within 60 s"
}

SERVER_STARTED=0

warm_import_cache() {  # pre-read WARM_CACHE_PATHS into the page cache, once, sequentially
  [[ -n "${WARM_CACHE_PATHS}" ]] || return 0
  local d t0
  t0=$(date +%s)
  for d in ${WARM_CACHE_PATHS}; do
    [[ -d "${d}" ]] || { log "warm_import_cache: skipping absent dir ${d}"; continue; }
    find "${d}" -type f -exec cat {} + >/dev/null 2>&1 || true
  done
  log "import cache warmed (${WARM_CACHE_PATHS}; $(( $(date +%s) - t0 ))s)"
}

server_start() {  # $1 = policy
  mkdir -p "${OUTPUT_DIR}/$1"
  warm_import_cache
  LOG_SCHEDULE_EVAL_PATH="$(requested_eval_path "$1")" \
    nohup bash "${SERVER_SCRIPT}" "$1" >/dev/null 2>&1 &
  SERVER_STARTED=1
  log "server starting (policy=$1); waiting for health (timeout ${HEALTH_TIMEOUT}s)"
  local waited=0
  until server_healthy; do
    if (( waited >= HEALTH_TIMEOUT )); then
      boot_log_tail "$1"
      die "server not healthy after ${HEALTH_TIMEOUT}s (log tail above)"
    fi
    # Fail fast when the boot process died (argparse error, port bind,
    # crash): a dead server never turns healthy, so waiting the full
    # HEALTH_TIMEOUT only burns the job wall (3023 lost 2 h to this).
    # Grace: the wrapper needs a few seconds to spawn python.
    if (( waited >= 60 )) && ! pgrep -f "${SERVER_PROC_PATTERN}" >/dev/null; then
      boot_log_tail "$1"
      die "server process died during boot (log tail above)"
    fi
    sleep 5; waited=$((waited + 5))
  done
  log "server healthy: policy=$(server_policy || echo 'unknown (no /server_info — dynamo frontend?)')"
}

boot_log_tail() {  # $1 = policy; the start script writes boot logs under its
  # own results root (<root>/<policy>/), which equals OUTPUT_DIR in sweep
  # mode but is OUTPUT_DIR's parent in profiling mode (OUTPUT_DIR=<root>/profiling).
  local boot_log
  boot_log="$(ls -t ${OUTPUT_DIR}/$1/server_*.log 2>/dev/null | head -1)"
  [[ -z "${boot_log}" ]] && \
    boot_log="$(ls -t ${OUTPUT_DIR%/*}/$1/server_*.log 2>/dev/null | head -1)"
  # The dynamo stack writes engine_*.log under <results_root>/logs/ instead.
  [[ -z "${boot_log}" ]] && \
    boot_log="$(ls -t ${OUTPUT_DIR%/*}/logs/engine_*.log 2>/dev/null | head -1)"
  [[ -n "${boot_log}" ]] && tail -20 "${boot_log}" >&2 || true
}

ensure_server() {  # $1 = policy; sets SERVER_FRESH_BOOT (1 = this run booted)
  SERVER_FRESH_BOOT=1
  if server_healthy && [[ "${FORCE_RESTART}" != "1" ]]; then
    if policy_params_match "$1"; then
      log "reusing running server (full policy spec matches: $1)"
      SERVER_FRESH_BOOT=0
      return
    fi
    log "running server's policy spec does not match ${1}; restarting"
    server_stop
  elif server_healthy; then
    log "FORCE_RESTART=1: stopping matching server for a cold radix cache"
    server_stop
  fi
  server_start "$1"
}

finish_server() {
  if [[ "${STOP_SERVER}" == "always" ]] \
     || { [[ "${STOP_SERVER}" == "auto" ]] && (( SERVER_STARTED )); }; then
    server_stop; log "server stopped"
  else
    log "leaving server running (STOP_SERVER=${STOP_SERVER}, started_by_script=${SERVER_STARTED})"
  fi
}

# --- Benchmark ---------------------------------------------------------------

new_run_dir() {  # $1 = parent dir, $2 = label prefix; echoes unique stamped dir
  local dir stamp n
  for n in 0 1 2 3 4; do
    stamp="$(date +%Y%m%d_%H%M%S)"; [[ ${n} -gt 0 ]] && stamp="${stamp}_${n}"
    dir="$1/${2}_${stamp}"
    [[ -e "${dir}" ]] || { mkdir -p "${dir}"; printf '%s' "${dir}"; return; }
    sleep 1
  done
  die "could not allocate a fresh run dir under $1"
}

GPU_SAMPLER_PID=""

start_gpu_sampler() {  # $1 = run_dir; nvidia-smi time series -> $1/gpu_samples.csv
  [[ "${GPU_SAMPLER_INTERVAL}" =~ ^[0-9]+$ ]] || die "GPU_SAMPLER_INTERVAL must be a number"
  (( GPU_SAMPLER_INTERVAL > 0 )) || return 0
  command -v nvidia-smi >/dev/null 2>&1 || { log "gpu sampler: nvidia-smi not found; skipping"; return 0; }
  nohup nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm \
    --format=csv -l "${GPU_SAMPLER_INTERVAL}" > "$1/gpu_samples.csv" 2>/dev/null &
  GPU_SAMPLER_PID=$!
  # die() inside run_aiperf must not leak the sampler past script exit.
  trap stop_gpu_sampler EXIT
  log "gpu sampler started (interval ${GPU_SAMPLER_INTERVAL}s, pid ${GPU_SAMPLER_PID})"
}

stop_gpu_sampler() {
  [[ -n "${GPU_SAMPLER_PID}" ]] || return 0
  kill "${GPU_SAMPLER_PID}" 2>/dev/null || true
  wait "${GPU_SAMPLER_PID}" 2>/dev/null || true
  GPU_SAMPLER_PID=""
  trap - EXIT
  log "gpu sampler stopped"
}

run_aiperf() {  # $1 = run_dir, $2 = concurrency, $3 = duration, extra flags in "$@"
  local run_dir="$1" conc="$2" dur="$3"; shift 3
  start_gpu_sampler "${run_dir}"
  local cmd=(
    "${AIPERF}" profile
    --scenario "${SCENARIO}"
    --url "${SGLANG_URL}"
    --model-names "${MODEL}"
    --tokenizer "${TOKENIZER}"
    --public-dataset "${DATASET}"
    --concurrency "${conc}"
    --benchmark-duration "${dur}"
    --benchmark-grace-period "${GRACE_PERIOD}"
    --random-seed "${SEED}"
    --export-level records
    --use-server-token-count
    --streaming                # required with --per-chunk-usage
    --per-chunk-usage
    --server-metrics ${SERVER_METRICS_URLS}   # extra scrape endpoints (empty = bare enable)
    --server-metrics-formats jsonl
    --artifact-dir "${run_dir}"
    "$@"
  )
  printf '%s\n' "${cmd[@]}" > "${run_dir}/command.txt"
  log "running aiperf: concurrency=${conc} duration=${dur}s -> ${run_dir}"
  if "${cmd[@]}" 2>&1 | tee "${run_dir}/aiperf_console.log"; then
    log "aiperf exited 0"
  else
    local status=$?
    die "aiperf FAILED (exit ${status}); see ${run_dir}/aiperf_console.log"
  fi
  stop_gpu_sampler
}

post_run_diagnostics() {  # $1 = run_dir, $2 = context cap for the report
  python3 - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path

run_dir, cap = Path(sys.argv[1]), int(sys.argv[2])
records = run_dir / "profile_export.jsonl"
if not records.exists():
    print(f"[FAIL] missing {records}"); sys.exit(1)
n = errs = max_isl = max_total = 0
for line in records.open():
    rec = json.loads(line)
    n += 1
    errs += bool(rec.get("error"))
    m = rec.get("metrics", {})
    isl = (m.get("input_sequence_length") or {}).get("value", 0) or 0
    osl = (m.get("output_sequence_length") or {}).get("value", 0) or 0
    max_isl, max_total = max(max_isl, isl), max(max_total, isl + osl)
print(f"[run] records: {n}  errored: {errs}")
print(f"[run] max ISL: {max_isl}  max ISL+OSL: {max_total}  (dataset cap: {cap})")
missing = [p.name for p in (run_dir / "profile_export_aiperf.json",
                            run_dir / "server_metrics_export.jsonl") if not p.exists()]
if missing:
    print(f"[FAIL] missing exports: {missing}"); sys.exit(1)
print("[run] exports present: profile_export.jsonl, profile_export_aiperf.json, server_metrics_export.jsonl")
PY
}

write_run_meta() {  # $1 = run_dir, $2 = policy, $3 = concurrency,
                    # $4/$5/$6 = schedule-eval path / pre-run offset / records
                    # $7 = server_fresh_boot (1 = this run booted the server)
  local sglang_commit
  sglang_commit="$(git -C "${E2_DIR}/../../.." rev-parse HEAD 2>/dev/null || echo unknown)"
  # The dynamo frontend has no /server_info (that is the plain sglang HTTP
  # server's endpoint): both curls fail and the `||` chain would abort the
  # whole run under set -e AFTER aiperf already finished — write a stub so
  # run_meta.json still lands (policy_params_match already handles absence
  # by refusing reuse, which FORCE_RESTART cells bypass).
  curl -sf "${SGLANG_URL}/server_info" > "$1/server_info.json" 2>/dev/null \
    || curl -sf "${SGLANG_URL}/get_server_info" > "$1/server_info.json" 2>/dev/null \
    || printf '{"note": "no /server_info on this frontend (dynamo stack): schedule policy is recorded in run_meta.json and the engine log"}\n' > "$1/server_info.json"
  python3 - "$1/run_meta.json" "$2" "$3" "$MODEL" "$TOKENIZER" "$SGLANG_URL" \
      "$sglang_commit" "$(cat "$1/command.txt" | tr '\n' ' ')" \
      "$4" "$5" "$6" "$7" <<'PY'
import json, sys, time
from pathlib import Path

meta = {
    "policy": sys.argv[2],
    "concurrency": int(sys.argv[3]),
    "concurrency_semantics": "AIPerf session-tree concurrency; instantaneous HTTP request count can exceed it (subagent fan-out)",
    "duration_s": float(__import__("os").environ.get("DURATION", 1800)),
    "seed": int(__import__("os").environ.get("SEED", 42)),
    "model": sys.argv[4],
    "tokenizer": sys.argv[5],
    "sglang_url": sys.argv[6],
    "sglang_commit": sys.argv[7],
    "scenario": "inferencex-agentx-mvp",
    "dataset": "semianalysis_cc_traces_weka_062126_256k (semianalysisai/cc-traces-weka-062126-256k, 256K-capped)",
    "command": sys.argv[8],
    # Boost arm knobs (from env; ensure_server verified them against /server_info).
    # boost_gamma null = the clock-mode default (1e-4 tokens / 0.1 1/s wall).
    "boost_gamma": (
        float(__import__("os").environ["BOOST_GAMMA"])
        if __import__("os").environ.get("BOOST_GAMMA")
        else None
    ),
    "boost_bin_size": int(__import__("os").environ.get("BOOST_BIN_SIZE", "256")),
    # gamma-Ada arm: true = boost_gamma above is the initial gamma (gamma0)
    # and the server adapts it online from the admission wait tail.
    "boost_adaptive_gamma": __import__("os").environ.get("BOOST_ADAPTIVE_GAMMA", "0") == "1",
    # Boost clock: 'wall' (arrival in seconds, cache-state-pinned gamma) or
    # 'tokens' (paper-exact counted-token clock, the pre-fix behavior).
    "boost_clock": __import__("os").environ.get("BOOST_CLOCK", "wall"),
    # In-batch dedup follower-pin hold bound (seconds); 0 = unbounded (pre-fix).
    "in_batch_prefix_caching_max_pin_s": float(
        __import__("os").environ.get("IN_BATCH_PREFIX_CACHING_MAX_PIN_S", "60")
    ),
    # Dataset subsetting: null = full dataset (official matrix point); N = the
    # first N entries in scan order (targeted-regime runs; same subset across
    # arms since the mmap cache is keyed by seed only).
    "num_dataset_entries": (
        int(__import__("os").environ["NUM_DATASET_ENTRIES"])
        if __import__("os").environ.get("NUM_DATASET_ENTRIES")
        else None
    ),
    # Trajectory resume window (null = scenario default [0.0, 1.0]; the
    # targeted regime restricts it for arrival density -- see run_e2.sh).
    "trajectory_start_min_ratio": (
        float(__import__("os").environ["TRAJECTORY_START_MIN_RATIO"])
        if __import__("os").environ.get("TRAJECTORY_START_MIN_RATIO")
        else None
    ),
    "trajectory_start_max_ratio": (
        float(__import__("os").environ["TRAJECTORY_START_MAX_RATIO"])
        if __import__("os").environ.get("TRAJECTORY_START_MAX_RATIO")
        else None
    ),
    # Per-request schedule-eval log: slice the snapshot from offset_lines to
    # isolate this run's records (a reused server appends across runs).
    "schedule_eval": {
        "path": sys.argv[9],
        "offset_lines": int(sys.argv[10]),
        "records": int(sys.argv[11]),
    },
    # Server protocol: 1 = this run booted a fresh server (cold radix cache);
    # 0 = it reused a running server whose cache carries earlier runs.
    "server_fresh_boot": sys.argv[12] == "1",
    # Client drain: seconds waited for in-flight requests after duration
    # ("inf" = all); responses landing in the window count in metrics.
    "grace_period_s": __import__("os").environ.get("GRACE_PERIOD", "inf"),
    # 1 = ensure_server was forced to stop a spec-matching server so this
    # cell starts from a cold radix cache.
    "server_force_restart": __import__("os").environ.get("FORCE_RESTART", "0") == "1",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
Path(sys.argv[1]).write_text(json.dumps(meta, indent=2) + "\n")
PY
}

# --- Modes -------------------------------------------------------------------

mode_smoke() {
  echo "=== E2 SMOKE TEST (NOT A VALID AGENTX BENCHMARK) ==="
  local run_dir eval_path eval_offset
  run_dir="$(new_run_dir "${OUTPUT_DIR}/smoke" "smoke_c${CONCURRENCY}")"
  ensure_server "${SMOKE_POLICY}"
  eval_path="$(requested_eval_path "${SMOKE_POLICY}")"
  eval_offset=$(schedule_eval_lines "${eval_path}")
  local smoke_flags=(--num-dataset-entries "${NUM_ENTRIES}" --unsafe-override)
  [[ -n "${TRAJECTORY_START_MIN_RATIO}" ]] && smoke_flags+=(--trajectory-start-min-ratio "${TRAJECTORY_START_MIN_RATIO}")
  [[ -n "${TRAJECTORY_START_MAX_RATIO}" ]] && smoke_flags+=(--trajectory-start-max-ratio "${TRAJECTORY_START_MAX_RATIO}")
  run_aiperf "${run_dir}" "${CONCURRENCY}" "${SMOKE_DURATION}" "${smoke_flags[@]}"
  if [[ -f "${eval_path}" ]]; then
    cp "${eval_path}" "${run_dir}/schedule_eval.jsonl"
    log "schedule-eval: +$(( $(schedule_eval_lines "${eval_path}") - eval_offset )) records -> ${run_dir}/schedule_eval.jsonl"
  fi
  post_run_diagnostics "${run_dir}" 256000
  finish_server
  echo "=== SMOKE DONE: ${run_dir} (submission_valid=false; do NOT use for comparison) ==="
}

run_label() {  # $1 = policy, $2 = concurrency; boost arms carry clock/gamma/k
  if [[ "$1" == "boost" && "${BOOST_ADAPTIVE_GAMMA}" == "1" ]]; then
    printf 'c%s_gada_%s_k%s' "$2" "${BOOST_CLOCK}" "${BOOST_BIN_SIZE}"
  elif [[ "$1" == "boost" ]]; then
    printf 'c%s_g%s_%s_k%s' "$2" "${BOOST_GAMMA:-def}" "${BOOST_CLOCK}" "${BOOST_BIN_SIZE}"
  else
    printf 'c%s' "$2"
  fi
}

mode_policy() {  # $1 = policy, $2 = concurrency (optional)
  local policy="$1" conc="${2:-${CONCURRENCY}}"
  valid_policy "${policy}" || die "policy must be one of: ${POLICIES[*]} (got '${policy}')"
  echo "=== E2 policy run: policy=${policy} concurrency=${conc} duration=${DURATION}s seed=${SEED}${NUM_DATASET_ENTRIES:+ entries=${NUM_DATASET_ENTRIES}} ==="
  local run_dir
  run_dir="$(new_run_dir "${OUTPUT_DIR}/${policy}" "$(run_label "${policy}" "${conc}")")"
  ensure_server "${policy}"
  # Offset AFTER the server is up: a fresh boot resets the eval log, a reused
  # server carries earlier runs' records above this line count.
  local eval_path eval_offset eval_records
  eval_path="$(requested_eval_path "${policy}")"
  eval_offset=$(schedule_eval_lines "${eval_path}")
  local extra_flags=()
  [[ -n "${NUM_DATASET_ENTRIES}" ]] && extra_flags+=(--num-dataset-entries "${NUM_DATASET_ENTRIES}")
  [[ -n "${TRAJECTORY_START_MIN_RATIO}" ]] && extra_flags+=(--trajectory-start-min-ratio "${TRAJECTORY_START_MIN_RATIO}")
  [[ -n "${TRAJECTORY_START_MAX_RATIO}" ]] && extra_flags+=(--trajectory-start-max-ratio "${TRAJECTORY_START_MAX_RATIO}")
  run_aiperf "${run_dir}" "${conc}" "${DURATION}" "${extra_flags[@]}"
  eval_records=$(( $(schedule_eval_lines "${eval_path}") - eval_offset ))
  if [[ -f "${eval_path}" ]]; then
    cp "${eval_path}" "${run_dir}/schedule_eval.jsonl"
    log "schedule-eval: +${eval_records} records this run (offset ${eval_offset}) -> ${run_dir}/schedule_eval.jsonl"
  else
    eval_records=0
  fi
  write_run_meta "${run_dir}" "${policy}" "${conc}" "${eval_path}" "${eval_offset}" "${eval_records}" "${SERVER_FRESH_BOOT}"
  post_run_diagnostics "${run_dir}" 256000
  python3 "${METRICS_SCRIPT}" "${run_dir}"
  finish_server
  echo "=== RUN DONE: ${run_dir} ==="
}

mode_sweep() {  # remaining args = concurrency list (default: $CONCURRENCY)
  local concs=("$@"); (( ${#concs[@]} )) || concs=("${CONCURRENCY}")
  local p c
  for p in "${POLICIES[@]}"; do
    for c in "${concs[@]}"; do
      echo "=== E2 sweep point: policy=${p} concurrency=${c} ==="
      mode_policy "${p}" "${c}"
    done
  done
}

mode_array() {
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "array mode requires SLURM_ARRAY_TASK_ID"
  local idx="${SLURM_ARRAY_TASK_ID}"
  (( idx >= 0 && idx < ${#POLICIES[@]} )) || die "SLURM_ARRAY_TASK_ID=${idx} out of range 0..$((${#POLICIES[@]} - 1))"
  mode_policy "${POLICIES[idx]}"
}

mkdir -p "${E2_DIR}/logs"

case "${1:-}" in
  smoke)  shift; mode_smoke ;;
  policy) shift; mode_policy "$@" ;;
  sweep)  shift; mode_sweep "$@" ;;
  array)  shift; mode_array ;;
  *)
    cat <<EOF
Usage: $0 <mode> [args]
  smoke                    pipeline smoke test (NOT a valid benchmark)
  policy <policy> [conc]   one official point; policy in: ${POLICIES[*]}
  sweep [conc...]          all policies at the given concurrencies (default ${CONCURRENCY})
  array                    Slurm array mode (SLURM_ARRAY_TASK_ID 0..5 -> policies)
Env knobs: SGLANG_URL MODEL TOKENIZER CONCURRENCY DURATION SEED OUTPUT_DIR
           SMOKE_POLICY SMOKE_DURATION NUM_ENTRIES HEALTH_TIMEOUT STOP_SERVER
           BOOST_GAMMA BOOST_BIN_SIZE NUM_DATASET_ENTRIES GRACE_PERIOD FORCE_RESTART
           TRAJECTORY_START_MIN_RATIO TRAJECTORY_START_MAX_RATIO GPU_SAMPLER_INTERVAL
EOF
    exit 2 ;;
esac
