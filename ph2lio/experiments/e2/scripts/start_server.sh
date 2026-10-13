#!/usr/bin/env bash
# E2 server start helper (USER-EDITABLE).
#
# The server-side SGLang configuration is maintained manually by the user.
# This script is only a convenience wrapper: edit MODEL / flags below to
# match your intended serving config. The benchmark scripts never hard-code
# model parameters; they read the running server's config via
# /get_server_info and record it into each run directory.
#
# Usage:
#   scripts/start_server.sh <policy>     # policy in: fcfs fcfs-match lpm hrrn dfs-weight lof boost
#   scripts/start_server.sh --stop
#
# Boost-policy knobs (env): BOOST_GAMMA (default: unset -> clock-mode default,
# 1e-4 tokens / 0.1 1/s wall), BOOST_BIN_SIZE (default 256), BOOST_CLOCK
# (default wall; the fixed wall-clock boost), BOOST_ADAPTIVE_GAMMA (gamma-Ada,
# requires wall clock) + BOOST_ADAPTIVE_MAX_JUMP_S (its gamma floor's jump bound).
# IN_BATCH_PREFIX_CACHING_MAX_PIN_S (default 60) bounds the in-batch dedup
# follower pin for every cache-aware policy. LOG_SCHEDULE_EVAL_PATH (absolute
# .jsonl, default off) enables the per-request schedule-eval log; the file is
# reset on every server boot.
#
# Log: results/1x-rtx6000pro/agentx/<policy>/server_<timestamp>.log

set -euo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${E2_DIR}/results/1x-rtx6000pro/agentx"

# ===== User-editable serving configuration (E2, set 2026-10-05) =============
# User-specified config; POLICY is passed as $1 so SERVER_FLAGS excludes it.
# Model path uses the HF id (local /models/Qwen3.8-27B-FP8 does not exist;
# the checkpoint is in the HF cache). --enable-metrics is added by us:
# E2's queue-time histogram scrapes Prometheus /metrics, which requires it.
CONDA_PREFIX="/home/user1111/miniconda3/envs/sglang"
CONDA_PYTHON="${CONDA_PREFIX}/bin/python3"
CUDA_HOME="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cu13"
CUDA_SHIM="/home/user1111/sglang/.cuda_shim"

MODEL="Qwen/Qwen3.8-27B-FP8"
PORT=30000
HOST="0.0.0.0"

# Boost-policy hyperparameters (larger gamma = more FCFS-like). BOOST_GAMMA
# empty = the server's clock-mode default (1e-4 tokens / 0.1 1/s wall).
BOOST_GAMMA="${BOOST_GAMMA:-}"
BOOST_BIN_SIZE="${BOOST_BIN_SIZE:-256}"
# gamma-Ada (Task 8): 1 = adapt boost_gamma online from the completed-request
# response-time tail (UniBoost Eq. 9; gamma0 = BOOST_GAMMA or the clock-mode
# default); off by default, static-gamma path untouched.
BOOST_ADAPTIVE_GAMMA="${BOOST_ADAPTIVE_GAMMA:-0}"
# gamma-Ada max-jump bound (wall seconds): the gamma floor keeps
# b_gamma(bin_size / token_rate) within this.
BOOST_ADAPTIVE_MAX_JUMP_S="${BOOST_ADAPTIVE_MAX_JUMP_S:-60}"
# Boost clock: wall (default) stamps arrival in seconds and pins the
# effective gamma / adaptive fixed point independent of cache state;
# tokens is the paper-exact counted-token clock.
BOOST_CLOCK="${BOOST_CLOCK:-wall}"
# In-batch dedup follower-pin hold bound (seconds), shared by every
# cache-aware policy; 0 disables (pre-fix unbounded behavior).
IN_BATCH_PREFIX_CACHING_MAX_PIN_S="${IN_BATCH_PREFIX_CACHING_MAX_PIN_S:-60}"
LOG_SCHEDULE_EVAL_PATH="${LOG_SCHEDULE_EVAL_PATH:-}"

SERVER_FLAGS=(
  --trust-remote-code
  --kv-cache-dtype fp8_e4m3
  --mem-fraction-static 0.80
  --attention-backend flashinfer
  --chunked-prefill-size 8192
  --max-prefill-tokens 16384
  --context-length 262144
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --enable-linear-replayssm-spec
  --mamba-full-memory-ratio 1.5
  --enable-hierarchical-cache
  --hicache-size 64
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --enable-metrics                    # Prometheus /metrics (E2 queue-time histogram)
  # usage.prompt_tokens_details.cached_tokens per request: populates AIPerf's
  # usage_prompt_cache_read_tokens. Added 2026-10-06 AFTER the seed-42 matrix;
  # those 11 runs have no client-side cache metrics (eval-log classification
  # covers them) and are not config-identical to runs after this change.
  --enable-cache-report
)
# ==========================================================================

if [[ "${1:-}" == "--stop" ]]; then
  pkill -f "sglang.launch_server" 2>/dev/null || true
  echo "Server stop signal sent."
  exit 0
fi

POLICY="${1:?usage: start_server.sh <policy> | --stop}"
case "$POLICY" in
  fcfs|fcfs-match|lpm|hrrn|dfs-weight|lof|boost) ;;
  *) echo "ERROR: policy must be one of: fcfs fcfs-match lpm hrrn dfs-weight lof boost (got '$POLICY')" >&2; exit 2 ;;
esac

# Policy-specific flags (E2 additions only; the pinned serving config above
# is untouched).
POLICY_FLAGS=()
if [[ "${POLICY}" == "boost" ]]; then
  [[ -n "${BOOST_GAMMA}" ]] && POLICY_FLAGS+=(--boost-gamma "${BOOST_GAMMA}")
  POLICY_FLAGS+=(--boost-bin-size "${BOOST_BIN_SIZE}" --boost-clock "${BOOST_CLOCK}")
  [[ "${BOOST_ADAPTIVE_GAMMA}" == "1" ]] && POLICY_FLAGS+=(--boost-adaptive-gamma --boost-adaptive-max-jump-s "${BOOST_ADAPTIVE_MAX_JUMP_S}")
fi
# Follower-pin hold bound: applies to every cache-aware policy.
POLICY_FLAGS+=(--in-batch-prefix-caching-max-pin-s "${IN_BATCH_PREFIX_CACHING_MAX_PIN_S}")
if [[ -n "${LOG_SCHEDULE_EVAL_PATH}" ]]; then
  POLICY_FLAGS+=(--log-schedule-eval-path "${LOG_SCHEDULE_EVAL_PATH}")
fi

mkdir -p "${RESULTS_ROOT}/${POLICY}"
LOG="${RESULTS_ROOT}/${POLICY}/server_$(date +%Y%m%d_%H%M%S).log"

export SGLANG_ENABLE_JIT_DEEPGEMM=0    # Blackwell sm_120: JIT DeepGEMM off
# nvcc 13.4 (nvidia/cu13) vs required >=12.9 is fine, but system nvcc 12.4
# must not win PATH.
export CUDA_HOME="${CUDA_HOME}"
export LIBRARY_PATH="${CUDA_SHIM}:${CUDA_HOME}/lib"
export LD_LIBRARY_PATH="${CUDA_SHIM}:${CUDA_HOME}/lib"
export PATH="${CUDA_HOME}/bin:${CONDA_PREFIX}/bin:${PATH}"
# nvcc 13.4 vs the cu13 pip headers' CUDART 13.0: flashinfer's vendored CCCL
# rejects the minor-version mismatch (same opt-out as the sglang JIT one).
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

echo "Starting SGLang server  model=${MODEL}  policy=${POLICY}  port=${PORT}"
if [[ "${POLICY}" == "boost" ]]; then
  echo "boost: gamma=${BOOST_GAMMA:-<clock default>}  bin_size=${BOOST_BIN_SIZE}  clock=${BOOST_CLOCK}  adaptive=${BOOST_ADAPTIVE_GAMMA}"
fi
echo "in-batch dedup pin bound: ${IN_BATCH_PREFIX_CACHING_MAX_PIN_S}s"
if [[ -n "${LOG_SCHEDULE_EVAL_PATH}" ]]; then
  echo "schedule-eval log: ${LOG_SCHEDULE_EVAL_PATH} (reset on boot)"
  rm -f "${LOG_SCHEDULE_EVAL_PATH}"
fi
echo "Log: ${LOG}"
exec "${CONDA_PYTHON}" -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" --port "${PORT}" \
  --schedule-policy "${POLICY}" \
  "${POLICY_FLAGS[@]}" \
  "${SERVER_FLAGS[@]}" \
  >"${LOG}" 2>&1
