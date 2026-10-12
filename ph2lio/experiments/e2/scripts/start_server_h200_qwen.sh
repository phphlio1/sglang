#!/usr/bin/env bash
# E2 server start helper, 8x H200 / Qwen3.8-27B-FP8 (USER-EDITABLE).
#
# Serving config = the 1x-RTX6000Pro config (scripts/start_server.sh, set
# 2026-10-05) carried over VERBATIM per the user's direction: this node
# simulates a production multi-node deployment, so the per-node serving
# config is NOT re-tuned per the cookbook. The 8x H200 node runs it as 8
# single-GPU replicas behind one endpoint: plain --dp-size 8 (full
# replicas; DataParallelController dispatches round-robin). Each replica's
# per-GPU serving is exactly the RTX config, so the RTX per-GPU numbers
# (mamba clamp 33, cache behavior, ITL) carry over per replica.
#
# Deltas from start_server.sh (everything else verbatim):
#   - --dp-size 8 topology flag (env QWEN_DP_SIZE). NOT the deprecated
#     "--dp-size N --enable-dp-attention" spelling — that resolves to
#     attn_dp_size, not replicas.
#   - SERVER_PYTHON: /shared venv (compute nodes see /shared, not /home).
#   - Results root results/8x-h200/qwen/agentx.
#   - RTX Blackwell env workarounds dropped (CUDA shim, JIT DeepGEMM off,
#     flashinfer CCCL flag): H200 is SM90.
#
# Usage / env knobs: identical to start_server.sh (<policy> | --stop;
# BOOST_*, IN_BATCH_PREFIX_CACHING_MAX_PIN_S, LOG_SCHEDULE_EVAL_PATH).
#
# Weights: flat local-dir download at /mnt/hf_hub_cache/Qwen3.8-27B-FP8
# (~27 GB; download permission granted 2026-10-09 — download_qwen_h200.sbatch.
# The pre-existing cache-style entry models--Qwen--Qwen3.8-27B-FP8 is a
# 366 MB stub owned by another user, so we use our own flat dir instead).
# Served name pinned to the HF id so the harness MODEL is unchanged.
# Log: results/8x-h200/qwen/agentx/<policy>/server_<timestamp>.log

set -euo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${E2_DIR}/results/8x-h200/qwen/agentx"

# ===== User-editable serving configuration (E2 H200, set 2026-10-08) =======
# RTX config (2026-10-05) unchanged; only the topology and the interpreter
# path move. POLICY is passed as $1 so SERVER_FLAGS excludes it.
SERVER_PYTHON="${SERVER_PYTHON:-/shared/loipp/venv-server/bin/python3}"

# Flat local-dir weights (team /mnt convention); env-overridable.
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/mnt/hf_hub_cache/Qwen3.8-27B-FP8}"
# Client-facing name pinned to the HF id (model-path is a local dir).
SERVED_MODEL_NAME="Qwen/Qwen3.8-27B-FP8"
PORT=30000
HOST="0.0.0.0"
# 8 single-GPU replicas = 8 simulated nodes behind the engine's
# round-robin dispatch. No dp-attention, no external router (stage 1).
QWEN_DP_SIZE="${QWEN_DP_SIZE:-8}"

# Boost-policy hyperparameters (larger gamma = more FCFS-like). BOOST_GAMMA
# empty = the server's clock-mode default (1e-4 tokens / 0.1 1/s wall).
BOOST_GAMMA="${BOOST_GAMMA:-}"
BOOST_BIN_SIZE="${BOOST_BIN_SIZE:-256}"
# gamma-Ada (Task 8): 1 = adapt boost_gamma online from the admission wait
# tail (gamma0 = BOOST_GAMMA or the clock-mode default); off by default,
# static-gamma path untouched.
BOOST_ADAPTIVE_GAMMA="${BOOST_ADAPTIVE_GAMMA:-0}"
# Boost clock: wall (default) stamps arrival in seconds and pins the
# effective gamma / adaptive fixed point independent of cache state;
# tokens is the paper-exact counted-token clock.
BOOST_CLOCK="${BOOST_CLOCK:-wall}"
# In-batch dedup follower-pin hold bound (seconds), shared by every
# cache-aware policy; 0 disables (pre-fix unbounded behavior).
IN_BATCH_PREFIX_CACHING_MAX_PIN_S="${IN_BATCH_PREFIX_CACHING_MAX_PIN_S:-60}"
LOG_SCHEDULE_EVAL_PATH="${LOG_SCHEDULE_EVAL_PATH:-}"

SERVER_FLAGS=(
  --dp-size "${QWEN_DP_SIZE}"           # 8 replicas; per-replica flags below
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
  # usage_prompt_cache_read_tokens.
  --enable-cache-report
  --served-model-name "${SERVED_MODEL_NAME}"   # pin: model-path is local
)
# ==========================================================================

if [[ "${1:-}" == "--stop" ]]; then
  pkill -f "sglang.launch_server" 2>/dev/null || true
  echo "Server stop signal sent."
  exit 0
fi

POLICY="${1:?usage: start_server_h200_qwen.sh <policy> | --stop}"
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
  [[ "${BOOST_ADAPTIVE_GAMMA}" == "1" ]] && POLICY_FLAGS+=(--boost-adaptive-gamma)
fi
# Follower-pin hold bound: applies to every cache-aware policy.
POLICY_FLAGS+=(--in-batch-prefix-caching-max-pin-s "${IN_BATCH_PREFIX_CACHING_MAX_PIN_S}")
if [[ -n "${LOG_SCHEDULE_EVAL_PATH}" ]]; then
  POLICY_FLAGS+=(--log-schedule-eval-path "${LOG_SCHEDULE_EVAL_PATH}")
fi

mkdir -p "${RESULTS_ROOT}/${POLICY}"
LOG="${RESULTS_ROOT}/${POLICY}/server_$(date +%Y%m%d_%H%M%S).log"

# Local-weights-only: the checkpoint must already sit at QWEN_MODEL_PATH.
export HF_HUB_OFFLINE=1
# deep_gemm JIT-compiles FP8 GEMM kernels at runtime and invokes nvcc,
# requiring >= 12.9 (deep_jit assertion, see smoke 2692). No CUDA_HOME was
# set, so find_cuda_home() fell through to the node's /usr/local/cuda
# (older toolkit). Point it at the nvcc bundled with the cu13 torch wheels
# (CUDA 13.4) inside the server venv. The venv's bin/ must precede PATH
# too: the same JIT then invokes ninja (smoke 2746/2747, all ranks died at
# cuda-graph capture), and ninja only exists at venv-server/bin — the
# server is launched via the venv's python3 directly, not an activated venv.
export CUDA_HOME="${CUDA_HOME:-/shared/loipp/venv-server/lib/python3.11/site-packages/nvidia/cu13}"
export PATH="${CUDA_HOME}/bin:$(dirname "${SERVER_PYTHON}"):${PATH}"
# The pip cu13 dir mixes nvcc 13.4 (nvidia-cuda-nvcc) with CUDART 13.0
# headers (nvidia-cuda-runtime), and CCCL hard-errors on that mismatch.
# Opt out of the check for every JIT: nvcc >= 11.7 prepends
# NVCC_PREPEND_FLAGS to its own command line, which covers deep_jit's and
# flashinfer's ninja-spawned nvcc alike (same opt-out the RTX campaign
# used via FLASHINFER_EXTRA_CUDAFLAGS, kept as belt-and-suspenders for any
# path that builds flags itself).
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

echo "Starting SGLang server  model=${QWEN_MODEL_PATH}  served=${SERVED_MODEL_NAME}  policy=${POLICY}  port=${PORT}  dp(replicas)=${QWEN_DP_SIZE}"
if [[ "${POLICY}" == "boost" ]]; then
  echo "boost: gamma=${BOOST_GAMMA:-<clock default>}  bin_size=${BOOST_BIN_SIZE}  clock=${BOOST_CLOCK}  adaptive=${BOOST_ADAPTIVE_GAMMA}"
fi
echo "in-batch dedup pin bound: ${IN_BATCH_PREFIX_CACHING_MAX_PIN_S}s"
if [[ -n "${LOG_SCHEDULE_EVAL_PATH}" ]]; then
  echo "schedule-eval log: ${LOG_SCHEDULE_EVAL_PATH} (reset on boot)"
  rm -f "${LOG_SCHEDULE_EVAL_PATH}"
fi
echo "Log: ${LOG}"
exec "${SERVER_PYTHON}" -m sglang.launch_server \
  --model-path "${QWEN_MODEL_PATH}" \
  --host "${HOST}" --port "${PORT}" \
  --schedule-policy "${POLICY}" \
  "${POLICY_FLAGS[@]}" \
  "${SERVER_FLAGS[@]}" \
  >"${LOG}" 2>&1
