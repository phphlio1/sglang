#!/usr/bin/env bash
# E2 server start helper, 8x H200 / GLM-5.3 W4AFP8 (USER-EDITABLE).
#
# Serving config = the production GLM-5.3 engine config VERBATIM (13-10-2026
# Config GLM5.3 Prod, image sglang-runtime:1.5.0, engine glm53-w4afp8-dynamo-
# engine), on this branch's flag spelling. Mentor directive 2026-10-13: keep
# the engine config exactly like prod so the bottleneck profile measures the
# deployed configuration. Spellings that differ in text but not in resolved
# config (branch renames; dp_size is the deprecated alias, server_args.py:
# attn-dp-size > 1 implies enable-dp-attention):
#
#   production: --tp=8 --dp=8 --enable-dp-attention --ep=8
#   here:       --tp-size 8 --attn-dp-size 8 --ep-size 8
#
# Production port 8889 moves to 30000 (the harness's SGLANG_URL).
# --dsa-decode-backend flashmla_kv is spelled out (production left decode to
# auto-detect; flashmla_kv is the required Hopper pairing of the production
# prefill choice flashmla_sparse_q8 on this branch, dsa_backend.py).
# --schedule-policy comes from the experiment's $1 (prod runs dfs-weight;
# dfs-weight is one of the five arms).
#
# Divergence from production (unavoidable or by design):
#   - no dynamo frontend (router-mode=kv) in front: stage-2 arm; the engine's
#     TokenizerManager dispatches across the 8 attention-DP ranks round-robin
#   - --model-path is the node-local /mnt copy of the same weights (this
#     account cannot download models; HF_HUB_OFFLINE=1 enforces local-only)
#   - deployment-env necessities absent from the prod ENV list: CUDA_HOME +
#     NVCC_PREPEND_FLAGS (driver/toolchain), NVSHMEM_HCA_LIST (deepep NIC),
#     SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=600 (worker-death fix,
#     verified by smoke 2925)
#
# Usage / env knobs: identical to start_server.sh (<policy> | --stop;
# BOOST_*, IN_BATCH_PREFIX_CACHING_MAX_PIN_S, LOG_SCHEDULE_EVAL_PATH,
# GLM_MODEL_PATH).
# Log: results/8x-h200/glm53/agentx/<policy>/server_<timestamp>.log

set -euo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${E2_DIR}/results/8x-h200/glm53/agentx"

# ===== User-editable serving configuration (E2 H200, set 2026-10-08) =======
# Production engine config (han/glm53-w4afp8-h200-8x), flag-mapped as above.
# POLICY is passed as $1 so SERVER_FLAGS excludes it.
SERVER_PYTHON="${SERVER_PYTHON:-/shared/loipp/venv-server/bin/python3}"

# Team-placed checkpoint, flat dir (see header); env-overridable.
GLM_MODEL_PATH="${GLM_MODEL_PATH:-/mnt/hf_hub_cache/GLM-5.3-W4AFP8}"
SERVED_MODEL_NAME="zai-org/GLM-5.3"
PORT=30000
HOST="0.0.0.0"

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
  --tp-size 8
  --attn-dp-size 8
  --ep-size 8
  --quantization w4afp8
  --enable-dp-attention-local-control-broadcast
  --enable-dp-lm-head
  --dist-init-addr 127.0.0.1:10888
  --numa-node 0 0 0 0 1 1 1 1
  --moe-a2a-backend deepep
  --chunked-prefill-size 32768
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.75
  --max-running-requests 256
  --context-length 300000
  --kv-cache-dtype fp8_e4m3
  --dsa-prefill-backend flashmla_sparse_q8
  --dsa-decode-backend flashmla_kv
  --allow-auto-truncate
  --enable-metrics                    # Prometheus /metrics (E2 queue-time histogram)
  --enable-metrics-for-all-schedulers
  # usage.prompt_tokens_details.cached_tokens per request: populates AIPerf's
  # usage_prompt_cache_read_tokens.
  --enable-cache-report
  --enable-hierarchical-cache
  --hicache-size 128
  --hicache-io-backend direct
  --hicache-write-policy write_back
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --enable-prefill-delayer
  # Verbatim prod: KV event publisher feeding the dynamo KV router
  # (stage-2 arm subscribes to this endpoint). Flag and value are separate
  # array elements: one quoted "--flag value" element would expand to a
  # single argv token and argparse rejects it (3023 boot failure).
  --kv-events-config
  "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:8900\"}"
  --served-model-name "${SERVED_MODEL_NAME}"
)
# ==========================================================================

if [[ "${1:-}" == "--stop" ]]; then
  pkill -f "sglang.launch_server" 2>/dev/null || true
  echo "Server stop signal sent."
  exit 0
fi

POLICY="${1:?usage: start_server_h200_glm53.sh <policy> | --stop}"
case "$POLICY" in
  fcfs|fcfs-match|lpm|hrrn|dfs-weight|lof|boost) ;;
  *) echo "ERROR: policy must be one of: fcfs fcfs-match lpm hrrn dfs-weight lof boost (got '$POLICY')" >&2; exit 2 ;;
esac

# Policy-specific flags (E2 additions only; the production config above is
# untouched).
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

# Production engine env (k8s config), carried over verbatim. expandable_segments
# + gatherv + dp-attention metrics are dp-attention relevant; NCCL_P2P_LEVEL=NVL
# matches the single-node NVLink fabric; offline enforces local weights.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_DP_USE_GATHERV=1
export SGLANG_ENABLE_METRICS_DP_ATTENTION=1
# Verbatim prod ENV.
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
export NCCL_P2P_LEVEL=NVL
export HF_HUB_OFFLINE=1
# deep_gemm JIT-compiles FP8 GEMM kernels at runtime and invokes nvcc,
# requiring >= 12.9 (deep_jit assertion, see smoke 2688). No CUDA_HOME was
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
# deep_ep's low-latency buffer force-enables NVSHMEM_IB_ENABLE_IBGDA (deep_ep
# buffers/legacy.py), and IBGDA init fails on this node's HCAs, so NVSHMEM
# falls back to the ibrc transport, which pins host memory. Slurm's default
# soft memlock (~1 GB) is too small for 8 ranks' QPs/teams: srq creation
# returns NULL and Buffer init segfaults (smoke 2750). The hard limit is
# unlimited, so raise the soft limit before exec — propagates to every
# scheduler subprocess.
ulimit -l unlimited
# NVSHMEM scans ALL 16 RoCE ports on this node by default; 14 of them sit on
# odd devices (vendor_part 5522, fw 1.74) whose SRQ/QP creation fails, which
# killed transport init at every boot (smokes 2750/2754: ibrc srq NULL ->
# segfault in deep_ep Buffer init). The one real ConnectX (rocep41s0f0,
# vendor_part 4125, fw 22.43.2566, port ACTIVE) creates SRQ/QP fine
# (validated with an ibv_srq_pingpong pair). deep_ep overwrites its own
# NVSHMEM_* envs at Buffer init but never touches the HCA list, so this
# restriction survives. deep_ep reads it at import, before its own setup.
export NVSHMEM_HCA_LIST="rocep41s0f0"
# uvicorn's multiprocess supervisor SIGKILLs any worker that fails to answer
# its pipe healthcheck within the timeout while still importing the app
# (smokes 2759/2794/2837: a worker needs minutes to import the sglang stack;
# the default 10 s kills every worker in a ~10 s death/respawn loop before
# it can ever log anything -- SIGKILL leaves no traceback, which is why the
# earlier boots looked silent). 600 s covers worker spawn + import + app
# lifespan; once serving, the pong thread answers instantly so the timeout
# is never approached again.
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=600

echo "Starting SGLang server  model=${GLM_MODEL_PATH}  served=${SERVED_MODEL_NAME}  policy=${POLICY}  port=${PORT}  tp=8 attn-dp=8 ep=8"
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
  --model-path "${GLM_MODEL_PATH}" \
  --host "${HOST}" --port "${PORT}" \
  --schedule-policy "${POLICY}" \
  "${POLICY_FLAGS[@]}" \
  "${SERVER_FLAGS[@]}" \
  >"${LOG}" 2>&1
