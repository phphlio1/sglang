#!/usr/bin/env bash
# E2 dynamo full-stack start (bootstrap): nats-server + dynamo.frontend
# (router-mode=kv, PROD frontend config verbatim) + engine worker under
# `python -m dynamo.sglang` (PROD engine config verbatim, our branch's sglang
# in-process via sgl.Engine).
#
# Bootstrap note: dynamo 1.4.2 from the team-placed nvcr vllm-runtime image —
# NOT the prod image (sglang-runtime:1.5.0-kvoverlap1, not on the node; team
# registry credentials needed). When that image lands, swap the extracted
# packages in /shared/loipp/dynamo-pkg and rebuild venv-dynamo; this script's
# flags do not change.
#
# Ports: frontend HTTP 8888 (prod port; aiperf target = SGLANG_URL), NATS
# 4222 (dynamo default), engine system metrics DYN_SYSTEM_PORT=8081 (sglang:*
# multiprocess Prometheus — the harness's metrics scrape URL), engine
# dist-init 127.0.0.1:10888 + kv-events ZMQ 8900+dp_rank (prod values).
#
# Usage like start_server_h200_glm53.sh: <policy> | --stop
# Env: POLICY flags (BOOST_*...) identical; DYNAMO_VENV overrides venv-dynamo.
set -euo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${E2_DIR}/results/glm53/agentx_dynamo"

BENCH=/shared/loipp
# Node-local execution mirror (mirror_from_login.sh): when present, run the
# whole stack from /mnt — NFS ~1.4 MB/s makes a bare-/shared boot take hours
# (same lesson as venv-server; see mirror_node_side.sh header).
NODE_LOCAL="${NODE_LOCAL:-/mnt/loipp}"
if [ -x "${NODE_LOCAL}/venv-dynamo/bin/python3" ]; then
  DYNAMO_VENV="${DYNAMO_VENV:-${NODE_LOCAL}/venv-dynamo}"
  BENCH_MIRROR="${NODE_LOCAL}"
else
  DYNAMO_VENV="${DYNAMO_VENV:-${BENCH}/venv-dynamo}"
  BENCH_MIRROR="${BENCH}"
  echo "=== node-local mirror absent on $(hostname); running from /shared (slow NFS) ===" >&2
fi
PY="${DYNAMO_VENV}/bin/python3"
NATS="${BENCH_MIRROR}/bin/nats-server"
FRONTEND_PORT=8888
DYN_SYSTEM_PORT=8081

GLM_MODEL_PATH="${GLM_MODEL_PATH:-/mnt/hf_hub_cache/GLM-5.3-W4AFP8}"
SERVED_MODEL_NAME="zai-org/GLM-5.3"
PORT="${PORT:-${FRONTEND_PORT}}"
HOST="0.0.0.0"

# Boost-policy hyperparameters — identical to start_server_h200_glm53.sh.
BOOST_GAMMA="${BOOST_GAMMA:-}"
BOOST_BIN_SIZE="${BOOST_BIN_SIZE:-256}"
BOOST_ADAPTIVE_GAMMA="${BOOST_ADAPTIVE_GAMMA:-0}"
BOOST_ADAPTIVE_MAX_JUMP_S="${BOOST_ADAPTIVE_MAX_JUMP_S:-60}"
BOOST_CLOCK="${BOOST_CLOCK:-wall}"
IN_BATCH_PREFIX_CACHING_MAX_PIN_S="${IN_BATCH_PREFIX_CACHING_MAX_PIN_S:-60}"
LOG_SCHEDULE_EVAL_PATH="${LOG_SCHEDULE_EVAL_PATH:-}"

mkdir -p "${RESULTS_ROOT}/logs"
NATS_LOG="${RESULTS_ROOT}/logs/nats.log"
FRONTEND_LOG="${RESULTS_ROOT}/logs/frontend_$(date +%Y%m%d_%H%M%S).log"
ENGINE_LOG="${RESULTS_ROOT}/logs/engine_$(date +%Y%m%d_%H%M%S).log"

if [[ "${1:-}" == "--stop" ]]; then
  pkill -f "dynamo.sglang" 2>/dev/null || true
  pkill -f "dynamo.frontend" 2>/dev/null || true
  pkill -f "${NATS}" 2>/dev/null || true
  echo "Dynamo stack stop signal sent."
  exit 0
fi

POLICY="${1:?usage: start_dynamo_stack.sh <policy> | --stop}"
case "$POLICY" in
  fcfs|fcfs-match|lpm|hrrn|dfs-weight|lof|shortest-prefill-first|boost) ;;
  *) echo "ERROR: unknown policy '$POLICY'" >&2; exit 2 ;;
esac

POLICY_FLAGS=()
if [[ "${POLICY}" == "boost" ]]; then
  [[ -n "${BOOST_GAMMA}" ]] && POLICY_FLAGS+=(--boost-gamma="${BOOST_GAMMA}")
  POLICY_FLAGS+=(--boost-bin-size="${BOOST_BIN_SIZE}" --boost-clock="${BOOST_CLOCK}")
  [[ "${BOOST_ADAPTIVE_GAMMA}" == "1" ]] && POLICY_FLAGS+=(--boost-adaptive-gamma --boost-adaptive-max-jump-s "${BOOST_ADAPTIVE_MAX_JUMP_S}")
fi
POLICY_FLAGS+=(--in-batch-prefix-caching-max-pin-s="${IN_BATCH_PREFIX_CACHING_MAX_PIN_S}")
[[ -n "${LOG_SCHEDULE_EVAL_PATH}" ]] && POLICY_FLAGS+=(--log-schedule-eval-path="${LOG_SCHEDULE_EVAL_PATH}")

# Prod ENV block (k8s), verbatim — same as start_server_h200_glm53.sh.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_DP_USE_GATHERV=1
export SGLANG_ENABLE_METRICS_DP_ATTENTION=1
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
# OOM recovery (scheduler.py prepare_for_extend guard) returns requests to the
# queue instead of crashing; the strict idle mem-check invariant would still
# kill the engine after a recovery eviction, so it must be off (same as
# start_server.sh:119-123 in the boost tree).
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export NCCL_P2P_LEVEL=NVL
export HF_HUB_OFFLINE=1
# Compute nodes have no /home/loipp (login-local): torch's dynamo/inductor
# cache bootstrap makedirs under $HOME at import time and dies with EACCES
# (smoke 2603) — point HOME at the mirror home (compile caches live there).
export HOME="${BENCH_MIRROR}/home"
mkdir -p "${HOME}"
# Deployment-env necessities (see start_server_h200_glm53.sh for the full
# why): CUDA_HOME at the venv's cu13 wheels (venv-dynamo carries its own),
# nvcc CCCL opt-out, NVSHMEM HCA restriction, memlock for deepep QPs,
# uvicorn healthcheck N/A here (no uvicorn — dynamo frontend is the HTTP face).
export CUDA_HOME="${CUDA_HOME:-${DYNAMO_VENV}/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="${CUDA_HOME}/bin:$(dirname "${PY}"):${PATH}"
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export NVSHMEM_HCA_LIST="rocep41s0f0"
ulimit -l unlimited
# OpenSSL dev for sglang's hicache_hash_cpp JIT extension (openssl/sha.h +
# -lcrypto): compute nodes have no libssl-dev. The extension compiles ON the
# node once (~10s) and caches under HOME with the node's own /mnt source
# paths — login-built caches get rejected by ninja (deps-log paths differ).
export CPATH="${BENCH_MIRROR}/openssl/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${BENCH_MIRROR}/openssl/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
# Engine worker's sglang:* multiprocess Prometheus server (publisher.py).
export DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT}"

# Dynamo discovery topology (bootstrap divergence from prod k8s, which runs
# etcd): bare-metal single node uses the filesystem KV store instead. Both
# frontend and engine read these env vars (runtime_args.py env_var wiring):
#   DYN_DISCOVERY_BACKEND=file  (default etcd -> frontend died in smoke 3167
#                                after the 120s etcd retry loop)
#   DYN_REQUEST_PLANE=tcp       (default; frontend<->worker direct TCP)
#   DYN_EVENT_PLANE=zmq         (default; kv events for router-mode=kv)
# The nats-server below stays up but is OFF the default path — only needed
# if request/event plane is switched to nats later.
export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_FILE_KV="${DYN_FILE_KV:-/tmp/dynamo_store_kv}"
# Fresh discovery state per stack start: stale worker registrations from a
# previous run would poison the frontend's router table.
rm -rf "${DYN_FILE_KV}"
mkdir -p "${DYN_FILE_KV}"

echo "=== dynamo stack: nats + frontend(:${FRONTEND_PORT}) + engine(policy=${POLICY}) ==="
echo "logs: ${NATS_LOG} / ${FRONTEND_LOG} / ${ENGINE_LOG}"

# 1. NATS — dynamo's discovery/event plane backend (static go binary).
"${NATS}" --addr 127.0.0.1 --port 4222 >"${NATS_LOG}" 2>&1 &
NATS_PID=$!
sleep 1
kill -0 "${NATS_PID}" 2>/dev/null || { cat "${NATS_LOG}" >&2; echo "ERROR: nats-server died" >&2; exit 1; }

# 2. Frontend — prod frontend config verbatim (13-10-2026).
"${PY}" -m dynamo.frontend \
  --http-host="${HOST}" \
  --http-port="${FRONTEND_PORT}" \
  --router-mode=kv \
  --router-prefill-load-scale=16 \
  --router-min-initial-workers=1 \
  --dyn-chat-processor=sglang \
  --dyn-preprocess-workers=8 \
  --tool-call-parser=glm47 \
  --reasoning-parser=glm45 \
  >"${FRONTEND_LOG}" 2>&1 &
FRONTEND_PID=$!

# 3. Engine worker — prod engine config verbatim under the dynamo wrapper.
#    Flag spelling follows prod (= form); the branch resolves --dp=8 to the
#    attn-dp alias exactly like start_server_h200_glm53.sh documents.
#    Divergence vs prod CLI: --page-size=64 explicit (prod leaves auto).
#    Same resolved value — FlashMLA forces exactly 64 (overrides.py) — but
#    dynamo's publisher reads server_args.page_size directly off the
#    UNRESOLVED input record (our resolution lives in an overlay, the input
#    field stays None), so the auto default crashes KvEventPublisher
#    (smoke 3190). Explicit input = dynamo reads 64.
# Optional NSys daemon-mode wrap (NSYS_PROFILE=1, set by profile_cell.sh):
# `nsys launch` starts the app under injection in a named session; collection
# stays OFF until `nsys start --session=<name>` from the cell script, so boot
# (weight load, JIT, CUDA-graph capture) runs unprofiled and only the serving
# window is collected. Default off = byte-identical engine command.
#   --session-new : creates the session (plain --session= requires an
#                   already-listed session; nsys launch --help)
#   -t cuda,nvtx  : CUDA API+activity (memcpy/kernel/graph-exec) + NVTX only;
#                   no osrt/CPU sampling (py-spy covers python CPU)
#   --cuda-graph-trace=graph : whole-graph exec records. SGLang captures its
#                   decode AND breakable-prefill graphs during BOOT, before
#                   collection starts, so node granularity cannot instrument
#                   them (and adds heavy per-node overhead per the help).
#                   NOT --trace-fork-before-exec: its own help warns of
#                   undefined behavior (boot-crash risk) and the mp-spawned
#                   schedulers are covered by the default process-tree scope.
if [[ "${NSYS_PROFILE:-0}" == "1" ]]; then
  NSYS_SESSION_NAME="${NSYS_SESSION_NAME:-e2nsys-$(date +%Y%m%d_%H%M%S)}"
  export NSYS_SESSION_NAME
  NSYS_LAUNCH=(nsys launch --session-new="${NSYS_SESSION_NAME}" \
    -t cuda,nvtx --cuda-graph-trace=graph)
else
  NSYS_LAUNCH=()
fi

"${NSYS_LAUNCH[@]}" "${PY}" -m dynamo.sglang \
  --model-path="${GLM_MODEL_PATH}" \
  --served-model-name="${SERVED_MODEL_NAME}" \
  --quantization=w4afp8 \
  --tp=8 --dp=8 --ep=8 \
  --enable-dp-attention \
  --enable-dp-attention-local-control-broadcast \
  --enable-dp-lm-head \
  --dist-init-addr=127.0.0.1:10888 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --moe-a2a-backend=deepep \
  --chunked-prefill-size=32768 \
  --tool-call-parser=glm47 \
  --reasoning-parser=glm45 \
  --mem-fraction-static=0.75 \
  --max-running-requests=256 \
  --context-length=300000 \
  --kv-cache-dtype=fp8_e4m3 \
  --dsa-prefill-backend=flashmla_sparse_q8 \
  --allow-auto-truncate \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --enable-cache-report \
  --enable-hierarchical-cache \
  --hicache-size=128 \
  --hicache-io-backend=direct \
  --hicache-write-policy=write_back \
  --speculative-algorithm=EAGLE \
  --speculative-num-steps=3 \
  --speculative-eagle-topk=1 \
  --speculative-num-draft-tokens=4 \
  --schedule-policy="${POLICY}" \
  --enable-prefill-delayer \
  --page-size=64 \
  --kv-events-config='{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:8900"}' \
  "${POLICY_FLAGS[@]}" \
  >"${ENGINE_LOG}" 2>&1 &
ENGINE_PID=$!

echo "pids: nats=${NATS_PID} frontend=${FRONTEND_PID} engine=${ENGINE_PID}"
echo "engine metrics (sglang:*): http://127.0.0.1:${DYN_SYSTEM_PORT}/metrics"
echo "frontend API (SGLANG_URL): http://127.0.0.1:${FRONTEND_PORT}"

# Foreground-wait on the engine; if anything dies, take the stack down.
wait "${ENGINE_PID}" || RC=$?
kill "${FRONTEND_PID}" "${NATS_PID}" 2>/dev/null || true
exit "${RC:-0}"
