#!/usr/bin/env bash
# E2 smoke — dynamo FULL STACK (frontend router-mode=kv + engine in-process),
# the first dynamo boot on this branch. Self-contained on purpose: run_e2.sh
# wiring (SGLANG_URL=8888, metrics scrape 8081) comes AFTER this passes —
# boot under dynamo has its own failure surface (NATS registration, frontend
# worker discovery, sgl.Engine boot inside the dynamo wrapper, shim
# ConfigArgumentMerger) that must be diagnosed from the raw stack logs, not
# through the harness.
#
# Verifies, in order:
#   1. nats-server up + frontend listening :8888 (registration path alive)
#   2. /health on the FRONTEND port goes green (engine worker discovered +
#      registered its endpoints — dynamo routes through NATS, so health here
#      proves the whole discovery chain, not just one process)
#   3. GET /v1/models serves the prod served-model name
#   4. one tiny /v1/chat/completions round-trip: non-stream AND streaming
#      (SSE) — the aiperf client runs --streaming, and the dynamo frontend's
#      SSE path (router kv -> engine worker, chunked transfer) was never
#      exercised by earlier smokes
#   5. engine worker's sglang:* metrics live on :8081 (DYN_SYSTEM_PORT)
#      + probes :8888/metrics (whether the Rust frontend proxies/exposes
#      anything — decides the harness scrape URL later)
#   5c. scheduler processes answer to the sglang::scheduler pgrep pattern
#      AND a sibling py-spy can attach (SGLANG_ALLOW_PTRACE=1 makes the
#      schedulers prctl(PR_SET_PTRACER_ANY) — branch scheduler.py; without
#      it every flame capture is denied under yama ptrace_scope=1, which is
#      how profile cell 3065 lost all its flames)
#
# Submit AFTER mirror_from_login.sh hgx-h200-01 printed MIRROR OK — without
# the /mnt mirror the boot reads the venv over ~1.4 MB/s NFS (hours).
#SBATCH --job-name=e2-dynamo-smoke
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# Same cpus/mem reasoning as smoke_glm53_h200.sbatch: >96 CPUs forces Slurm
# to span both NUMA nodes for the --numa-node 0 0 0 0 1 1 1 1 binding; GLM's
# 8 ranks measure ~1.17 TB RSS (hicache pools) so --mem=1600G.
#SBATCH --cpus-per-task=160
#SBATCH --mem=1600G
#SBATCH --time=04:30:00
#SBATCH --output=/shared/loipp/logs/e2_dynamo_smoke_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"
RESULTS_ROOT="${E2_DIR}/results/8x-h200/glm53/agentx_dynamo"
LOGS_DIR="${RESULTS_ROOT}/logs"

NODE_LOCAL="${NODE_LOCAL:-/mnt/loipp}"
if [ ! -x "${NODE_LOCAL}/venv-dynamo/bin/python3" ]; then
  echo "=== FATAL: node-local mirror absent on $(hostname) ==="
  echo "=== run mirror_from_login.sh hgx-h200-01 first (needs MIRROR OK) ==="
  exit 1
fi
if [ ! -d /mnt/hf_hub_cache/GLM-5.3-W4AFP8 ]; then
  echo "=== FATAL: GLM weights not on this node's /mnt/hf_hub_cache ==="
  exit 1
fi

FRONTEND_URL="http://127.0.0.1:8888"
METRICS_URL="http://127.0.0.1:8081"

# --- 0. hicache_hash_cpp JIT pre-compile (fail fast, clear log) -----------
# The first real request triggers this JIT build inside scheduler processes
# (hierarchical-cache hashing); the node has no libssl-dev, so the headers
# come from the mirror + CPATH/LIBRARY_PATH (see start_dynamo_stack.sh).
# Pre-compiling here makes a header/toolchain problem surface in ~30s with
# this job's log, instead of 5 min into the boot inside a scheduler proc.
echo "=== 0. pre-compiling hicache_hash_cpp (openssl from mirror) ==="
HOME="${NODE_LOCAL}/home" \
CPATH="${NODE_LOCAL}/openssl/include" \
LIBRARY_PATH="${NODE_LOCAL}/openssl/lib" \
UV_PYTHON_INSTALL_DIR="${NODE_LOCAL}/uv-python" \
PATH="${NODE_LOCAL}/venv-dynamo/bin:${PATH}" \
"${NODE_LOCAL}/venv-dynamo/bin/python3" -c \
  "import sglang.srt.mem_cache.cpp_utils.native_hash as nh; nh._load_native_hash_module(); print('hicache_hash_cpp ready')" \
  || { echo "=== FAIL: hicache_hash_cpp pre-compile (openssl dev / ninja) ==="; exit 1; }

mkdir -p "${LOGS_DIR}"
# Temp dir for curl bodies: never `curl | grep -m N`/`curl | head -c N` under
# `set -o pipefail` — the downstream tool exits after N matches/bytes, closes
# the pipe, curl dies with SIGPIPE (exit 23) and pipefail fails the WHOLE
# pipeline even though the request succeeded (smoke 3307 step 5a "failed" this
# way while its log shows the sglang:* metrics it had already received).
SMOKE_TMP="$(mktemp -d)"
trap 'bash "${E2_DIR}/scripts/start_dynamo_stack.sh" --stop || true; rm -rf "${SMOKE_TMP}"' EXIT

echo "=== dynamo stack smoke start $(date) on $(hostname) ==="
# Schedulers call prctl(PR_SET_PTRACER_ANY) when this is 1 (branch
# scheduler.py) — validated by step 5c's py-spy attach below.
export SGLANG_ALLOW_PTRACE=1
bash "${E2_DIR}/scripts/start_dynamo_stack.sh" dfs-weight &
STACK_PID=$!
trap 'bash "${E2_DIR}/scripts/start_dynamo_stack.sh" --stop || true' EXIT

# --- 1. frontend port listening (engine still booting at this point) --------
echo "=== 1. waiting for frontend :8888 to listen (NATS up, registration alive) ==="
FRONTEND_TIMEOUT="${FRONTEND_TIMEOUT:-600}"
until curl -s -o /dev/null --max-time 5 "${FRONTEND_URL}/" 2>/dev/null; do
  FRONTEND_TIMEOUT=$((FRONTEND_TIMEOUT - 5))
  if [ "${FRONTEND_TIMEOUT}" -le 0 ]; then
    echo "=== FAIL: frontend never listened on :8888 ==="
    tail -n 60 "${LOGS_DIR}"/nats.log 2>/dev/null || true
    tail -n 60 "${LOGS_DIR}"/frontend_*.log 2>/dev/null | tail -n 120 || true
    exit 1
  fi
  sleep 5
done
echo "frontend listening after $((180 - FRONTEND_TIMEOUT))s"
tail -n 20 "${LOGS_DIR}"/frontend_*.log 2>/dev/null | tail -n 25 || true

# --- 2. frontend /health green (HTTP service up; login-node measurement
# showed this can lag stack start by ~1-2 min — the 8 --dyn-preprocess-workers
# import the sglang tokenizer stack before the HTTP service starts) ----------
# NOTE: /health returns 200 EVEN WITH NO WORKERS REGISTERED (measured on the
# login node: {"status":"healthy","endpoints":[],"instances":[]}) — it only
# proves the frontend's HTTP face, NOT the discovery chain.
echo "=== 2. waiting for /health on ${FRONTEND_URL} (frontend HTTP service) ==="
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
T0=$(date +%s)
until curl -sf -o /dev/null --max-time 5 "${FRONTEND_URL}/health" 2>/dev/null; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - T0))
  if [ "${ELAPSED}" -ge "${HEALTH_TIMEOUT}" ]; then
    echo "=== FAIL: /health not green after ${ELAPSED}s ==="
    for f in "${LOGS_DIR}"/nats.log "${LOGS_DIR}"/frontend_*.log "${LOGS_DIR}"/engine_*.log; do
      echo "--- tail ${f} ---"
      tail -n 80 "${f}" 2>/dev/null || true
    done
    exit 1
  fi
  sleep 10
done
echo "=== /health GREEN after $(( $(date +%s) - T0 ))s ==="
curl -s --max-time 10 "${FRONTEND_URL}/health" -o "${SMOKE_TMP}/health.json"
head -c 500 "${SMOKE_TMP}/health.json"; echo

# --- 2b. THE REAL BOOT WAIT: the served model appearing on /v1/models. -----
# dynamo.sglang registers the model only AFTER sgl.Engine finishes booting
# (init_llm.py: engine = sgl.Engine(...) at :78 precedes
# register_model_with_readiness_gate at :170) — so "zai-org/GLM-5.3" in the
# models list proves the whole chain: engine boot -> registration through
# the file discovery KV -> frontend router table. Same cold-boot envelope
# as smoke_glm53_h200.sbatch (GLM boot measured ~2.5-3 h cold; the mirror
# keeps imports local but the py3.12 venv's HOME JIT caches start cold).
echo "=== 2b. waiting for zai-org/GLM-5.3 on /v1/models (engine boot + registration) ==="
MODELS_TIMEOUT="${MODELS_TIMEOUT:-12000}"
T0=$(date +%s)
until curl -sf --max-time 10 "${FRONTEND_URL}/v1/models" -o "${SMOKE_TMP}/models.json" 2>/dev/null \
    && grep -q 'zai-org/GLM-5.3' "${SMOKE_TMP}/models.json"; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - T0))
  if ! kill -0 "${STACK_PID}" 2>/dev/null; then
    echo "=== FAIL: stack driver process died (engine exit) at ${ELAPSED}s ==="
    for f in "${LOGS_DIR}"/nats.log "${LOGS_DIR}"/frontend_*.log "${LOGS_DIR}"/engine_*.log; do
      echo "--- tail ${f} ---"
      tail -n 80 "${f}" 2>/dev/null || true
    done
    exit 1
  fi
  if [ "${ELAPSED}" -ge "${MODELS_TIMEOUT}" ]; then
    echo "=== FAIL: model never registered after ${ELAPSED}s ==="
    for f in "${LOGS_DIR}"/nats.log "${LOGS_DIR}"/frontend_*.log "${LOGS_DIR}"/engine_*.log; do
      echo "--- tail ${f} ---"
      tail -n 80 "${f}" 2>/dev/null || true
    done
    exit 1
  fi
  if [ $((ELAPSED % 300)) -eq 0 ] && [ "${ELAPSED}" -gt 0 ]; then
    echo "[${ELAPSED}s] still booting; engine log tail:"
    tail -n 5 "${LOGS_DIR}"/engine_*.log 2>/dev/null | tail -n 8 || true
  fi
  sleep 10
done
echo "=== MODEL REGISTERED after $(( $(date +%s) - T0 ))s ==="

# --- 3. /v1/models -----------------------------------------------------------
echo "=== 3. GET /v1/models ==="
curl -sf --max-time 30 "${FRONTEND_URL}/v1/models" -o "${SMOKE_TMP}/models_now.json" \
  || { echo "=== FAIL: /v1/models errored ==="; exit 1; }
head -c 2000 "${SMOKE_TMP}/models_now.json"; echo

# --- 4. one chat completion --------------------------------------------------
# (curl-body-to-file pattern — see SMOKE_TMP note above the stack start)
echo "=== 4. POST /v1/chat/completions (tiny, non-stream) ==="
curl -sf --max-time 300 "${FRONTEND_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/GLM-5.3","messages":[{"role":"user","content":"Trả lời đúng một từ: thủ đô Việt Nam là gì?"}],"max_tokens":64,"temperature":0}' \
  -o "${SMOKE_TMP}/chat.json" || { echo "=== FAIL: chat completion errored ==="; exit 1; }
head -c 4000 "${SMOKE_TMP}/chat.json"; echo

# --- 4b. streaming chat (SSE) — the aiperf client path (--streaming) --------
echo "=== 4b. POST /v1/chat/completions (tiny, STREAMING/SSE) ==="
curl -sfN --max-time 300 "${FRONTEND_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/GLM-5.3","messages":[{"role":"user","content":"Đếm từ 1 đến 5, mỗi số một dòng."}],"max_tokens":64,"temperature":0,"stream":true}' \
  -o "${SMOKE_TMP}/chat_stream.txt" || { echo "=== FAIL: streaming chat errored ==="; exit 1; }
grep -m 3 '^data:' "${SMOKE_TMP}/chat_stream.txt" || {
  echo "=== FAIL: no SSE data events in the stream ==="; exit 1; }
grep -q '\[DONE\]' "${SMOKE_TMP}/chat_stream.txt" || {
  echo "=== FAIL: stream never terminated with [DONE] (client would hang) ==="; exit 1; }
echo "stream OK ($(grep -c '^data:' "${SMOKE_TMP}/chat_stream.txt") SSE events, [DONE] present)"

# --- 5. metrics --------------------------------------------------------------
echo "=== 5a. engine sglang:* metrics on :8081 (DYN_SYSTEM_PORT) ==="
curl -sf --max-time 30 "${METRICS_URL}/metrics" -o "${SMOKE_TMP}/metrics.txt" \
  || { echo "=== FAIL: ${METRICS_URL}/metrics errored ==="; exit 1; }
grep -m 8 'sglang:' "${SMOKE_TMP}/metrics.txt" \
  || { echo "=== FAIL: no sglang:* metrics on ${METRICS_URL}/metrics ==="; exit 1; }
echo "=== 5b. frontend :8888/metrics probe (harness scrape-URL decision) ==="
curl -s --max-time 30 -o /dev/null -w 'http_code=%{http_code}\n' "${FRONTEND_URL}/metrics"
curl -s --max-time 30 "${FRONTEND_URL}/metrics" -o "${SMOKE_TMP}/metrics_8888.txt" 2>/dev/null
grep -m 5 'sglang:' "${SMOKE_TMP}/metrics_8888.txt" || \
  echo "(no sglang:* on 8888 — harness must scrape :8081)"
# dynamo's own frontend/router metrics live here (TTFT/ITL, kv_hit_rate,
# routing overhead, per-worker gauges). aiperf auto-scrapes <url>/metrics,
# so the profiling cell's export gets these for free — assert they exist.
grep -m 5 'dynamo_frontend_' "${SMOKE_TMP}/metrics_8888.txt" \
  || { echo "=== FAIL: no dynamo_frontend_* metrics on ${FRONTEND_URL}/metrics ==="; exit 1; }

# --- 5c. schedulers + py-spy attach (profiling-cell prerequisites) -----------
# SGLANG_ALLOW_PTRACE is exported above so the schedulers (branch
# scheduler.py: run_scheduler_process) call prctl(PR_SET_PTRACER_ANY) at
# spawn; this proves the profiling cell's flame capture will work.
echo "=== 5c. sglang::scheduler procs + py-spy attach ==="
SCHED_PIDS="$(pgrep -f 'sglang::scheduler' || true)"
echo "scheduler pids: ${SCHED_PIDS:-none}"
[[ -n "${SCHED_PIDS}" ]] || {
  echo "=== FAIL: no sglang::scheduler processes (py-spy target pattern wrong) ==="; exit 1; }
FIRST_SCHED="$(echo "${SCHED_PIDS}" | head -1)"
PYSPY="${NODE_LOCAL}/venv-server/bin/py-spy"
if [ -x "${PYSPY}" ]; then
  if "${PYSPY}" dump --pid "${FIRST_SCHED}" >"${SMOKE_TMP}/pyspy_dump.txt" 2>&1; then
    echo "py-spy attach OK on scheduler pid ${FIRST_SCHED}:"
    head -n 6 "${SMOKE_TMP}/pyspy_dump.txt"
  else
    echo "=== FAIL: py-spy attach denied (SGLANG_ALLOW_PTRACE not effective?) ==="
    head -n 4 "${SMOKE_TMP}/pyspy_dump.txt"
    exit 1
  fi
else
  echo "(py-spy not on the mirror; skipping the attach test)"
fi

echo "=== DYNAMO SMOKE PASS $(date) ==="
echo "stack logs: ${LOGS_DIR}/"
exit 0
