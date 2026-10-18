#!/usr/bin/env bash
# GPQA-Diamond accuracy A/B for the upstream boost PR (mirrors #32911):
# boost (TOKENS default, default gamma, non-adaptive) vs fcfs, same model +
# serving config, same eval seed. Scheduling changes prefill admission order
# (batch composition) but not the model; accuracy should match within sampling
# noise on 198 questions.  Requires HF access to the gated Idavidrein/gpqa
# dataset (hf auth login once before running).
#
# Uses the e2 start_server.sh pinned config (chunked prefill, EAGLE spec,
# hicache) so the only variable is --schedule-policy. pin=0 matches the PR.

set -euo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SG_EVAL=/home/user1111/miniconda3/envs/sglang/bin/sgl-eval
OUT_ROOT="${E2_DIR}/results/1x-rtx6000pro/agentx/gpqa_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"
PORT=30000
BASE="http://127.0.0.1:${PORT}/v1"
MODEL="Qwen/Qwen3.8-27B-FP8"

log() { printf '[%s] %s\n' "$(date +%F_%T)" "$*"; }

run_arm() {  # $1 = label, $2 = policy, $3 = boost env exports
  local label="$1" policy="$2" boost_env="$3"
  local out="${OUT_ROOT}/${label}"
  mkdir -p "${out}"
  log "=== ${label}: booting --schedule-policy ${policy} ==="
  env IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0 ${boost_env} \
      bash "${E2_DIR}/scripts/start_server.sh" "${policy}" >"${out}/server.log" 2>&1 &
  local srv_pid=$!
  # wait for health (model load can take a few minutes)
  for i in $(seq 1 120); do
    if curl -s -m 3 "${BASE}/models" >/dev/null 2>&1; then break; fi
    sleep 5
  done
  curl -s -m 5 "${BASE}/models" >/dev/null || { log "FAIL: server ${label} did not come up"; kill ${srv_pid} 2>/dev/null || true; return 1; }
  log "=== ${label}: server up; running GPQA-Diamond ==="
  "${SG_EVAL}" run gpqa \
      --base-url "${BASE}" --model "${MODEL}" \
      --num-threads 8 --seed 0 --out-dir "${out}" >"${out}/sgl_eval.log" 2>&1 || true
  log "=== ${label}: eval done; stopping server ==="
  bash "${E2_DIR}/scripts/start_server.sh" --stop >"${out}/stop.log" 2>&1 || true
  wait ${srv_pid} 2>/dev/null || true
  # extract accuracy (sgl-eval may print to stdout or write a JSON in out-dir)
  local acc
  acc=$(grep -rhoE '"accuracy"[: ]+[0-9.]+' "${out}" 2>/dev/null | head -1 | grep -oE '[0-9.]+' | tail -1 || echo "?")
  printf '%s\t%s\n' "${label}" "${acc}" >> "${OUT_ROOT}/summary.tsv"
  log "=== ${label}: accuracy=${acc} ==="
}

run_arm "boost_tokens" "boost" "BOOST_CLOCK=tokens BOOST_ADAPTIVE_GAMMA=0 BOOST_GAMMA="
run_arm "fcfs"        "fcfs" ""

log "=== A/B SUMMARY ==="
cat "${OUT_ROOT}/summary.tsv"
log "=== DONE (results under ${OUT_ROOT}) ==="
