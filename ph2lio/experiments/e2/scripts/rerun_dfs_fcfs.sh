#!/usr/bin/env bash
# Rerun dfs-weight and fcfs with current commit (60060211) and pin=0
# to match lpm/hrrn/boost config. Each cell ~54 min, 2 cells ~1.8 h.

set -uo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${E2_DIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${E2_DIR}/logs/rerun_dfs_fcfs_${STAMP}.log"
SEED=42
OK=0
FAIL=0

log() { printf '[%s] %s\n' "$(date +%F_%T)" "$*" | tee -a "${LOG}"; }

run_cell() {  # $1=policy
  local policy="$1"
  log "=== CELL START ${policy} seed=${SEED} c16 pin=0 ==="
  if SEED="${SEED}" CONCURRENCY=16 DURATION=1800 FORCE_RESTART=1 \
     GRACE_PERIOD=600 IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0 \
     bash "${E2_DIR}/scripts/run_e2.sh" policy "${policy}" 16 >> "${LOG}" 2>&1; then
    OK=$((OK + 1))
    log "=== CELL OK ${policy} ==="
  else
    FAIL=$((FAIL + 1))
    log "=== CELL FAIL ${policy} ==="
  fi
}

log "rerun dfs-weight + fcfs: commit=$(git rev-parse --short HEAD) seed=${SEED} c16 pin=0"

run_cell "dfs-weight"
run_cell "fcfs"

log "=== SWEEP DONE ok=${OK} fail=${FAIL} log=${LOG} ==="
