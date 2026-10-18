#!/usr/bin/env bash
# Gamma sweep for ada-boost tuning (seed 42 = contention-heavy, c16, pin=0).
# Purpose: find the gamma sweet spot across the paper's full range (0.09-15).
#
# Key insight: the paper's gamma range 0.09-15 is in 1/s (wall mode). In tokens
# mode, work is in tokens (256+), so gamma=0.09 gives gamma*w=23 -> boost=0
# (pure FCFS). Tokens mode needs gamma ~ 1e-4 to 1e-2. We sweep wall mode
# across the full paper range, plus a few tokens-mode points at correct scale.
#
# Adaptive gamma: the 2026-10-13 run showed gamma converging to ~0.042, but
# J=60 floors gamma at ~0.077, preventing convergence. We test J=120 to let
# the adaptive gamma reach its natural fixed point.
#
# Each cell: ~45 min.  12 cells ~ 9 h.  Run overnight.

set -uo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${E2_DIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${E2_DIR}/logs/gamma_sweep_${STAMP}.log"
SEED=42
OK=0
FAIL=0

log() { printf '[%s] %s\n' "$(date +%F_%T)" "$*" | tee -a "${LOG}"; }

run_cell() {  # $1=label $2=gamma $3=clock $4=adaptive $5=max_jump_s [$6=bin_size]
  local label="$1" gamma="$2" clock="$3" adaptive="$4" jump="$5" k="${6:-256}"
  log "=== CELL START ${label} seed=${SEED} c16 pin=0 k=${k} ==="
  if SEED="${SEED}" CONCURRENCY=16 DURATION=1800 FORCE_RESTART=1 \
     GRACE_PERIOD=600 IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0 \
     BOOST_GAMMA="${gamma}" BOOST_BIN_SIZE="${k}" BOOST_CLOCK="${clock}" \
     BOOST_ADAPTIVE_GAMMA="${adaptive}" BOOST_ADAPTIVE_MAX_JUMP_S="${jump}" \
     bash "${E2_DIR}/scripts/run_e2.sh" policy boost 16 >> "${LOG}" 2>&1; then
    OK=$((OK + 1))
    log "=== CELL OK ${label} ==="
  else
    FAIL=$((FAIL + 1))
    log "=== CELL FAIL ${label} ==="
  fi
}

log "gamma sweep start: seed=${SEED} c16 pin=0 12 cells"

# Phase A: WALL static across the paper's full range (0.09-15)
# gamma=0.09: strong boost (~50s jump for 256-token req at 2k tok/s)
# gamma=15.0: near-FCFS (~0.1s jump)
run_cell "wall_g0.09"   "0.09"  "wall" "0" "60"
run_cell "wall_g0.3"    "0.3"   "wall" "0" "60"
run_cell "wall_g1.0"    "1.0"   "wall" "0" "60"
run_cell "wall_g3.0"    "3.0"   "wall" "0" "60"
run_cell "wall_g5.0"    "5.0"   "wall" "0" "60"
run_cell "wall_g15.0"   "15.0"  "wall" "0" "60"

# Phase B: WALL adaptive with larger J to let gamma reach its fixed point (~0.042)
# J=60 floors gamma at ~0.077; J=120 floors at ~0.054, letting gamma converge.
run_cell "wall_ada_J60_g0.09"  "0.09" "wall" "1" "60"
run_cell "wall_ada_J120_g0.09" "0.09" "wall" "1" "120"

# Phase C: Smaller bin_size for finer-grained prioritization
# k=256 lumps all 1-256-token requests into one bin (same boost = FCFS among them).
# k=64 gives 4x finer granularity for short-prefill workloads.
run_cell "wall_g0.09_k64"  "0.09" "wall" "0" "60" "64"

# Phase D: Tokens mode at correct scale (gamma ~ 1e-4 to 1e-2)
run_cell "tokens_g3e-4"  "0.0003" "tokens" "0" "60"
run_cell "tokens_g1e-3"  "0.001"  "tokens" "0" "60"

log "=== SWEEP DONE ok=${OK} fail=${FAIL} log=${LOG} ==="
