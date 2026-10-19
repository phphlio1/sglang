#!/usr/bin/env bash
# Focused gamma sweep around the sweet spot (gamma=5.0 from the broad sweep).
# Broad sweep showed: g3.0=good, g5.0=BEST, g15.0=good but slightly worse.
# Narrow down to find the exact optimum between 3.0 and 15.0.
#
# Each cell: ~54 min.  8 cells ~ 7.2 h.

set -uo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${E2_DIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${E2_DIR}/logs/gamma_focused_${STAMP}.log"
SEED=42
OK=0
FAIL=0

log() { printf '[%s] %s\n' "$(date +%F_%T)" "$*" | tee -a "${LOG}"; }

run_cell() {  # $1=label $2=gamma
  local label="$1" gamma="$2"
  log "=== CELL START ${label} seed=${SEED} c16 pin=0 k=256 ==="
  if SEED="${SEED}" CONCURRENCY=16 DURATION=1800 FORCE_RESTART=1 \
     GRACE_PERIOD=600 IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0 \
     BOOST_GAMMA="${gamma}" BOOST_BIN_SIZE="256" BOOST_CLOCK="wall" \
     BOOST_ADAPTIVE_GAMMA="0" BOOST_ADAPTIVE_MAX_JUMP_S="60" \
     bash "${E2_DIR}/scripts/run_e2.sh" policy boost 16 >> "${LOG}" 2>&1; then
    OK=$((OK + 1))
    log "=== CELL OK ${label} ==="
  else
    FAIL=$((FAIL + 1))
    log "=== CELL FAIL ${label} ==="
  fi
}

log "focused gamma sweep start: seed=${SEED} c16 pin=0 8 cells around sweet spot g=5.0"

# Fine sweep between 3.0 and 15.0 (broad sweep: g3.0, g5.0, g15.0 done)
run_cell "wall_g4.0"  "4.0"
run_cell "wall_g6.0"  "6.0"
run_cell "wall_g7.0"  "7.0"
run_cell "wall_g8.0"  "8.0"
run_cell "wall_g10.0" "10.0"
run_cell "wall_g12.0" "12.0"

# Wider brackets to confirm the trend
run_cell "wall_g2.0"  "2.0"
run_cell "wall_g20.0" "20.0"

# Adaptive gamma with size-aware fix (verify convergence to ~5.0, not ~0.08)
run_cell_ada() {
  local label="$1" max_jump="$2"
  log "=== CELL START ${label} seed=${SEED} c16 pin=0 k=256 ada ==="
  if SEED="${SEED}" CONCURRENCY=16 DURATION=1800 FORCE_RESTART=1 \
     GRACE_PERIOD=600 IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0 \
     BOOST_GAMMA="0.09" BOOST_BIN_SIZE="256" BOOST_CLOCK="wall" \
     BOOST_ADAPTIVE_GAMMA="1" BOOST_ADAPTIVE_MAX_JUMP_S="${max_jump}" \
     bash "${E2_DIR}/scripts/run_e2.sh" policy boost 16 >> "${LOG}" 2>&1; then
    OK=$((OK + 1))
    log "=== CELL OK ${label} ==="
  else
    FAIL=$((FAIL + 1))
    log "=== CELL FAIL ${label} ==="
  fi
}

run_cell_ada "wall_ada_szfix_J60" "60"

log "=== SWEEP DONE ok=${OK} fail=${FAIL} log=${LOG} ==="
