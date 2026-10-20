#!/usr/bin/env bash
# J-grid validation: boost arms at seed 77188 (the collapse seed), c16,
# pin=0, ALL with the leader-relative dedup fix (requires the
# boost-audit-fixes code; run from a checkout that has it merged).
#
# Purpose (2026-10-20 paper audit, findings F2 + F3):
#   - boost-g5.0 / boost-ada-j60 with the fix vs the same arms in the
#     fair-sweep campaign (identical seed, pin, commit-family, no fix)
#     isolate the starvation fix: prediction -- the 1500-1950 s demoted
#     followers drop to the fcfs/lpm band while keeping the cache hit.
#   - ada J in {60, 15, 5} isolates the max-jump knob: under overload the
#     tail fit sits far below the floor the whole run (ada@77188 live
#     gamma p50 ~ 0.120 ~= floor 0.115), so J -- not the fit -- sets the
#     effective gamma; smaller J = smaller allowed jump = more FCFS-ward.
#     The trajectory log now prints pinned=% to verify the floored state.
#
# Resumable: cells recorded in logs/boost_jgrid.progress are skipped.
# No -e: one failed cell must not kill the grid.

set -uo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${E2_DIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${E2_DIR}/logs/boost_jgrid_${STAMP}.log"
PROGRESS="${E2_DIR}/logs/boost_jgrid.progress"
SEED=77188
ARMS=(boost-g5.0 boost-ada-j60 boost-ada-j15 boost-ada-j5)
OK=0
FAIL=0
SKIP=0

log() { printf '[%s] %s\n' "$(date +%F_%T)" "$*" | tee -a "${LOG}"; }

arm_env() {  # arm -> boost env assignments
  case "$1" in
    boost-g5.0) echo "BOOST_GAMMA=5.0 BOOST_ADAPTIVE_GAMMA=0" ;;
    boost-ada-j60) echo "BOOST_GAMMA=0.09 BOOST_ADAPTIVE_GAMMA=1 BOOST_ADAPTIVE_MAX_JUMP_S=60" ;;
    boost-ada-j15) echo "BOOST_GAMMA=0.09 BOOST_ADAPTIVE_GAMMA=1 BOOST_ADAPTIVE_MAX_JUMP_S=15" ;;
    boost-ada-j5) echo "BOOST_GAMMA=0.09 BOOST_ADAPTIVE_GAMMA=1 BOOST_ADAPTIVE_MAX_JUMP_S=5" ;;
    *) echo "" ;;
  esac
}

cell_done() {  # $1 = arm
  [[ -f "${PROGRESS}" ]] && grep -qxF "$1 ${SEED}" "${PROGRESS}"
}

run_cell() {  # $1 = arm
  local arm="$1"
  if cell_done "${arm}"; then
    SKIP=$((SKIP + 1))
    log "--- SKIP ${arm} seed=${SEED} (already in progress file)"
    return 0
  fi
  local benv; benv="$(arm_env "${arm}")"
  log "=== CELL START ${arm} policy=boost seed=${SEED} c16 pin=0 (leader-relative fix) ==="
  # GRACE_PERIOD=600 matches the established c16 protocol; `env` carries
  # the per-arm boost vars (see policy_fair_sweep.sh for the parsing trap).
  if env SEED="${SEED}" CONCURRENCY=16 DURATION=1800 FORCE_RESTART=1 \
     GRACE_PERIOD=600 IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0 \
     BOOST_BIN_SIZE=256 BOOST_CLOCK=wall ${benv} \
     bash "${E2_DIR}/scripts/run_e2.sh" policy boost 16 >> "${LOG}" 2>&1; then
    OK=$((OK + 1))
    echo "${arm} ${SEED}" >> "${PROGRESS}"
    log "=== CELL OK ${arm} seed=${SEED} ==="
  else
    FAIL=$((FAIL + 1))
    log "=== CELL FAIL ${arm} seed=${SEED} ==="
  fi
}

log "jgrid start: arms=${ARMS[*]} seed=${SEED} pin=0 commit=$(git -C "${E2_DIR}" rev-parse --short HEAD)"
for arm in "${ARMS[@]}"; do
  run_cell "${arm}"
done
log "=== JGRID DONE ok=${OK} fail=${FAIL} skip=${SKIP} log=${LOG} ==="
