#!/usr/bin/env bash
# Fair 7-policy sweep: 5 random seeds x {fcfs, lpm, hrrn, dfs-weight,
# boost g=5.0, boost g=15.0, boost gamma-Ada J=60}, c16, pin=0, commit
# 0229372590 (post adaptive-gamma audit). Seed-major order: each seed
# block completes all 7 policies, so a partial campaign still yields
# balanced paired comparisons (same seed = same request mix).
#
# Motivation (2026-10-19 audit): single-seed policy gaps (~15%) sit inside
# a 60-89% seed band (seeds 42/43/44 replicates), so the ranking needs
# per-seed pairing and 5 seeds/arm. ~54 min/cell -> ~31.5 h total.
#
# Resumable: cells recorded in logs/policy_fair_sweep.progress are skipped;
# delete the file (or lines) to re-run cells. Failed cells are NOT recorded
# and retry on the next launch.
#
# Seeds drawn 2026-10-19 (python random.seed(20261019), sample 1..99998,
# disjoint from the 42/43/44 replicates): arbitrary but fixed for
# reproducibility.
#
# No -e: one failed cell must not kill the campaign; failures are logged
# and summed in the tail.

set -uo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${E2_DIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${E2_DIR}/logs/policy_fair_sweep_${STAMP}.log"
PROGRESS="${E2_DIR}/logs/policy_fair_sweep.progress"
SEEDS=(77188 42302 94815 15767 27781)
POLICIES=(fcfs lpm hrrn dfs-weight boost-g5.0 boost-g15.0 boost-ada-j60)
OK=0
FAIL=0
SKIP=0

log() { printf '[%s] %s\n' "$(date +%F_%T)" "$*" | tee -a "${LOG}"; }

policy_arg() {  # sweep key -> run_e2.sh policy name
  case "$1" in
    boost-*) echo "boost" ;;
    *) echo "$1" ;;
  esac
}

boost_env() {  # sweep key -> boost-arm env assignments (empty off the boost arm)
  case "$1" in
    boost-g5.0) echo "BOOST_GAMMA=5.0 BOOST_ADAPTIVE_GAMMA=0" ;;
    boost-g15.0) echo "BOOST_GAMMA=15.0 BOOST_ADAPTIVE_GAMMA=0" ;;
    boost-ada-j60) echo "BOOST_GAMMA=0.09 BOOST_ADAPTIVE_GAMMA=1 BOOST_ADAPTIVE_MAX_JUMP_S=60" ;;
    *) echo "" ;;
  esac
}

cell_done() {  # $1 = key, $2 = seed
  [[ -f "${PROGRESS}" ]] && grep -qxF "$1 $2" "${PROGRESS}"
}

run_cell() {  # $1 = sweep key, $2 = seed
  local key="$1" seed="$2"
  if cell_done "${key}" "${seed}"; then
    SKIP=$((SKIP + 1))
    log "--- SKIP ${key} seed=${seed} (already in progress file)"
    return 0
  fi
  local pol; pol="$(policy_arg "${key}")"
  local benv; benv="$(boost_env "${key}")"
  log "=== CELL START ${key} policy=${pol} seed=${seed} c16 pin=0 ==="
  # GRACE_PERIOD=600 matches the established c16 protocol: the run_e2.sh
  # default (inf) hangs forever on credits that never return.
  if SEED="${seed}" CONCURRENCY=16 DURATION=1800 FORCE_RESTART=1 \
     GRACE_PERIOD=600 IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0 \
     BOOST_BIN_SIZE=256 BOOST_CLOCK=wall ${benv} \
     bash "${E2_DIR}/scripts/run_e2.sh" policy "${pol}" 16 >> "${LOG}" 2>&1; then
    OK=$((OK + 1))
    echo "${key} ${seed}" >> "${PROGRESS}"
    log "=== CELL OK ${key} seed=${seed} ==="
  else
    FAIL=$((FAIL + 1))
    log "=== CELL FAIL ${key} seed=${seed} ==="
  fi
}

log "fair sweep start: policies=${POLICIES[*]} seeds=${SEEDS[*]} pin=0 commit=$(git -C "${E2_DIR}" rev-parse --short HEAD)"
for seed in "${SEEDS[@]}"; do
  for key in "${POLICIES[@]}"; do
    run_cell "${key}" "${seed}"
  done
  log "=== SEED BLOCK DONE seed=${seed} ok=${OK} fail=${FAIL} skip=${SKIP} ==="
done
log "=== SWEEP DONE ok=${OK} fail=${FAIL} skip=${SKIP} log=${LOG} ==="
