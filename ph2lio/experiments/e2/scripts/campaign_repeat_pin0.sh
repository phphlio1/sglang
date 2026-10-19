#!/usr/bin/env bash
# Phase-1 repeat benchmark (pre-PR evidence): 3 seeds x {lpm, hrrn,
# boost gamma-Ada J=50} at c16, IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0.
# Protocol: .claude/plans/proud-inventing-panda.md Phase 1 (upstream boost
# PR). Fresh boot per cell (FORCE_RESTART=1, cold radix cache), full drain
# (GRACE_PERIOD default inf), SEED plumbed to the aiperf client and
# run_meta.json. Boost knobs only take effect on the boost arm (gated in
# start_server.sh); lpm/hrrn cells run exact upstream behavior at pin=0.
#
# No -e: one failed cell must not kill the campaign; failures are logged
# and summed in the tail.

set -uo pipefail

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${E2_DIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${E2_DIR}/logs/campaign_repeat_pin0_${STAMP}.log"
SEEDS=(42 43 44)
POLICIES=(lpm hrrn boost)
OK=0
FAIL=0

log() { printf '[%s] %s\n' "$(date +%F_%T)" "$*" | tee -a "${LOG}"; }

run_cell() {  # $1 = policy, $2 = seed
  local policy="$1" seed="$2"
  log "=== CELL START policy=${policy} seed=${seed} c16 pin=0 ==="
  # GRACE_PERIOD=600 matches the established c16 protocol (J=50 run): after
  # the client window closes, force-cancel remaining credits at 600 s. The
  # run_e2.sh default (inf) hangs forever on credits that never return
  # (rootless lane credits; server idle since close) -- lost cell 1 on the
  # first launch of this campaign.
  if SEED="${seed}" CONCURRENCY=16 DURATION=1800 FORCE_RESTART=1 \
     GRACE_PERIOD=600 IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0 \
     BOOST_GAMMA=0.09 BOOST_BIN_SIZE=256 BOOST_CLOCK=wall \
     BOOST_ADAPTIVE_GAMMA=1 BOOST_ADAPTIVE_MAX_JUMP_S=50 \
     bash "${E2_DIR}/scripts/run_e2.sh" policy "${policy}" 16 >> "${LOG}" 2>&1; then
    OK=$((OK + 1))
    log "=== CELL OK policy=${policy} seed=${seed} ==="
  else
    FAIL=$((FAIL + 1))
    log "=== CELL FAIL policy=${policy} seed=${seed} ==="
  fi
}

log "campaign start: policies=${POLICIES[*]} seeds=${SEEDS[*]} pin=0"
for policy in "${POLICIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_cell "${policy}" "${seed}"
  done
done
log "=== CAMPAIGN DONE ok=${OK} fail=${FAIL} log=${LOG} ==="
