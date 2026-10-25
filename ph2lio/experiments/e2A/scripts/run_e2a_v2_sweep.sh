#!/usr/bin/env bash
# e2A v2 (cache-aging) alpha sweep: the approved v2 redesign arm grid.
# 4 arms x SEEDS (default "77188 42302") at c16, fresh boot per cell
# (FORCE_RESTART=1), DURATION=1800s, GRACE_PERIOD=600s, pin=0 -- the final
# RTX-campaign protocol the archived baselines ran under.
#
# v2: phi' = A - b_gamma(q(u)) - alpha*m_hat*age, m_hat in [0,1]; the credit
# is earned by waiting (zero at arrival), so the newcomer jump bound stays
# b_gamma(k) and J keeps its plain-boost meaning. All arms run the adaptive
# base (gamma-Ada j50) per the H200 evidence that adaptive beats static.
#
# Arms, in run order per seed:
#   ada-ref   boost ada j50 wall k256              (alpha=0 reference; the
#             archived campaign has no plain ada j50 c16 pin=0 cell -- it
#             ran j60 -- so the reference is measured here, not reused)
#   bc-a025   boost-cache ada j50 wall k256 a0.25
#   bc-a050   boost-cache ada j50 wall k256 a0.5   (default knob point)
#   bc-a100   boost-cache ada j50 wall k256 a1.0
#
# Comparators (lpm, dfs-weight, boost g0.25 static) are NOT re-run: the
# archived lambda-sweep cells used the same protocol (c16, 1800s, pin=0,
# fresh boot) and stay valid baselines; see results/1x-rtx6000pro/agentx.
#
# Usage:
#   bash scripts/run_e2a_v2_sweep.sh                 # all arms, both seeds
#   SEEDS="77188" bash scripts/run_e2a_v2_sweep.sh bc-a050   # one cell
# Every harness knob stays overridable via env (CONCURRENCY, DURATION, ...).

set -uo pipefail   # no -e: one dead cell must not kill the rest of the grid
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-77188 42302}"
CONCURRENCY="${CONCURRENCY:-16}"
DURATION="${DURATION:-1800}"
if [[ $# -gt 0 ]]; then ARMS=("$@"); else
  ARMS=(ada-ref bc-a025 bc-a050 bc-a100)
fi

run_cell() {  # $1 = seed, $2 = arm
  local seed="$1" arm="$2"
  echo "=== E2A v2 sweep cell: arm=${arm} seed=${seed} c=${CONCURRENCY} ==="
  local common=(
    SEED="${seed}" CONCURRENCY="${CONCURRENCY}" DURATION="${DURATION}"
    FORCE_RESTART=1 GRACE_PERIOD=600 IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0
    BOOST_CLOCK=wall BOOST_BIN_SIZE=256
    BOOST_ADAPTIVE_GAMMA=1 BOOST_ADAPTIVE_MAX_JUMP_S=50
  )
  case "${arm}" in
    ada-ref)  env "${common[@]}" bash scripts/run_e2.sh policy boost "${CONCURRENCY}" ;;
    bc-a025)  env "${common[@]}" BOOST_CACHE_ALPHA=0.25 bash scripts/run_e2.sh policy boost-cache "${CONCURRENCY}" ;;
    bc-a050)  env "${common[@]}" BOOST_CACHE_ALPHA=0.5  bash scripts/run_e2.sh policy boost-cache "${CONCURRENCY}" ;;
    bc-a100)  env "${common[@]}" BOOST_CACHE_ALPHA=1.0  bash scripts/run_e2.sh policy boost-cache "${CONCURRENCY}" ;;
    *) echo "[e2a-v2-sweep] unknown arm: ${arm}" >&2; return 2 ;;
  esac
}

failures=()
total=0
for seed in ${SEEDS}; do
  for arm in "${ARMS[@]}"; do
    total=$((total + 1))
    if run_cell "${seed}" "${arm}"; then
      echo "=== E2A v2 sweep cell OK: arm=${arm} seed=${seed} ==="
    else
      echo "=== E2A v2 sweep cell FAILED: arm=${arm} seed=${seed} (continuing) ==="
      failures+=("${arm}@${seed}")
    fi
  done
done

echo "=== E2A V2 SWEEP DONE: ${total} cells, ${#failures[@]} failed ${failures[*]+"(${failures[*]})"} ==="
[[ ${#failures[@]} -eq 0 ]]
