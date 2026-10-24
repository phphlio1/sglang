#!/usr/bin/env bash
# e2A lambda sweep: the pre-registered arm grid from docs/e2a_design.md.
# 9 arms x SEEDS (default "77188 42302") at c16, fresh boot per cell
# (FORCE_RESTART=1), DURATION=1800s, GRACE_PERIOD=600s, pin=0 -- the final
# RTX-campaign protocol the archived baselines ran under.
#
# Arms, in run order per seed:
#   boost-ref   boost g0.25 wall k256                 (lambda=0 reference)
#   bc-l025     boost-cache g0.25 wall k256 l0.25 m8192
#   bc-l050     boost-cache g0.25 wall k256 l0.5  m8192   (default knob point)
#   bc-l100     boost-cache g0.25 wall k256 l1.0  m8192
#   bc-m2048    boost-cache g0.25 wall k256 l0.5  m2048   (cap sensitivity)
#   bc-m32768   boost-cache g0.25 wall k256 l0.5  m32768
#   bc-ada      boost-cache ada j50       l0.5  m8192   (shared-budget floor)
#   lpm         cache-reuse reference line
#   dfs-weight  throughput reference line
#
# Usage:
#   bash scripts/run_e2a_sweep.sh                 # all arms, both seeds
#   SEEDS="77188" bash scripts/run_e2a_sweep.sh bc-l050   # one cell
# Every harness knob stays overridable via env (CONCURRENCY, DURATION, ...).

set -uo pipefail   # no -e: one dead cell must not kill the rest of the grid
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-77188 42302}"
CONCURRENCY="${CONCURRENCY:-16}"
DURATION="${DURATION:-1800}"
if [[ $# -gt 0 ]]; then ARMS=("$@"); else
  ARMS=(boost-ref bc-l025 bc-l050 bc-l100 bc-m2048 bc-m32768 bc-ada lpm dfs-weight)
fi

run_cell() {  # $1 = seed, $2 = arm
  local seed="$1" arm="$2"
  echo "=== E2A sweep cell: arm=${arm} seed=${seed} c=${CONCURRENCY} ==="
  local common=(
    SEED="${seed}" CONCURRENCY="${CONCURRENCY}" DURATION="${DURATION}"
    FORCE_RESTART=1 GRACE_PERIOD=600 IN_BATCH_PREFIX_CACHING_MAX_PIN_S=0
    BOOST_CLOCK=wall BOOST_BIN_SIZE=256
  )
  case "${arm}" in
    boost-ref)  env "${common[@]}" BOOST_GAMMA=0.25 bash scripts/run_e2.sh policy boost "${CONCURRENCY}" ;;
    bc-l025)    env "${common[@]}" BOOST_GAMMA=0.25 BOOST_CACHE_LAMBDA=0.25 BOOST_CACHE_CAP=8192  bash scripts/run_e2.sh policy boost-cache "${CONCURRENCY}" ;;
    bc-l050)    env "${common[@]}" BOOST_GAMMA=0.25 BOOST_CACHE_LAMBDA=0.5  BOOST_CACHE_CAP=8192  bash scripts/run_e2.sh policy boost-cache "${CONCURRENCY}" ;;
    bc-l100)    env "${common[@]}" BOOST_GAMMA=0.25 BOOST_CACHE_LAMBDA=1.0  BOOST_CACHE_CAP=8192  bash scripts/run_e2.sh policy boost-cache "${CONCURRENCY}" ;;
    bc-m2048)   env "${common[@]}" BOOST_GAMMA=0.25 BOOST_CACHE_LAMBDA=0.5  BOOST_CACHE_CAP=2048  bash scripts/run_e2.sh policy boost-cache "${CONCURRENCY}" ;;
    bc-m32768)  env "${common[@]}" BOOST_GAMMA=0.25 BOOST_CACHE_LAMBDA=0.5  BOOST_CACHE_CAP=32768 bash scripts/run_e2.sh policy boost-cache "${CONCURRENCY}" ;;
    bc-ada)     env "${common[@]}" BOOST_ADAPTIVE_GAMMA=1 BOOST_ADAPTIVE_MAX_JUMP_S=50 BOOST_CACHE_LAMBDA=0.5 BOOST_CACHE_CAP=8192 bash scripts/run_e2.sh policy boost-cache "${CONCURRENCY}" ;;
    lpm)        env "${common[@]}" bash scripts/run_e2.sh policy lpm "${CONCURRENCY}" ;;
    dfs-weight) env "${common[@]}" bash scripts/run_e2.sh policy dfs-weight "${CONCURRENCY}" ;;
    *) echo "[e2a-sweep] unknown arm: ${arm}" >&2; return 2 ;;
  esac
}

failures=()
total=0
for seed in ${SEEDS}; do
  for arm in "${ARMS[@]}"; do
    total=$((total + 1))
    if run_cell "${seed}" "${arm}"; then
      echo "=== E2A sweep cell OK: arm=${arm} seed=${seed} ==="
    else
      echo "=== E2A sweep cell FAILED: arm=${arm} seed=${seed} (continuing) ==="
      failures+=("${arm}@${seed}")
    fi
  done
done

echo "=== E2A SWEEP DONE: ${total} cells, ${#failures[@]} failed ${failures[*]+"(${failures[*]})"} ==="
[[ ${#failures[@]} -eq 0 ]]
