#!/usr/bin/env bash
# =============================================================================
# E2 full five-policy comparison at c4/c8 -- the final RTX run
# =============================================================================
# 6 arms x {c4, c8} = 12 cells, fresh boot per cell (STOP_SERVER=always),
# seed 42, DURATION=1800, pinned serving config (scripts/start_server.sh,
# untouched). Order is risk-earliest per the fullrun plan: lpm baseline
# first, dfs-weight right after its pre-flight smoke, then the reference
# arms; boost is gamma-Ada (gamma0=1e-4, k=256; no static-gamma arm).
#
# A failed cell is logged and the sweep CONTINUES; quarantine the partial
# run dir under results/quarantine/ afterwards and report at the end.
#
# DFS_WEIGHT=0 drops the dfs-weight arms (pre-flight smoke failed).
#
# ~8 h wall-clock. Usage: nohup scripts/run_fullrun_c4c8.sh &

set -uo pipefail   # no -e: a failed cell must not stop the sweep
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export DURATION=1800 SEED=42 STOP_SERVER=always
DFS_WEIGHT="${DFS_WEIGHT:-1}"

LOG="logs/fullrun_c4c8_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG}") 2>&1

echo "=== [fullrun] start $(date)  DFS_WEIGHT=${DFS_WEIGHT} ==="
FAILED=()

run_cell() {  # $1 = policy, $2 = concurrency, rest = env VAR=VAL for the cell
  local policy="$1" conc="$2"; shift 2
  echo "=== [fullrun $(date +%H:%M:%S)] cell: policy=${policy} c=${conc} ==="
  if env "$@" scripts/run_e2.sh policy "${policy}" "${conc}"; then
    echo "=== [fullrun $(date +%H:%M:%S)] cell OK: ${policy} c${conc} ==="
  else
    echo "=== [fullrun $(date +%H:%M:%S)] cell FAILED (exit $?): ${policy} c${conc} ==="
    FAILED+=("${policy}_c${conc}")
  fi
}

for c in 4 8; do
  run_cell lpm "$c"
  [[ "${DFS_WEIGHT}" == "1" ]] && run_cell dfs-weight "$c"
  run_cell fcfs "$c"
  run_cell fcfs-match "$c"
  run_cell hrrn "$c"
  # BOOST_CLOCK=tokens pins this driver to the pre-wall-clock semantics its
  # landed cells ran (the RTX tables are tokens-clock numbers). The H200
  # campaign runs BOOST_CLOCK=wall (the fixed clock) with its own driver.
  run_cell boost "$c" BOOST_ADAPTIVE_GAMMA=1 BOOST_GAMMA=1e-4 BOOST_BIN_SIZE=256 BOOST_CLOCK=tokens
done

echo "=== [fullrun] complete $(date) ==="
if (( ${#FAILED[@]} )); then
  echo "=== [fullrun] FAILED cells: ${FAILED[*]} ==="
  exit 1
fi
echo "=== [fullrun] all cells landed ==="
