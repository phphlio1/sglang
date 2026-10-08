#!/usr/bin/env bash
# =============================================================================
# E2 full six-arm comparison at c16 -- post-fix RTX validation sweep
# =============================================================================
# 6 arms x c16 = 6 cells, fresh boot per cell (STOP_SERVER=always),
# seed 42, DURATION=1800, pinned serving config (scripts/start_server.sh,
# untouched). Order is risk-earliest per the fullrun plan: lpm baseline
# first, dfs-weight right after, then the reference arms; boost last
# (gamma-Ada, wall clock).
#
# This is the first sweep on the fixed scheduler: every cell runs with the
# 60 s follower-pin bound (IN_BATCH_PREFIX_CACHING_MAX_PIN_S default) and
# the boost arm runs the wall clock (BOOST_CLOCK=wall, gamma0 0.1 1/s = the
# clock default), i.e. the config the H200 campaign will run. The landed
# c16 tables are tokens-clock with the unbounded pin, so the deltas vs
# those tables are the fix, not noise: reference arms isolate the pin
# bound, boost shows the full clock fix.
#
# A failed cell is logged and the sweep CONTINUES; quarantine the partial
# run dir under results/quarantine/ afterwards and report at the end.
#
# DFS_WEIGHT=0 drops the dfs-weight arms.
#
# ~4 h wall-clock. Usage: nohup scripts/run_fullrun_c16.sh &

set -uo pipefail   # no -e: a failed cell must not stop the sweep
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export DURATION=1800 SEED=42 STOP_SERVER=always
DFS_WEIGHT="${DFS_WEIGHT:-1}"

LOG="logs/fullrun_c16_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG}") 2>&1

echo "=== [fullrun-c16] start $(date)  DFS_WEIGHT=${DFS_WEIGHT} ==="
FAILED=()

run_cell() {  # $1 = policy, $2 = concurrency, rest = env VAR=VAL for the cell
  local policy="$1" conc="$2"; shift 2
  echo "=== [fullrun-c16 $(date +%H:%M:%S)] cell: policy=${policy} c=${conc} ==="
  if env "$@" scripts/run_e2.sh policy "${policy}" "${conc}"; then
    echo "=== [fullrun-c16 $(date +%H:%M:%S)] cell OK: ${policy} c${conc} ==="
  else
    echo "=== [fullrun-c16 $(date +%H:%M:%S)] cell FAILED (exit $?): ${policy} c${conc} ==="
    FAILED+=("${policy}_c${conc}")
  fi
}

run_cell lpm 16
[[ "${DFS_WEIGHT}" == "1" ]] && run_cell dfs-weight 16
run_cell fcfs 16
run_cell fcfs-match 16
run_cell hrrn 16
# gamma0 = 0.1 1/s, the wall-clock default (geometric center of the measured
# c4-c16 adaptive fixed points); gamma-Ada converges from it.
run_cell boost 16 BOOST_ADAPTIVE_GAMMA=1 BOOST_GAMMA=0.1 BOOST_BIN_SIZE=256 BOOST_CLOCK=wall

# Print the canonical tables, counted per full run (this sweep's block
# first -- latest full run prints first).
echo "=== [fullrun-c16 $(date +%H:%M:%S)] results tables ==="
python3 analysis/print_results_table.py \
  || echo "=== [fullrun-c16] results tables FAILED (non-fatal) ==="

echo "=== [fullrun-c16] complete $(date) ==="
if (( ${#FAILED[@]} )); then
  echo "=== [fullrun-c16] FAILED cells: ${FAILED[*]} ==="
  exit 1
fi
echo "=== [fullrun-c16] all cells landed ==="
