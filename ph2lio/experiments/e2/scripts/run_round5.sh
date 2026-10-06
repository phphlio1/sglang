#!/usr/bin/env bash
# =============================================================================
# E2 round-5 driver -- Task 5.2 seed replicas at gamma*
# =============================================================================
# Seeds 43/44 for lpm vs boost(gamma*) at c16 (4 fresh-boot runs) to put error
# bars on the frontier pick. gamma* comes from the round-2 frontier analysis
# (analysis/frontier_plot.py + the guard table); this driver refuses to run
# without it so a stale default gamma can never silently fill the cell.
#
# Runs land in the official tree (results/agentx): cells become multi-seed and
# the final table averages them (latest-per-seed, n-weighted).
#
# ~2.7 h wall-clock (4 x 1800 s + 4 boots). Usage:
#   GAMMA_STAR=<pick> scripts/run_round5.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[[ -n "${GAMMA_STAR:-}" ]] || { echo "ERROR: set GAMMA_STAR (the round-2 frontier pick) first" >&2; exit 2; }

export DURATION=1800 STOP_SERVER=always CONCURRENCY=16 OUTPUT_DIR=results/agentx
echo "=== [round5] seed replicas at gamma*=${GAMMA_STAR} ==="

run() {
  echo "=== [round5 $(date +%H:%M:%S)] run_e2.sh policy $* ==="
  scripts/run_e2.sh policy "$@"
}

for seed in 43 44; do
  SEED="${seed}" run lpm 16
  SEED="${seed}" BOOST_GAMMA="${GAMMA_STAR}" run boost 16
done

echo "=== [round5 $(date +%H:%M:%S)] complete ==="
