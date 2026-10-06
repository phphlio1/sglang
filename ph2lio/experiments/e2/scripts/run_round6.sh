#!/usr/bin/env bash
# =============================================================================
# E2 round-6 driver -- Task 8.3 gamma-Ada validation
# =============================================================================
# Adaptive-gamma arms (server flag --boost-adaptive-gamma; gamma0 = 1e-4,
# EMA beta 0.2, clip [3e-7, 1e-3]) at c8 and c16, plus a fresh static-gamma*
# c8 reference (the c16 static gamma* cell already exists from round 2).
# Validation afterwards (analysis/adaptive_gamma_check.py): the converged
# gamma from the eval-log trajectory must land inside the round-2 sweep's best
# bracket, and hit / cold-class starvation must be no worse than the sweep's
# best static gamma.
#
# Arms land in results/agentx_adaptive (kept out of the official matrix; the
# ada eval path and run label keep them distinct from static cells).
#
# ~2 h wall-clock (3 x 1800 s + 3 boots). Usage:
#   GAMMA_STAR=<pick> scripts/run_round6.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[[ -n "${GAMMA_STAR:-}" ]] || { echo "ERROR: set GAMMA_STAR (the round-2 frontier pick) first" >&2; exit 2; }

export DURATION=1800 SEED=42 STOP_SERVER=always OUTPUT_DIR=results/agentx_adaptive
echo "=== [round6] gamma-Ada validation (static reference gamma*=${GAMMA_STAR}) ==="

run() {
  echo "=== [round6 $(date +%H:%M:%S)] run_e2.sh policy $* ==="
  scripts/run_e2.sh policy "$@"
}

# Adaptive arms: boost_gamma in the env is gamma0; the server rebinds it online.
BOOST_ADAPTIVE_GAMMA=1 run boost 8
BOOST_ADAPTIVE_GAMMA=1 run boost 16

# Fresh static gamma* reference at c8 (round-1 c8 boost was warm-start; the
# adaptive-vs-static c8 comparison needs both arms under the round-2 protocol).
BOOST_ADAPTIVE_GAMMA=0 BOOST_GAMMA="${GAMMA_STAR}" run boost 8

echo "=== [round6 $(date +%H:%M:%S)] complete: results/agentx_adaptive ==="
