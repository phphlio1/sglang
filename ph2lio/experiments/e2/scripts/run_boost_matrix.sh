#!/usr/bin/env bash
# =============================================================================
# E2 boost comparison driver -- spec section 7 core matrix + gamma sweep
# =============================================================================
# One long sequential driver over scripts/run_e2.sh. Ordering minimizes server
# boots: within a block the server spec never changes, so run_e2.sh reuses the
# running server (STOP_SERVER=never); between blocks ensure_server restarts on
# the full-spec mismatch. STOP_SERVER=never also means each policy's server
# carries its radix cache across that policy's concurrencies -- identical
# treatment for every arm (same order, same dataset per seed).
#
#   fcfs   c4 c8 c16        (calibration source: R from the c8 run)
#   lpm    c4 c8 c16
#   boost  c4 c8 c16        (gamma=1e-4, k=256 -- paper defaults)
#   boost  c8               (gamma=5e-5, k=256)
#   boost  c8               (gamma=2e-4, k=256)
#
# 11 x 1800 s benchmark + 5 server boots, roughly 6.5 h wall-clock.
# Optional arms (uncomment when budget allows):
#   scripts/run_e2.sh policy hrrn 8
#   for s in 43 44; do SEED=$s scripts/run_e2.sh policy lpm 8;
#                     SEED=$s scripts/run_e2.sh policy boost 8; done

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export DURATION=1800 SEED=42 STOP_SERVER=never

run() {
  echo "=== [matrix $(date +%H:%M:%S)] run_e2.sh policy $* ==="
  scripts/run_e2.sh policy "$@"
}

for c in 4 8 16; do run fcfs "$c"; done
for c in 4 8 16; do run lpm "$c"; done
for c in 4 8 16; do BOOST_GAMMA=1e-4 run boost "$c"; done
BOOST_GAMMA=5e-5 run boost 8
BOOST_GAMMA=2e-4 run boost 8

bash scripts/start_server.sh --stop || true
echo "=== [matrix $(date +%H:%M:%S)] driver complete ==="
