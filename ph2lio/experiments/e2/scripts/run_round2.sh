#!/usr/bin/env bash
# =============================================================================
# E2 round-2 driver -- uniboost follow-up (Tasks 1b, 2, 5.1)
# =============================================================================
# Fresh-boot-per-cell protocol: every run below boots its own server
# (STOP_SERVER=always). The round-1 matrix reused one server across each
# policy's concurrencies, so its c8/c16 runs started cache-warm; round-2 cells
# are all cold-start and comparable to each other.
#
#   R1  fcfs-match c4, c16       (Task 1b: prefix-match bookkeeping forced on;
#                                 c4 gated on hit ~ lpm level before c16 runs)
#   R2  boost c16 gamma sweep    (Task 2: 3e-6, 1e-5, 2e-5, 3e-5, then a fresh
#                                 1e-4 re-run whose delta vs the round-1 point
#                                 measures the warm-start confound directly)
#   R3  hrrn c8, c16             (Task 5.1 reference arms)
#
# 9 x 1800 s benchmark + 9 server boots, roughly 7.5 h wall-clock. Later stages
# (targeted regime, seeds 43/44 at gamma*, adaptive-gamma validation) run as
# separate invocations once gamma* is picked from the R2 frontier.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export DURATION=1800 SEED=42 STOP_SERVER=always

run() {
  echo "=== [round2 $(date +%H:%M:%S)] run_e2.sh policy $* ==="
  scripts/run_e2.sh policy "$@"
}

# R1: fcfs-match arm. c4 first: if the official hit does not jump to lpm level
# (round-1 lpm c4: 0.883; smoke fcfs-match: 0.837), the matching path has
# another gate -- stop before burning c16 time.
run fcfs-match 4
hit="$(python3 - <<'PY'
import glob, json
paths = sorted(glob.glob("results/agentx/fcfs-match/c4_*/e2_metrics.json"))
m = json.load(open(paths[-1]))
print(m["queue_wait_per_request"]["first_attempt_cache_hit_rate"])
PY
)"
echo "=== [round2] fcfs-match c4 hit = ${hit} (gate: >= 0.5) ==="
python3 -c "import sys; sys.exit(0 if float('${hit}') >= 0.5 else 1)" \
  || { echo "=== [round2] GATE FAILED: fcfs-match c4 hit ${hit} < 0.5; stopping before c16 ==="; exit 2; }
run fcfs-match 16

# R2: boost gamma sweep at c16 (Task 2), gamma ascending.
for g in 3e-6 1e-5 2e-5 3e-5 1e-4; do BOOST_GAMMA="${g}" run boost 16; done

# R3: hrrn reference arms (Task 5.1).
run hrrn 8
run hrrn 16

echo "=== [round2 $(date +%H:%M:%S)] R1-R3 complete ==="
