#!/usr/bin/env bash
# =============================================================================
# E2 round-4 driver -- Task 4 targeted starvation regime at c8
# =============================================================================
# Construction (numbers from analysis/select_starvation_entries.py): the pool
# is capped to the first 16 trajectories (16 long multi-turn root chains, mean
# 30 turns, + 121 short child chains that spawn at 28-285 s branch offsets),
# and sessions resume within the first quarter of each trajectory -- early
# turns carry seconds-scale think-times while late turns carry minutes-scale
# agent-work gaps, and the scenario forbids every delay-compression flag, so
# the resume window is the only arrival-density lever. The force-enabled
# first-turn-prefix cache-bust keeps wrapped re-draws cache-cold, so cold
# newcomers keep arriving after the pool wraps.
#
#   R4a  300 s gate (lpm, c8, diagnostic-only --unsafe-override run): verify
#         the regime builds a cold-class tail (cold n >= 10, cold p99 >= 5 s).
#         A fail is the trace-structure finding: stop and document instead of
#         burning 3 h of full runs.
#   R4b  lpm / boost (default gamma) / fcfs-match at c8, 1800 s, fresh boot
#         each, into results/agentx_targeted (kept out of the official matrix;
#         headline: cold-class p99, cold-vs-warm gap, arrival-order slope).
#
# ~3.5 h wall-clock (gate + 3 x 1800 s + 4 boots).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export NUM_DATASET_ENTRIES=16
export TRAJECTORY_START_MIN_RATIO=0.0
export TRAJECTORY_START_MAX_RATIO=0.25
export CONCURRENCY=8 SEED=42 STOP_SERVER=always

run() {
  echo "=== [round4 $(date +%H:%M:%S)] run_e2.sh policy $* ==="
  scripts/run_e2.sh policy "$@"
}

# R4a: gate.
GATE_DIR=results/agentx_targeted/gatecheck
SMOKE_POLICY=lpm SMOKEDURATION=300 OUTPUT_DIR="${GATE_DIR}" scripts/run_e2.sh smoke
gate_run="$(ls -td ${GATE_DIR}/smoke/smoke_c8_* | head -1)"
python3 analysis/agentx_metrics.py "${gate_run}"
gate_ok="$(python3 - "${gate_run}/e2_metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["queue_wait_per_request"]
cold = m["cold_class"]
print(f"cold_n={cold['n']} cold_p99_s={cold['p99_s'] if cold['p99_s'] is not None else -1.0}")
PY
)"
echo "=== [round4] gate: ${gate_ok} (need cold_n >= 10 and cold_p99_s >= 5.0) ==="
python3 -c "import sys; s='${gate_ok}'.split(); n=int(s[0].split('=')[1]); p=float(s[1].split('=')[1].split('s')[0] if s[1].split('=')[1].endswith('s') else s[1].split('=')[1]); sys.exit(0 if n >= 10 and p >= 5.0 else 1)" \
  || { echo "=== [round4] GATE FAILED: no cold-class tail at c8 under the targeted regime; trace-structure finding -- see docs/agentx_experiment.md ==="; exit 2; }

# R4b: the three arms.
OUTPUT_DIR=results/agentx_targeted run lpm 8
OUTPUT_DIR=results/agentx_targeted run boost 8
OUTPUT_DIR=results/agentx_targeted run fcfs-match 8

echo "=== [round4 $(date +%H:%M:%S)] complete: results/agentx_targeted ==="
