#!/usr/bin/env bash
# Watch a fair-sweep driver; when it exits, clear stray servers and
# relaunch the sweep. The driver skips cells recorded in the progress
# file, so the relaunch retries only unfinished cells.
#
# The relaunch guard must match only a live driver process: an
# unanchored "bash.*policy_fair_sweep.sh" also matched a concurrent
# monitoring shell (its cmdline merely mentioned the script) and
# aborted the relaunch. Anchor to the driver's exact cmdline shape.
#
# Cap: 36 h. Aborts (no relaunch) if the driver is still alive at the
# cap, or if another sweep driver is already running at relaunch time.

set -u

E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER_PID="${1:?usage: policy_fair_sweep_autorelaunch.sh <driver-pid>}"
RELAUNCH_LOG="${E2_DIR}/logs/policy_fair_sweep_relaunch.log"

say() { printf '[%s] %s\n' "$(date +%F_%T)" "$*" >> "${RELAUNCH_LOG}"; }

say "watcher start: driver pid=${DRIVER_PID} progress=$(wc -l < "${E2_DIR}/logs/policy_fair_sweep.progress" 2>/dev/null || echo 0) cells"

for _ in $(seq 1 2160); do  # 36 h at 60 s
  kill -0 "${DRIVER_PID}" 2>/dev/null || break
  sleep 60
done

if kill -0 "${DRIVER_PID}" 2>/dev/null; then
  say "watcher timeout (36 h) - driver still alive, no relaunch"
  exit 1
fi

say "driver ${DRIVER_PID} exited; progress=$(wc -l < "${E2_DIR}/logs/policy_fair_sweep.progress" 2>/dev/null || echo 0) cells"

# The driver exits only after its last run_e2.sh child returns, but a
# stray server from a torn-down cell would hold GPU memory into the next
# boot; clear anything left before relaunching.
sleep 15
if pgrep -f "launch_server|sglang::" >/dev/null 2>&1; then
  say "leftover server processes found; clearing"
  pkill -f "launch_server|sglang::" 2>/dev/null
  sleep 10
fi

if pgrep -f "bash [^ ]*policy_fair_sweep\.sh$" >/dev/null 2>&1; then
  say "another sweep driver is running - abort relaunch"
  exit 1
fi

say "relaunching fixed driver (retries the failed boost cells)"
cd "${E2_DIR}" || exit 1
setsid nohup bash scripts/policy_fair_sweep.sh >> "${RELAUNCH_LOG}" 2>"${E2_DIR}/logs/policy_fair_sweep_nohup2.out" < /dev/null &
sleep 5
say "relaunched: $(pgrep -f 'bash scripts/policy_fair_sweep.sh' | head -1)"
