#!/usr/bin/env bash
# Internal launcher: wait for the round-2 driver to release the GPU, then run
# the round-4 targeted-regime driver detached. Round-4 is independent of the
# gamma* pick (it uses the default gamma), so it is safe to chain blindly.
# Output: /tmp/e2_round4.out
E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R2_PID="$(pgrep -f '[r]un_round2.sh' | head -1)"
if [[ -z "${R2_PID}" ]]; then
  echo "no run_round2.sh process found; launching round-4 immediately"
  nohup bash "${E2_DIR}/scripts/run_round4.sh" >/tmp/e2_round4.out 2>&1 &
else
  # Wait on the pid, not a name pattern: the watcher's own command line would
  # match any pattern it greps for.
  nohup bash -c "
    while kill -0 ${R2_PID} 2>/dev/null; do sleep 60; done
    bash '${E2_DIR}/scripts/run_round4.sh'
  " >/tmp/e2_round4.out 2>&1 &
fi
echo "round-4 driver queued behind round-2 (pid ${R2_PID:-none}; watcher pid $!), output: /tmp/e2_round4.out"
