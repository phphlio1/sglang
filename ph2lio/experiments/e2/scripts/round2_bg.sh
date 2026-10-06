#!/usr/bin/env bash
# Internal launcher: run the round-2 driver detached (used when the
# interactive shell cannot host a multi-hour foreground job).
# Output: /tmp/e2_round2.out
E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nohup bash "${E2_DIR}/scripts/run_round2.sh" >/tmp/e2_round2.out 2>&1 &
echo "round-2 driver launched (pid $!), output: /tmp/e2_round2.out"
