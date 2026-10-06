#!/usr/bin/env bash
# Internal launcher: run the smoke test detached (used when the interactive
# shell cannot host a long foreground job). Output: /tmp/e2_smoke.out.
E2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nohup bash "${E2_DIR}/scripts/run_agentx_smoke.sh" >/tmp/e2_smoke.out 2>&1 &
echo "smoke test launched (pid $!), output: /tmp/e2_smoke.out"
