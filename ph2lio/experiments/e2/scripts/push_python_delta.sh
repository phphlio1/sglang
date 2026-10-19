#!/usr/bin/env bash
# Fast-path mirror push: sync ONLY the python/ files that changed since the
# last mirror, computed by git (login-side, instant) instead of walking the
# tree over NFS (the reason full mirrors stream a tar — see
# mirror_from_login.sh header for the measured numbers).
#
# When to use which:
#   - small committed edits under python/        -> this script (seconds)
#   - merges, venv/home/bin changes, or after a
#     rebase/reset that orphaned the stamp        -> mirror_from_login.sh
#     (full tar ~25-40 min; rewrites the stamp)
#   - ph2lio scripts/docs/results                -> nothing: sbatch drivers
#     read them live from /shared at submit time
#
# Only COMMITTED changes are pushed (git diff stamp..HEAD): commit first —
# the campaign records sglang_commit in run_meta/e2_metrics_v2 anyway, so
# commit-before-run is already the discipline. Uncommitted python/ changes
# trigger a warning.
#
# The stamp /mnt/loipp/.mirror_commit (written by this script and by
# mirror_from_login.sh) names the commit the node mirror currently holds.
# CPU-only: no GPU is requested, safe while GPUs are allocated elsewhere
# (the srun may still queue for CPUs).
#
# Usage (from the login node):
#   bash push_python_delta.sh [hgx-h200-01]
# Success prints "=== DELTA PUSH OK <host> ===".
set -euo pipefail

NODE="${1:-hgx-h200-01}"
BENCH=/shared/loipp
REPO="${BENCH}/sglang"
SRUN="srun -p hardware-hcm -w ${NODE} --cpus-per-task=2 --mem=2G --time=00:10:00"

cd "${REPO}"
HEAD="$(git rev-parse HEAD)"

if [ -n "$(git status --porcelain -- python/)" ]; then
  echo "WARN: uncommitted changes under python/ exist and are NOT pushed:" >&2
  git status --short -- python/ >&2
fi

STAMP="$(${SRUN} cat /mnt/loipp/.mirror_commit 2>/dev/null | tr -dc '0-9a-f' || true)"
if [ -z "${STAMP}" ]; then
  echo "=== DELTA PUSH FAIL: no mirror stamp on ${NODE} — run mirror_from_login.sh ${NODE} once first ===" >&2
  exit 1
fi
if ! git cat-file -e "${STAMP}^{commit}" 2>/dev/null; then
  echo "=== DELTA PUSH FAIL: stamp ${STAMP} is not a commit in this repo (rebase/reset?) — rerun mirror_from_login.sh ${NODE} ===" >&2
  exit 1
fi
if [ "${STAMP}" = "${HEAD}" ]; then
  echo "=== DELTA PUSH OK ${NODE}: mirror already at HEAD ${HEAD} (nothing to do) ==="
  exit 0
fi

CHANGED="$(git diff -z --name-only --diff-filter=ACMR "${STAMP}".."${HEAD}" -- python/ | tr '\0' '\n')"
DELETED="$(git diff -z --name-only --diff-filter=D "${STAMP}".."${HEAD}" -- python/ | tr '\0' '\n')"
count_lines() { if [ -n "$1" ]; then printf '%s\n' "$1" | wc -l; else echo 0; fi; }
N_CHANGED="$(count_lines "${CHANGED}")"
N_DELETED="$(count_lines "${DELETED}")"
echo "[$(date +%H:%M:%S)] ${STAMP}..${HEAD}: ${N_CHANGED} changed, ${N_DELETED} deleted"

if [ -n "${CHANGED}" ]; then
  # Stream only the changed members; --null -T - keeps paths (and any odd
  # characters) intact. Paths are repo-relative (python/...), and the node
  # mirror preserves that layout under /mnt/loipp/sglang.
  git diff -z --name-only --diff-filter=ACMR "${STAMP}".."${HEAD}" -- python/ \
    | tar --null -C "${REPO}" -cf - -T - \
    | ${SRUN} bash -c 'tar -C /mnt/loipp/sglang -xf -'
fi

if [ -n "${DELETED}" ]; then
  # Mirror deletions too (stale modules could otherwise shadow nothing but
  # linger); hard-validated to stay under python/ before rm.
  printf '%s\n' "${DELETED}" | ${SRUN} bash -c '
    while IFS= read -r f; do
      case "${f}" in
        python/*) rm -f -- "/mnt/loipp/sglang/${f}" ;;
        *) echo "REFUSING to delete outside python/: ${f}" >&2; exit 1 ;;
      esac
    done'
fi

# Self-verify (CPU-only): byte-compile every changed file from the mirror +
# import the stack the way the engine does (dynamo venv runs it in-process).
git diff -z --name-only --diff-filter=ACMR "${STAMP}".."${HEAD}" -- python/ \
  | tr "\0" "\n" | ${SRUN} bash -c '
    files=(); while IFS= read -r f; do
      [ -f "/mnt/loipp/sglang/${f}" ] && files+=("/mnt/loipp/sglang/${f}")
    done
    export HOME=/mnt/loipp/home
    if [ "${#files[@]}" -gt 0 ]; then
      /mnt/loipp/venv-server/bin/python3 -m py_compile "${files[@]}" \
        || { echo "=== DELTA PUSH FAIL: py_compile failed ==="; exit 1; }
      echo "OK py_compile x${#files[@]}"
    fi
    /mnt/loipp/venv-dynamo/bin/python3 -I - <<"PY"
import sys
assert sys.executable.startswith("/mnt/loipp"), sys.executable
import sglang
assert "/mnt/loipp/sglang" in sglang.__file__, sglang.__file__
import dynamo.sglang
print("OK imports: sglang + dynamo.sglang from mirror")
PY
  '
${SRUN} bash -c "echo ${HEAD} > /mnt/loipp/.mirror_commit"

echo "=== DELTA PUSH OK ${NODE}: ${N_CHANGED} pushed, ${N_DELETED} deleted, mirror now at ${HEAD} ==="
