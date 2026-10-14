#!/usr/bin/env bash
# Node side of the /mnt execution mirror. Executed via srun from the login
# node by mirror_from_login.sh, with the tar stream on stdin: extract the
# stream into /mnt/loipp, rewrite the venv-internal absolute paths, and
# self-verify by importing the whole stack from the mirror. Prints
# "=== MIRROR OK <host> ===" only on a full pass.
#
# All heavy I/O here is against node-local /mnt (fast). The only /shared
# access is reading this script itself (a few KB).
set -euo pipefail

MNT=/mnt/loipp
T0=$(date +%s)
say() { echo "[$(date +%H:%M:%S)] $* ($(( $(date +%s) - T0 ))s elapsed)"; }

say "node side start on $(hostname)"
FREE_GB=$(df -BG /mnt | awk 'NR==2{print $4}' | tr -dc '0-9')
say "/mnt free: ${FREE_GB}G (mirror needs ~11G)"
if [ "${FREE_GB:-0}" -lt 30 ]; then
  echo "=== MIRROR FAIL: /mnt has only ${FREE_GB}G free (<30G) ==="
  exit 1
fi
mkdir -p "${MNT}"

say "extracting tar stream from stdin (~10.6G)"
tar -C "${MNT}" -xf -
say "extract done"

fixup_venv() { # rewrite every /shared/loipp/ reference inside a mirrored
  # venv (shebangs, pyvenv.cfg home, editable-install finders) and re-point
  # absolute /shared/loipp/* symlinks (bin/python -> uv-python) at the mirror.
  local v="$1" f tgt
  grep -rlI "/shared/loipp/" "$v" 2>/dev/null | xargs -r sed -i "s|/shared/loipp/|/mnt/loipp/|g"
  find "$v" -type l -lname '/shared/loipp/*' 2>/dev/null | while IFS= read -r f; do
    tgt=$(readlink "$f")
    ln -sfn "${tgt/\/shared\/loipp\//\/mnt\/loipp\/}" "$f"
  done
}

say "rewriting venv-internal absolute paths (mirror copies only)"
fixup_venv "${MNT}/venv-server"
fixup_venv "${MNT}/venv-aiperf"
fixup_venv "${MNT}/venv-dynamo"

say "self-verify: import the whole stack from the mirror"
export HOME="${MNT}/home"
set +e
"${MNT}/venv-server/bin/python3" - <<'PY'
import sys
assert sys.executable.startswith("/mnt/loipp"), sys.executable
import sglang, torch, flashinfer, deep_gemm
assert "/mnt/loipp/sglang" in sglang.__file__, sglang.__file__
print("OK python:", sys.executable)
print("OK sglang:", sglang.__file__, getattr(sglang, "__version__", "?"))
print("OK torch:", torch.__version__, "cuda", torch.version.cuda)
print("OK flashinfer:", getattr(flashinfer, "__version__", "?"))
print("OK deep_gemm:", getattr(deep_gemm, "__version__", "?"))
PY
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "=== MIRROR FAIL: import verification failed (rc=${rc}) ==="
  exit 1
fi
"${MNT}/venv-aiperf/bin/aiperf" --help >/dev/null 2>&1 \
  || { echo "=== MIRROR FAIL: aiperf entrypoint broken ==="; exit 1; }
echo "OK aiperf"

# dynamo bootstrap venv (py3.12): dynamo frontend/sglang wrappers + our sglang
# branch + torch, plus the static nats-server binary.
export UV_PYTHON_INSTALL_DIR="${MNT}/uv-python"
set +e
"${MNT}/venv-dynamo/bin/python3" - <<'PY'
import sys
assert sys.executable.startswith("/mnt/loipp"), sys.executable
import dynamo.frontend, dynamo.sglang, dynamo.runtime
import sglang, torch
assert "/mnt/loipp/sglang" in sglang.__file__, sglang.__file__
print("OK dynamo python:", sys.executable)
print("OK dynamo.frontend / dynamo.sglang / dynamo.runtime")
print("OK torch:", torch.__version__, "cuda", torch.version.cuda)
PY
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "=== MIRROR FAIL: venv-dynamo verification failed (rc=${rc}) ==="
  exit 1
fi
"${MNT}/bin/nats-server" --version >/dev/null 2>&1 \
  || { echo "=== MIRROR FAIL: nats-server broken in mirror ==="; exit 1; }
echo "OK nats-server $(${MNT}/bin/nats-server --version 2>&1 | tail -1)"
NVCC="${MNT}/venv-server/lib/python3.11/site-packages/nvidia/cu13/bin/nvcc"
"${NVCC}" --version >/dev/null 2>&1 \
  || { echo "=== MIRROR FAIL: mirror nvcc broken ==="; exit 1; }
echo "OK $(${NVCC} --version | tail -1)"
"${MNT}/venv-server/bin/py-spy" --version >/dev/null 2>&1 \
  || echo "WARN: py-spy missing in mirror"
touch "${MNT}/home/.mirror_write_test" \
  || { echo "=== MIRROR FAIL: mirror home not writable ==="; exit 1; }
echo "OK home writable"

say "mirror sizes:"
du -sh "${MNT}"/* 2>/dev/null
echo "=== MIRROR OK $(hostname) ==="
