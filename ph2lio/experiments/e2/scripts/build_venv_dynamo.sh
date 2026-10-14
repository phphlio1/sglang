#!/usr/bin/env bash
# Build venv-dynamo: python3.12 venv carrying BOTH sides of the dynamo stack —
# the extracted dynamo packages (cp312, from extract_dynamo_pkg.sh) AND our
# sglang branch — because dynamo.sglang runs the engine IN-PROCESS via
# sgl.Engine, in one interpreter.
#
# Runs on the LOGIN node (like the original venv-server build): uv downloads +
# unpacks are low-RSS, everything lands on /shared. ~10-15 min.
#
# Prereq: /shared/loipp/dynamo-pkg/ from scripts/extract_dynamo_pkg.sh.
set -euo pipefail

UV=/home/loipp/.local/bin/uv
BENCH=/shared/loipp
VENV="${BENCH}/venv-dynamo"
PY="${VENV}/bin/python3"
SGLANG_ROOT="${BENCH}/sglang/python"

[[ -x "${UV}" ]] || { echo "ERROR: uv not found at ${UV}" >&2; exit 1; }
[[ -d "${BENCH}/dynamo-pkg/pkg/dynamo" ]] || {
  echo "ERROR: ${BENCH}/dynamo-pkg/pkg/dynamo missing — run scripts/extract_dynamo_pkg.sh first (srun, on a node)" >&2
  exit 1
}

# Same uv python install dir as venv-server's 3.11 (uv keeps each version in
# its own cpython-3.12.* subdir); nodes resolve the venv's bin/python symlink.
export UV_PYTHON_INSTALL_DIR="${BENCH}/uv-python"

echo "=== 1/6 uv python 3.12 + venv ==="
"${UV}" python install 3.12
"${UV}" venv --python 3.12 --seed "${VENV}"

# Install torch FIRST at the exact venv-server version so sglang's dependency
# resolution can't drag in a different one (default PyPI index; the nvidia
# cu13 wheels it pulls match venv-server's, whose CUDA_HOME lives inside).
echo "=== 2/6 torch + sglang (editable, our branch) ==="
"${UV}" pip install --python "${PY}" torch==2.14.1
SGLANG_BUILD_RUST_EXTS=none "${UV}" pip install --python "${PY}" -e "${SGLANG_ROOT}"

echo "=== 3/6 kernel/companion wheels (same versions as venv-server) ==="
# sgl-deep-ep / sgl-deep-gemm / sglang-kernel must have cp312 wheels; a hard
# error here means a source build is needed for one of them (surface it, do
# not silently substitute a different version).
"${UV}" pip install --python "${PY}" \
  flashinfer-python==0.7.0.post1 \
  sgl-deep-ep==0.1.2.post1 \
  sgl-deep-gemm==0.2.1.post1 \
  sglang-kernel==0.4.9

echo "=== 4/6 dynamo runtime deps (ai_dynamo METADATA) ==="
"${UV}" pip install --python "${PY}" \
  "pydantic>=2.10.6,<=2.13" \
  uvloop \
  "aiohttp>=3.14.3" \
  "kubernetes>=32.0.1,<33.0.0" \
  "msgspec>=0.19.0" \
  "prometheus-client>=0.23.1,<1.0" \
  "pyzmq>=26.0.0" \
  "zstandard>=0.23.0,<1.0"

echo "=== 5/6 drop in extracted dynamo packages (cp312) ==="
SP="${VENV}/lib/python3.12/site-packages"
cp -a "${BENCH}/dynamo-pkg/pkg/." "${SP}/"
[[ -x "${BENCH}/dynamo-pkg/bin/nats-server" ]] && {
  mkdir -p "${BENCH}/bin"
  cp -a "${BENCH}/dynamo-pkg/bin/nats-server" "${BENCH}/bin/nats-server"
}

echo "=== 6/6 import sanity (login, no GPU) ==="
"${PY}" - <<'EOF'
# The native runtime ships inside the `dynamo` package (dynamo/_internal);
# ai_dynamo_runtime only contributes dist-info metadata here.
import dynamo, dynamo.frontend, dynamo.sglang, dynamo.runtime
import sglang, torch, flashinfer
from sglang.srt.server_args_config_parser import ConfigArgumentMerger  # branch shim
print("imports OK: dynamo frontend/sglang/runtime | torch", torch.__version__,
      "| sglang importable, shim OK")
EOF

echo "=== BUILD VENV DYNAMO OK: ${VENV} ==="
