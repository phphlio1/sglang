#!/usr/bin/env bash
# Extract the dynamo python packages + nats-server from the ai-dynamo
# container image into /shared/loipp/dynamo-pkg/ so build_venv_dynamo.sh can
# drop them into a py3.12 venv. Runs ON A COMPUTE NODE via srun (unsquashfs is
# not installed on the login node):
#   srun -p hardware-hcm -w hgx-h200-01 --cpus-per-task=4 --mem=16G \
#     --time=00:40:00 bash scripts/extract_dynamo_pkg.sh
#
# Source image: nvcr.io_nvidia_ai-dynamo_vllm_runtime_1.4.2.sqsh (team-placed
# in /mnt/containers). Prod frontend image (sglang-runtime:1.5.0-kvoverlap1)
# is not on the node yet; 1.4.2 is the bootstrap substitute — swap the image
# path below when the exact prod image lands.
set -euo pipefail

IMAGE=/mnt/containers/nvcr.io_nvidia_ai-dynamo_vllm-runtime_1.4.2.sqsh
DEST=/shared/loipp/dynamo-pkg
SP=usr/local/lib/python3.12/dist-packages

[[ -f "${IMAGE}" ]] || { echo "ERROR: image not found: ${IMAGE}" >&2; exit 1; }
mkdir -p "${DEST}"

echo "=== listing dynamo entries in image ==="
unsquashfs -l "${IMAGE}" > /tmp/dynamo_full_list.txt 2>/dev/null
# dynamo python stack: the `dynamo` package (ai_dynamo wheel), the
# ai_dynamo_runtime native package (Rust bindings, cp312) and both dist-infos.
AI_DYNAMO_PATHS=$(grep -E "^squashfs-root/${SP}/(dynamo|ai_dynamo[^/]*)/?" /tmp/dynamo_full_list.txt | sed 's|^squashfs-root/||' | sed 's|/$||' | sort -u)
[[ -n "${AI_DYNAMO_PATHS}" ]] || { echo "ERROR: no dynamo entries found in image listing" >&2; exit 1; }
echo "${AI_DYNAMO_PATHS}"

echo "=== extracting python packages ==="
# shellcheck disable=SC2086
unsquashfs -f -q -d "${DEST}/rootfs" "${IMAGE}" ${AI_DYNAMO_PATHS}

echo "=== extracting nats-server (static go binary) ==="
unsquashfs -f -q -d "${DEST}/rootfs" "${IMAGE}" usr/bin/nats-server
"${DEST}/rootfs/usr/bin/nats-server" --version

echo "=== flattening into ${DEST} ==="
rm -rf "${DEST}/pkg" "${DEST}/bin"
mkdir -p "${DEST}/pkg" "${DEST}/bin"
cp -a "${DEST}/rootfs/${SP}/." "${DEST}/pkg/"
cp -a "${DEST}/rootfs/usr/bin/nats-server" "${DEST}/bin/nats-server"
chmod +x "${DEST}/bin/nats-server"
rm -rf "${DEST}/rootfs"

echo "=== sanity: dist-info versions ==="
ls -d "${DEST}"/pkg/ai_dynamo*dist-info "${DEST}"/pkg/dynamo >/dev/null
echo "=== EXTRACT DYNAMO PKG OK: $(du -sh ${DEST}/pkg | cut -f1) at ${DEST} ==="
