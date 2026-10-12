#!/usr/bin/env bash
# Login-side driver for the node-local /mnt execution mirror.
#
# WHY this shape: compute nodes read /shared at ~0.1-1.4 MB/s (per-file NFS
# round-trip latency dominates), so any copy that walks or reads the SOURCE
# from the node spends hours in metadata before moving data — the first
# mirror design (mirror_to_mnt.sbatch: du-based tree split + 12 rsync
# streams from the node) stalled at 0 MB for 11+ min on venv-server and
# projected a 3-5 h du walk alone. The login node reads /shared at
# ~8.4 MB/s and pipes to the node at ~26 MB/s (both measured 2026-10-12),
# so one tar streamed over an srun pipe moves the whole ~10.6 G in
# ~25-30 min. Resumable the blunt way: if it dies, rerun — tar overwrites.
#
# Usage (from the login node, one node at a time):
#   nohup bash mirror_from_login.sh hgx-h200-01 \
#     > /shared/loipp/logs/e2_mirror_01_fromlogin.log 2>&1 &
# Do NOT run two drivers against the same node at once. For -02, wait until
# the node is un-drained. Success prints "=== MIRROR OK <host> ===" (from
# the node-side script); any failure exits non-zero with a MIRROR FAIL line.
set -euo pipefail

NODE="${1:?usage: mirror_from_login.sh <hgx-h200-01|hgx-h200-02>}"
BENCH=/shared/loipp
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo "[$(date +%H:%M:%S)] mirror driver start: /shared/loipp -> ${NODE}:/mnt/loipp"
echo "[$(date +%H:%M:%S)] trees: uv-python venv-server home venv-aiperf sglang/python (~10.6G)"

tar -C "${BENCH}" -cf - uv-python venv-server home venv-aiperf sglang/python \
  | srun -p hardware-hcm -w "${NODE}" --cpus-per-task=8 --mem=16G \
      --time=03:00:00 bash "${SCRIPTS}/mirror_node_side.sh"

echo "[$(date +%H:%M:%S)] mirror driver done"
