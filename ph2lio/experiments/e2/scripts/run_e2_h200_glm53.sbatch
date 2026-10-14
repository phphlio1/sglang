#!/usr/bin/env bash
# E2 8x H200 sweep — GLM-5.3 W4AFP8: 5 policies x concurrency {16,32,64},
# 1 run x 30 min per cell, fresh boot per cell, seed 42.
# ~15 cells x (400 GB weight load + JIT + 30 min + drain) ≈ 12-13 h.
#
# Prerequisites (run on the node first, NOT from the login node):
#   - repo copy at ${BENCH}/sglang (this branch) + server venv
#     (venv-server, pip install -e python) + aiperf venv (venv-aiperf)
#   - GLM-5.3-W4AFP8 checkpoint (~399.7 GB) placed on node-local /mnt
#     (GLM_MODEL_PATH in start_server_h200_glm53.sh must match)
#   - HF_HOME populated with the agentx dataset; the client tokenizer
#     (PhalaCloud/GLM-5.3-W4AFP8) cached or reachable
#   - node driver supports the cu13 wheel line (probe first)
#   - scripts/start_server_h200_glm53.sh smoke-tested (run_e2.sh smoke)
#SBATCH --job-name=schedPolicies-glm53-h200
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# The server pins GPUs 4-7 to NUMA node 1 (--numa-node 0 0 0 0 1 1 1 1).
# Each NUMA node has 96 CPUs (2x48c/96t), so any request <= 96 lands on one
# node only and the node-1 binding is skipped at boot (2623, with 64). 160
# > 96 forces Slurm to span both nodes (measured demand is only ~16-20
# cores total, ~155% per scheduler rank)
#SBATCH --cpus-per-task=160
# GLM's 8 ranks allocate ~1.23 TB of host hicache/indexer pools (8 x 128 GB
# KV + 8 x 25.8 GB DSA); measured peak RSS 146.5 GB/rank (sacct 2992) ~= 1.17
# TB across 8 ranks. --mem=0 (whole node) blocked every other job: this
# cluster schedules memory as a consumable (CR_Core_Memory), so a whole-node
# grant excludes teammates even with 32 CPUs left free. 1600G covers the
# measured demand with ~370 GB headroom and leaves ~430 GB for teammates.
#SBATCH --mem=1600G
# Jobs on this account must not exceed 4 h. A mirrored-boot cell = boot
# (~7 min from /mnt) + warmup + 30 min bench + grace ~= 50-65 min, so a
# submission completes ~4-5 cells; the wall kills the next cell's boot
# mid-flight and the resume loop below re-runs it on resubmit.
#SBATCH --time=04:00:00
#SBATCH --output=/shared/loipp/logs/schedPolicies-glm53-h200_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"

# --- node-local execution mirror (built by mirror_from_login.sh + ----------
# --- mirror_node_side.sh, run from the login node) ----------------------------
# venvs, the uv base interpreter, the sglang import tree, and the HOME
# compile caches live on this node's /mnt after the mirror job: imports and
# JIT-cache verification stop touching the ~1.4 MB/s NFS and survive
# teammates' page-cache eviction. Falls back to the /shared paths (slow)
# whenever the mirror is absent.
NODE_LOCAL="${NODE_LOCAL:-/mnt/loipp}"
if [ -x "${NODE_LOCAL}/venv-server/bin/python3" ]; then
  export SERVER_PYTHON="${NODE_LOCAL}/venv-server/bin/python3"
  export AIPERF="${NODE_LOCAL}/venv-aiperf/bin/aiperf"
  export CUDA_HOME="${NODE_LOCAL}/venv-server/lib/python3.11/site-packages/nvidia/cu13"
  export HOME="${NODE_LOCAL}/home"
else
  echo "=== node-local mirror absent on $(hostname); running from /shared (slow NFS) ==="
  export HOME="${BENCH}/home"
fi
mkdir -p "${HOME}"

# --- run_e2.sh knobs (pinned for this campaign) -----------------------------
export SGLANG_URL="http://127.0.0.1:30000"
# MODEL is the name clients request: the server's --served-model-name, not
# the weight path. TOKENIZER points at the local checkpoint dir so the
# aiperf client never touches the network.
export MODEL="zai-org/GLM-5.3"
export TOKENIZER="/mnt/hf_hub_cache/GLM-5.3-W4AFP8"
export OUTPUT_DIR="${E2_DIR}/results/8x-h200/glm53/agentx"
export SERVER_SCRIPT="${E2_DIR}/scripts/start_server_h200_glm53.sh"
# Cold boot measured ~2.5-3 h (2758: prefill-graph JIT to 05:16, still short
# of ready at the 7200 s cutoff — no crash, just slow; /shared HOME JIT
# caches cut later boots). Per-cell boot: keep the window generous; timeout
# only fails if actually exceeded.
export HEALTH_TIMEOUT=12000
# Campaign regime (mentor, 2026-10-08): fresh boot per cell (cold radix
# cache); client drain capped at 360 s (user-set; requests in flight
# longer than that at close are cut from metrics).
export FORCE_RESTART=1
export GRACE_PERIOD=360
export SEED=42
export DURATION=1800
# Boost arm on this hardware: adaptive gamma (the RTX static 0.09 was
# tuned for the 1x-RTX regime; gamma-Ada self-tunes from the admission
# wait tail, gamma0 = clock default). Veto back to the RTX arm:
# BOOST_ADAPTIVE_GAMMA=0 + BOOST_GAMMA=0.09.
export BOOST_ADAPTIVE_GAMMA=1

# Import-cache pre-read DISABLED: under the 4 h job cap the arithmetic
# inverts. One cold boot costs ~2h40m without the pass; the pass costs
# ~2.5h alone and only saves ~1h40m of boot — net loss for a single boot,
# and with the pass the cell finishes past the 4 h wall (killed mid-bench,
# no e2_metrics.json). The pass only paid off when one long job amortized
# it over many reboots. If the node page cache survived since the last
# job, boots are fast anyway without the pass. Durable fix (not done):
# copy venv+repo to node-local /mnt once per node.
# Overrideable: submit with WARM_CACHE_PATHS=<paths> (via --export) to re-enable.
export WARM_CACHE_PATHS="${WARM_CACHE_PATHS:-}"

# --- client-side caches (on /shared so every job reuses them) ---------------
export HF_HOME="${BENCH}/hf"
export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache"

cd "${E2_DIR}"

# Resume-capable sweep: a cell is complete iff any of its run dirs holds
# e2_metrics.json (written at the end of mode_policy in run_e2.sh). A cell
# cancelled mid-run has no e2_metrics.json and is re-run on resubmit, so an
# abrupt scancel (e.g. a colleague needs the node) never loses completed
# cells. The loop lives here, not in run_e2.sh: run_e2.sh must not be edited
# while other jobs execute it, and sbatch snapshots this file at submit time.
# || echo keeps one failed cell (e.g. health timeout) from killing the rest.
SWEEP_CONCS=(16 32 64)
# dfs-weight runs FIRST: it is the prod config's own --schedule-policy, so
# those cells double as the production-config profile cells; boost stays last
# (waits on the teammate's adaptive-gamma code).
for p in dfs-weight fcfs-match lpm hrrn boost; do
  for c in "${SWEEP_CONCS[@]}"; do
    # Mirrors run_label in run_e2.sh; BOOST_ADAPTIVE_GAMMA=1 is pinned above,
    # so the boost label is always the adaptive spelling.
    label="c${c}"
    [[ "${p}" == "boost" ]] && label="c${c}_gada_${BOOST_CLOCK:-wall}_k${BOOST_BIN_SIZE:-256}"
    if ls "${OUTPUT_DIR}/${p}/${label}_"*/e2_metrics.json >/dev/null 2>&1; then
      echo "=== E2 resume: skip completed ${p} c${c} ==="
      continue
    fi
    echo "=== E2 sweep point: policy=${p} concurrency=${c} ==="
    bash scripts/run_e2.sh policy "${p}" "${c}" \
      || echo "=== E2 cell failed: ${p} c${c} (will retry on resubmit) ==="
  done
done
