#!/usr/bin/env bash
# E2 8x H200 sweep — Qwen3.8-27B-FP8: 5 policies x concurrency {32,64,128},
# 1 run x 30 min per cell, fresh boot per cell, seed 42.
# 15 cells x (boot + 30 min + drain): first boots on the cold NFS venv
# measured ~9 min/replica serial (2751), so the 16 h estimate could be cut
# short mid-sweep; 20 h is only a ceiling (cells finish, job ends early).
#
# Prerequisites (run on the node first, NOT from the login node):
#   - repo copy at ${BENCH}/sglang (this branch) + server venv
#     (venv-server, pip install -e python) + aiperf venv (venv-aiperf)
#   - Qwen3.8-27B-FP8 checkpoint (~27 GB) flat at /mnt/hf_hub_cache/
#     Qwen3.8-27B-FP8 (download_qwen_h200.sbatch; permission granted
#     2026-10-09); HF_HOME needs only the agentx dataset
#   - node driver supports the cu13 wheel line (probe first)
#   - scripts/start_server_h200_qwen.sh smoke-tested (run_e2.sh smoke)
#SBATCH --job-name=e2-qwen-h200
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-02    # Qwen arm node; weights seeded by download_qwen_h200_n02.sbatch (GLM arm runs on -01 in parallel)
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# All 192 CPUs so the 8 replicas are not squeezed onto NUMA node 0 (the
# node grants 64-core jobs node-0-only affinity).
#SBATCH --cpus-per-task=192
#SBATCH --output=/shared/loipp/logs/e2_h200_qwen_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"

# Compute nodes inherit HOME=/home/loipp from the submitter, but /home is
# login-local — on the node that path doesn't exist and its parent is
# root-owned, so torch dynamo/inductor's cache bootstrap (makedirs under
# $HOME at import time) dies with EACCES before the server ever boots.
# HOME on /shared also gives torch/triton compile caches job-to-job reuse.
export HOME="${BENCH}/home"
mkdir -p "${HOME}"

# --- run_e2.sh knobs (pinned for this campaign) -----------------------------
export SGLANG_URL="http://127.0.0.1:30000"
export MODEL="Qwen/Qwen3.8-27B-FP8"
# TOKENIZER points at the local checkpoint dir so the aiperf client never
# touches the network.
export TOKENIZER="/mnt/hf_hub_cache/Qwen3.8-27B-FP8"
export OUTPUT_DIR="${E2_DIR}/results/8x-h200/qwen/agentx"
export SERVER_SCRIPT="${E2_DIR}/scripts/start_server_h200_qwen.sh"
export AIPERF="${BENCH}/venv-aiperf/bin/aiperf"
# 8 replicas import the NFS venv independently at boot (see 2623's
# > 60 min cold-boot measurement on the GLM arm).
# Per-cell reboot: 8 replicas each import the NFS venv and (first time)
# JIT+graph-capture ~9 min serially (2751); /shared HOME caches cut later
# boots. Keep the window generous — timeout only fails if actually exceeded.
export HEALTH_TIMEOUT=9000
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

# --- client-side caches (on /shared so every job reuses them) ---------------
export HF_HOME="${BENCH}/hf"
# Separate mmap cache from the -01 jobs: both arms' clients may build it
# from the dataset concurrently on their first run.
export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache_n02"

cd "${E2_DIR}"
bash scripts/run_e2.sh sweep 32 64 128
