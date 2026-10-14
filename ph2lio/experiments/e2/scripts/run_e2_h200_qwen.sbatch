#!/usr/bin/env bash
# E2 8x H200 sweep — Qwen3.8-27B-FP8: DEPRECATED submission entry, kept only
# so existing muscle memory and references keep working. It now simply runs
# the resume variant (run_e2_h200_qwen_resume.sbatch), which skips any cell
# whose run dir already holds e2_metrics.json — the original no-resume body
# re-ran completed cells on resubmission. The sbatch headers live HERE
# (sbatch ignores the inner file's #SBATCH lines); the cell loop, env knobs,
# and the NODE_LOCAL mirror block live in the resume file as the single
# source of truth. Never launch both files at once — they are the same job.
#SBATCH --job-name=e2-qwen-h200
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-02    # Qwen arm node; weights seeded by download_qwen_h200_n02.sbatch (GLM arm runs on -01 in parallel)
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# The 8 dp replicas must not be squeezed onto NUMA node 0: each NUMA node
# has 96 CPUs (2x48c/96t), so any request <= 96 lands on one node only
# (the node grants 64-core jobs node-0-only affinity). 160 > 96 forces
# Slurm to span both nodes while leaving 32 CPUs for teammates.
#SBATCH --cpus-per-task=160
# Jobs on this account must not exceed 4 h; the resume loop in the body
# continues from the first unfinished cell on the next submission.
#SBATCH --time=04:00:00
#SBATCH --output=/shared/loipp/logs/e2_h200_qwen_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"

exec bash "${E2_DIR}/scripts/run_e2_h200_qwen_resume.sbatch"
