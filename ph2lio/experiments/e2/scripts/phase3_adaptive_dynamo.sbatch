#!/usr/bin/env bash
# E2 boost campaign — Phase 3: adaptive gamma (gamma-Ada, UniBoost Eq. 9) on
# the dynamo arm. BOOST_ADAPTIVE_GAMMA=1, gamma0=0.09, wall clock, k=256.
# J=60 vs J=120: the 2026-10-19 RTX audit (plan §5.5) found that under heavy
# overload the raw E2E-tail fit always sits below the jump floor, so gamma
# pins at gamma_floor(J) — the controller is inert and J is the effective
# knob (J=60 vs J=120 = two floor levels). With ~8x H200 prefill rate,
# tau = k/rate shrinks ~8x, so the same J gives a higher floor and the
# lighter load may let the raw fit exceed it: the controller may actually
# be active here. Concurrency = PHASE3_CONC (phase-0 band).
#
# Paired seeds (SEEDS_LIST, default "42"): seed variance dwarfs the J
# effect in RTX replicates, so multi-seed (--export=SEEDS_LIST="42 43 44")
# is recommended for the J comparison. Seed-major order.
#
# Gates: engine log carries the "boost adaptive gamma: gamma=..." trajectory
# (schedule_policy.py:504; logs the raw pre-floor fit, its tail source
# "overall"/"short", the raw width, and the floor; needs >=128 finished
# latencies) and schedule_eval.jsonl records show the live gamma moving.
#SBATCH --job-name=e2boost-p3-adaptive
#SBATCH --partition=hardware-hcm
#SBATCH --nodelist=hgx-h200-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# Must stay >96: the server binds GPUs 4-7 to NUMA node 1 (--numa-node 0 0 0 0
# 1 1 1 1); <=96 CPUs lands the whole allocation on one NUMA node and the
# binding is skipped at boot (smoke 2623 died at 64). Measured demand is
# ~16-20 cores — the count is purely topological. 128 since 2026-10-19.
#SBATCH --cpus-per-task=128
#SBATCH --mem=1600G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/loipp/logs/e2boost_p3_adaptive_%j.log

set -euo pipefail

BENCH="${BENCH:-/shared/loipp}"
E2_DIR="${BENCH}/sglang/ph2lio/experiments/e2"
PHASE3_CONC="${PHASE3_CONC:-16}"   # from the phase-0 gate
SEEDS_LIST="${SEEDS_LIST:-42}"     # paired seeds, seed-major (see header)

NODE_LOCAL="${NODE_LOCAL:-/mnt/loipp}"
if [ -x "${NODE_LOCAL}/venv-dynamo/bin/python3" ]; then
  export AIPERF="${NODE_LOCAL}/venv-aiperf/bin/aiperf"
  export HOME="${NODE_LOCAL}/home"
else
  echo "=== node-local mirror absent on $(hostname); running from /shared (slow NFS) ==="
  export HOME="${BENCH}/home"
fi
mkdir -p "${HOME}"

export SERVER_SCRIPT="${E2_DIR}/scripts/start_dynamo_stack.sh"
export SGLANG_URL="http://127.0.0.1:8888"
export MODEL="zai-org/GLM-5.3"
export TOKENIZER="/mnt/hf_hub_cache/GLM-5.3-W4AFP8"
export RESULTS_ROOT="${E2_DIR}/results/glm53/agentx_dynamo"
export OUTPUT_DIR="${RESULTS_ROOT}/policies_v2/phase3_adaptive"
export AIPERF_DATASET_MMAP_CACHE_DIR="${BENCH}/aiperf_mmap_cache"
export READY_ON_MODEL=1
export SERVER_PROC_PATTERN='dynamo\.sglang'
export SERVER_METRICS_URLS="127.0.0.1:8081"
export SGLANG_ALLOW_PTRACE=1
export HEALTH_TIMEOUT=3600
export FORCE_RESTART=1
export GRACE_PERIOD=360
export SEED=42
export DURATION=1800
export STOP_SERVER=always
export HF_HOME="${BENCH}/hf"
export BOOST_ADAPTIVE_GAMMA=1
export BOOST_GAMMA=0.09          # gamma0
export BOOST_CLOCK=wall
export BOOST_BIN_SIZE=256
export IN_BATCH_PREFIX_CACHING_MAX_PIN_S=60

cd "${E2_DIR}"

# run_label: adaptive boost = c<conc>_gada_<clock>_k<bin>_j<jump>
seed_done() {  # $1 = policy, $2 = label, $3 = seed: completed cell with this seed?
  local d
  for d in "${OUTPUT_DIR}/$1/$2"_*/; do
    [[ -f "${d}e2_metrics.json" && -f "${d}run_meta.json" ]] || continue
    grep -q "\"seed\": $3," "${d}run_meta.json" && return 0
  done
  return 1
}

read -ra SEEDS <<< "${SEEDS_LIST}"
for seed in "${SEEDS[@]}"; do
  for j in 60 120; do
    label="c${PHASE3_CONC}_gada_wall_k256_j${j}"
    if seed_done boost "${label}" "${seed}"; then
      echo "=== E2 resume: skip completed boost ${label} seed=${seed} ==="
      continue
    fi
    echo "=== E2 phase3 adaptive cell: J=${j} c${PHASE3_CONC} seed=${seed} ==="
    BOOST_ADAPTIVE_MAX_JUMP_S="${j}" SEED="${seed}" bash scripts/run_e2.sh policy boost "${PHASE3_CONC}" \
      || echo "=== E2 cell failed: boost ${label} seed=${seed} (will retry on resubmit) ==="
    find "${OUTPUT_DIR}" -name e2_metrics.json | while read -r m; do
      d="$(dirname "${m}")"
      # `if` (not `[[ ]] && gzip`): a false guard as the loop's last status
      # returns 1 and, under set -euo pipefail (find | while), kills the whole
      # driver after the last cell — job 4391 died exactly this way.
      if [[ -f "${d}/server_metrics_export.jsonl" ]]; then gzip -f "${d}/server_metrics_export.jsonl"; fi
    done
  done
done
echo "=== PHASE3 DONE (check the gamma trajectory before phase 4) ==="
