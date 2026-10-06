# E2 Environment Inspection (AgentX / Weka workload)

Date: 2026-10-05. Inspection only — no server or repo configuration was modified.

## Host / GPU

| Item | Value |
|---|---|
| GPU | 1x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm_120, compute 12.0) |
| VRAM | 97,887 MiB total (~2 MiB in use at inspection — idle) |
| OS | Linux 5.15.0-179-generic |
| Disk | 155 GB free on `/` (dataset + caches need well under this) |

Single-GPU only. No routing, load balancing, or multi-node orchestration is in
scope for E2.

## SGLang

| Item | Value |
|---|---|
| Repo | `/home/user1111/sglang` (git, branch `main`) |
| Commit | `b1bbd74f287f13ed1276b0403a01ebb55c597e93` (2026-10-05) |
| Tag | `gateway-v0.3.1-10661-gb1bbd74f28` |
| Installed pkg | `0.5.20.dev746+g537ac52c5` (conda env `sglang`, editable install) |
| Python (server env) | 3.11.16 (`/home/user1111/miniconda3/envs/sglang/bin/python3`) |
| Endpoint | `http://127.0.0.1:30000` — **NOT reachable at inspection time** (no `sglang.launch_server` process, nothing listening on 30000-30009) |

The server is started/stopped manually by the user. Benchmark scripts must
take the endpoint as a variable and fail cleanly when it is down. The
server-side model/config is user-maintained and out of benchmark scope.
E2 serving config (set 2026-10-05, user-specified): `Qwen/Qwen3.8-27B-FP8`,
TP=1, flashinfer attention backend, fp8_e4m3 KV cache, mem-fraction 0.80,
context 262144, chunked prefill 8192, EAGLE spec decoding (steps 3 / topk 1 /
draft tokens 4, bundled MTP head from the target checkpoint),
`--enable-linear-replayssm-spec`, `--mamba-full-memory-ratio 1.5`,
hierarchical cache (hicache 64), qwen3 reasoning/tool-call parsers, plus
`--enable-metrics` (required for E2's queue-time histogram). Blackwell
env: `SGLANG_ENABLE_JIT_DEEPGEMM=0`, `CUDA_HOME` = conda `nvidia/cu13`
(nvcc 13.4 first on PATH), `.cuda_shim` on `LIBRARY_PATH`/`LD_LIBRARY_PATH`.

## Python tooling (installed in Step 2)

| Item | Status |
|---|---|
| `uv` | Installed: `~/.local/bin/uv` 0.12.23 |
| `aiperf` | Installed: 0.13.0 in isolated venv `e2/.venv/` (Python 3.12), binary at `e2/.venv/bin/aiperf` |
| pip / PyPI access | Working (`https://pypi.org` reachable, HTTP 200) |

The `inferencex-agentx-mvp` scenario is recognized by the installed aiperf
(verified: flag-conflict error on a probe invocation, not an
unknown-scenario error).

## Hugging Face

| Item | Status |
|---|---|
| `HF_TOKEN` | Not set (no authenticated `whoami`) |
| Dataset `semianalysisai/cc-traces-weka-062126-256k` | **Public — accessible without a token** (verified via `HfApi.dataset_info`) |

No gated-model or authenticated pull is required for the pinned public dataset.

## Existing experiment code (`ph2lio/experiments/`)

- `e1/` — complete. Synthetic-workload sweep (5 workloads x 7 policies x 5
  serving modes, `sglang.benchmark.serving` client), with the starvation
  metric pipeline in `e1/analysis/starvation.py` (Jain fairness, CV,
  cold/warm TTFT split, Spearman cache correlation — pure functions).
- `e2/` — at inspection contained stale copies of E1 files
  (`matrix.py`, `runner.py`, `RUN_GUIDE.md`, `analysis/`), carried over from
  the E1 Qwen3.8-27B-FP8 run. These do not apply to E2 (synthetic workloads,
  `sglang.benchmark.serving` client) and were duplicates of the originals in
  `e1/`, so they were deleted. Only the reusable pure metric functions were
  kept, as `e2/analysis/starvation.py`.

## Benchmark client (AIPerf, installed in Step 2)

- AIPerf (NVIDIA, `ai-dynamo/aiperf`, formerly GenAI-Perf) supports the
  `inferencex-agentx-mvp` scenario: it locks the benchmark rules, rejects
  conflicting flags, and stamps `submission_valid` on outputs.
- Pinned dataset loader: `--public-dataset semianalysis_cc_traces_weka_062126_256k`
  (256K-capped variant: every surviving request satisfies
  `input_tokens + output_tokens <= 256000`).
- Concurrency semantics: `--concurrency N` counts concurrent session *trees*
  (subagent fan-out means instantaneous HTTP request concurrency can exceed N).
- Minimum benchmark duration: 900 s enforced by the scenario; smoke tests must
  use `--unsafe-override` (stamps output as not submission-valid) or a small
  `--num-dataset-entries` subset.
- Dataset reconstruction cache: `~/.cache/aiperf/dataset_mmap`, keyed by
  `--random-seed`. Cold reconstruction can exceed the default 300 s config
  timeout — raise `AIPERF_DATASET_CONFIGURATION_TIMEOUT` /
  `AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT` (~1800 s) for the first run.

## Deliverables created (Steps 4-6, 10, 12)

| Path | Purpose |
|---|---|
| `scripts/run_e2.sh` | Single entrypoint: `smoke` / `policy <p> [c]` / `sweep` / `array` (Slurm) modes; manages server lifecycle, runs AIPerf, computes metrics |
| `scripts/start_server.sh` | User-editable server start helper (policy arg); all serving config lives here |
| `analysis/agentx_metrics.py` | E2 metrics from AIPerf exports (reuses `analysis/starvation.py` pure functions) |
| `results/agentx/{fcfs,lpm,hrrn,dfs-weight,smoke}/` | Per-policy run output layout (see `results/agentx/README.md`) |
| `docs/agentx_experiment.md`, `docs/RUN.md` | Experiment design; short how-to-run guide |

## Known constraints to watch

1. **Context length**: the 256K dataset contains requests up to 256k tokens.
   The served model's context window may be smaller. The scenario forbids
   client-side truncation (`--synthesis-max-isl` is rejected); the server is
   expected to surface its own context errors, which AIPerf tracks.
2. **Server down at inspection**: scripts must health-check before running.
3. **Baseline description**: "Default SGLang serving configuration with
   `schedule-policy=lpm` override." (LPM is *not* SGLang's global default —
   the default policy is `fcfs`.)
4. **RESOLVED (2026-10-05)**: the conda `sglang` env had drifted behind repo
   HEAD (b1bbd74f28). Realigned: sglang-kernel 0.4.9, flashinfer-python
   0.7.0.post1 (required by `--attention-backend flashinfer`), torch 2.14.1,
   torchvision 0.29.1, sgl-deep-ep/-gemm .post1; editable sglang refreshed to
   `0.5.20.dev1833+gb1bbd74f2`. Residual version drift (transformers
   5.12.1 vs 5.17.0, tokenizers, xgrammar, torchcodec) is non-fatal for this
   text-only workload — only benign multimodal import warnings at startup.
5. **RESOLVED (2026-10-05) — CCCL toolkit-check**: nvcc 13.4
   (`nvidia/cu13`) vs the cu13 pip headers' CUDART 13.0 makes CCCL's
   `cuda_toolkit.h` compat check reject JIT builds. Opt-out applied in both
   JIT pipelines: `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` in sglang's JIT
   default flags (`python/sglang/kernels/jit/utils/arch.py`) and via
   `FLASHINFER_EXTRA_CUDAFLAGS` in `scripts/start_server.sh` for
   flashinfer's JIT.

Smoke test passed 2026-10-05 (run `results/agentx/smoke/smoke_c1_20261005_184731`):
65 records, 0 errors, all exports present (incl. 1488 queue-time histogram
samples), max ISL 195,999 / max ISL+OSL 198,366 — within the 256000 dataset
cap and the 262144 context. submission_valid=false by design (smoke).
