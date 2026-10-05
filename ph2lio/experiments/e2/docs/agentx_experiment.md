# E2: Schedule-policy comparison under the AgentX agentic workload

E2 measures how SGLang schedule policies behave under a realistic agentic
coding workload, using the official AIPerf `inferencex-agentx-mvp` scenario
with the SemiAnalysis Weka agentic-coding trace dataset. It replaces E1's
synthetic workloads (`sglang.benchmark.serving`) with trace-replayed
multi-turn, subagent-fan-out sessions.

Scope is intentionally narrow: **single GPU, single SGLang server, six
schedule policies**. No routing, load balancing, multi-node orchestration,
Docker production images, CI workflows, or dashboard ingestion.

## Workload

- **Scenario**: `inferencex-agentx-mvp` (AIPerf 0.13.0). Locks the benchmark
  rules: streaming on, `ignore_eos` on, cache-bust on first-turn prefix,
  agentic replay timing, no client-side input truncation, minimum benchmark
  duration 900 s (default 1800 s).
- **Dataset**: `semianalysis_cc_traces_weka_062126_256k`
  (HF: `semianalysisai/cc-traces-weka-062126-256k`), 393 v7 agentic-coding
  traces, 256K-capped: every request satisfies `input + output <= 256000`
  tokens. Public, no HF token needed. First-use download + reconstruction is
  cached under `~/.cache/aiperf/dataset_mmap` keyed by `--random-seed`.
- **Serving model**: user-maintained (default `Qwen/Qwen3.8-27B-FP8`,
  context 262144 >= the 256k dataset cap, so no truncation is needed).

## Concurrency semantics

`--concurrency N` in AIPerf/AgentX counts concurrent **session trees**, not
in-flight HTTP requests. Agentic sessions fan out subagents, so the
instantaneous HTTP request count can exceed N. Run each point separately
via `run_e2.sh policy <policy> <conc>`.

## Policies (exactly six)

`fcfs`, `lpm`, `hrrn`, `dfs-weight`, `lof` — stock SGLang `--schedule-policy`
implementations, unmodified — plus `boost`, the gamma-boosted arrival policy
added in this branch (opt-in; no default behavior changes). Nothing in E2
patches policy implementations at runtime; if a policy is missing or behaves
differently in the installed SGLang version, the run stops and the issue is
reported rather than substituted.

**Baseline description**: *Default SGLang serving configuration with
`schedule-policy=lpm` override.* LPM is **not** SGLang's global default
(the default policy is `fcfs`); it is the baseline because it is the
cache-aware policy E1 identified as starvation-prone (cold-cache requests
wait behind warm-prefix ones; empirically ~1.55x TTFT penalty in E1 W3).
HRRN exists to address exactly that weakness.

Note: SGLang's LPM/HRRN/BOOST implementations silently fall back to FCFS when the
waiting queue exceeds 128 requests (`_determine_active_policy`,
`python/sglang/srt/managers/schedule_policy.py`; boost joins the same
fallback for sort-cost parity); at the sweep concurrencies
this fallback is not expected to trigger, but queue-length in server logs
should be checked if a run looks FCFS-like.

## Boost arm: gamma-boosted arrival (UniBoost)

`boost` implements the gamma-Boost waiting-queue policy (UniBoost, Li et
al., ICML 2026). Each waiting request is scored

    phi(r) = A(r) - b_gamma(q(u(r)))

and the queue is served in ascending phi. `A(r)` is the token-clock stamp
at the request's FIRST queue entry (stamped once; survives retraction
re-queueing, unlike HRRN's re-stamped arrival). `u(r)` is uncached work:
input + generated tokens minus matched prefix tokens. `q()` quantizes work
onto the geometric ladder {k, 2k, 4k, ...} at `k = --boost-bin-size`. The
boost itself is

    b_gamma(w) = -ln(1 - exp(-gamma * w)) / gamma    (expm1-stable)

A newcomer can only jump ahead of an older request by its boost, which is
bounded by `b_gamma(k)` clock-tokens — requests older than that bound are
unreachable by newcomers. Bounded displacement makes anti-starvation a
guaranteed fallback rather than an emergent property.

- **gamma direction**: larger gamma -> stronger anti-starvation,
  converging to FCFS; gamma -> 0+ favors short uncached work (LAS-like).
  gamma is the locality-vs-tail dial.
- **Flags**: `--schedule-policy boost`, `--boost-gamma` (default 1e-4, the
  paper's setting), `--boost-bin-size` (default 256). Validated at startup
  (`boost_gamma > 0`, `boost_bin_size >= 1`); boost degrades to FCFS with a
  warning when prefix caching is disabled.
- **Sweep**: core gamma {5e-5, 1e-4, 2e-4} at k=256, concurrency ~8;
  optional k in {1024, 2048} at gamma=1e-4. Boost run dirs carry the arm in
  the label: `c${conc}_g${gamma}_k${k}`; two boost runs with different
  (gamma, k) never reuse the same server (the reuse key checks
  /server_info for the full policy spec).
- **Calibration**: the starvation bound in wall-clock is approximately
  `b_gamma(k) / R` where R is the sustained prefill token rate; measure R
  from an fcfs run before the sweep. At k=256: gamma=5e-5 -> ~87.3k tokens,
  1e-4 -> ~36.8k, 2e-4 -> ~15.0k, 5e-4 -> ~4.2k.

## Metrics

Per-request (AIPerf `profile_export.jsonl`, records export level):

- **TTFT / TPOT / E2E mean + P95** — `time_to_first_token`,
  `request_latency`; TPOT derived per request as `(E2E - TTFT) / (OSL - 1)`.
  ITL is deliberately not used as the primary decode metric.
- **Jain fairness index on TTFT**, cold/warm TTFT split via
  `usage_prompt_cache_read_tokens` (server-reported; requires
  `--use-server-token-count --per-chunk-usage`, which enables SGLang's
  continuous usage stats), Spearman cache/TTFT correlation.

Aggregate (AIPerf `profile_export_aiperf.json`):

- **RPM / TPM** — request and output-token throughput.

Server-side (scraped Prometheus, `--server-metrics`, requires SGLang
`--enable-metrics`):

- **Queue waiting time mean / P95 / P99** — from the
  `sglang:queue_time_seconds` histogram delta across the profiling phase.
- **Starvation Rate** — share of requests whose queue wait exceeds a
  configurable threshold (default 10 s; `--starvation-threshold`). This is
  an E2-defined metric, not an official AgentX metric.
- **Service Difference** — P99 - P50 of queue wait (tail-vs-typical service
  disparity). Also E2-defined.

Per-request scheduler-side (`--log-schedule-eval-path`; one JSON line per
finished request from the scheduler process — every E2 policy/smoke run
requests it, so new run dirs carry a `schedule_eval.jsonl` snapshot plus
the line offset in `run_meta.json` that isolates this run's records on a
reused server):

- **queue_wait_per_request block** (`e2_metrics.json`) — real per-request
  queue wait (first admission minus first queue entry; both stamps are
  stamp-once, so the wait spans retraction cycles): cold/warm split on the
  FIRST-admission prefix match (`matched_prefix_tokens_first_admission == 0`
  vs `> 0`), per-class mean/P95/P99/max queue wait and per-class starvation
  rate at the same threshold, cold-warm queue-wait gap (P99 + mean),
  arrival-order starvation slope (OLS of queue wait on arrival rank within
  the cold class, with R^2; slope ~0 = no cohort trend), first-attempt
  cache hit rate (token-weighted), retraction count (server counter delta,
  else per-interval gauge sum, else eval-log sum), and the
  waiting-queue-length gauge. Supersedes the TTFT-gap proxy for runs that
  have this log; the proxy stays, clearly labeled, for runs without it.
- **run block** — policy, concurrency, duration, seed, `boost_gamma`,
  `boost_bin_size` copied from `run_meta.json` (cross-checked against
  `/server_info` by the runner before the benchmark starts).

### Documented observability limits (not approximated)

1. **Queue-wait Max** is not derivable from the histogram: the +Inf bucket
   is unbounded. (The per-request block reports a real max when the
   schedule-eval log is present.)
2. **Cold-vs-Warm Queue-Wait Gap** is not computable from the server
   histogram alone: queue wait exists only as an aggregate; there is no
   per-request queue wait to join with per-request cache classification.
   With `--log-schedule-eval-path` it is measured directly
   (`queue_wait_per_request`). Without the log, E2 reports the clearly
   labeled proxy `cold_warm_ttft_gap_p99_ms_proxy` (per-request TTFT
   includes network + queue + prefill), never presented as queue wait.
3. Queue-wait percentiles are bounded by Prometheus bucket resolution
   (linear interpolation within buckets). The per-request block is not
   bucket-bound.
4. AIPerf has no queue-time metric of its own; all queue-wait numbers come
   from the SGLang histogram or the schedule-eval log.

Deliberately excluded (per E2 design): ITL as primary decode metric,
waiting-time CDF, normalized waiting time, generic SLO/goodput.

## Procedure

`run_e2.sh` is the single entrypoint and manages the server lifecycle
itself: it starts `scripts/start_server.sh <policy>` (all pinned serving
flags live there) and reuses a running server only when its full policy
spec matches — `schedule_policy`, plus `boost_gamma`/`boost_bin_size` and
the schedule-eval log path for boost, all read from `/server_info`;
anything else is restarted first.

```bash
scripts/run_e2.sh policy lpm 4        # one official point (repeat per conc)
scripts/run_e2.sh policy boost 8      # boost at the paper default gamma/k
BOOST_GAMMA=2e-4 scripts/run_e2.sh policy boost 8   # sweep arm
scripts/run_e2.sh sweep 4 8           # all six policies at the given concs
sbatch --array=0-5 scripts/run_e2.sh array   # Slurm: one policy per task
python3 analysis/agentx_metrics.py results/agentx/boost/c8_g1e-4_k256_<ts>
```

Smoke test (pipeline validation only — 300 s duration + 16-entry subset
violates the scenario minimum, so it passes `--unsafe-override` and is
stamped not submission-valid; never used for comparison):

```bash
SMOKE_POLICY=boost scripts/run_e2.sh smoke
```

## Next steps after the sweep

1. Core matrix {lpm, fcfs, boost} x concurrencies {4, 8, 16}, 1800 s each,
   seed 42 (add hrrn at concurrency 8 if budget allows); optional repeat
   seeds {42, 43, 44} for the headline lpm-vs-boost pair.
2. Gamma sweep {5e-5, 1e-4, 2e-4} at k=256, concurrency ~8; calibrate the
   starvation bound first from an fcfs run (R = sustained prefill tok/s;
   bound ~ b_gamma(k)/R).
3. Compute `e2_metrics.json` per run; compare starvation rate, queue-time
   P99, cold/warm queue-wait gap + per-class starvation rate + slope, and
   cold-class TTFT P99/max across policies at matched concurrency.
   Guards: cache hit rate loss <= ~2 pts vs lpm; RPM / output-token
   throughput within +/-5%; TPOT P95 within +10%.
4. Check server logs for LPM/HRRN/BOOST -> FCFS fallback events
   (queue > 128).
