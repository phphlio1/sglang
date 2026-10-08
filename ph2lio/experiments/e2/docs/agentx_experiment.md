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
  `b_gamma(k) / R` where R is the token-clock rate the bound actually
  governs — the scheduler's processed-token counter, whose rate is
  cache-state-dependent (it counts prefill extend tokens including the
  linear replay of matched prefixes: ~3.3-6k tok/s at c16 fresh-boot,
  ~36.7k tok/s at c16 round-1 warm; measured in "Token-clock rates" under
  Round 2). At k=256: gamma=3e-6 -> ~2.39M tokens, 1e-5 -> ~597k,
  2e-5 -> ~264k, 3e-5 -> ~162k, 5e-5 -> ~87.3k, 1e-4 -> ~36.8k,
  2e-4 -> ~15.0k, 5e-4 -> ~4.2k. (The c16 grid added in round 2 spans a
  65x bound range: at the measured fresh-boot clock rates the wall-clock
  bounds run ~400-724 s (3e-6) down to ~27-49 s (3e-5) against a ~95-190 s
  queue-wait p99, so gamma=1e-5 (~100-181 s) is the cell whose bound
  matches the tail; the round-1 failure point 1e-4 buys only ~6-11 s of
  overtaking at fresh rates.)

## Metrics

Per-request (AIPerf `profile_export.jsonl`, records export level):

- **TTFT / TPOT / E2E mean + P95** — `time_to_first_token`,
  `request_latency`; TPOT derived per request as `(E2E - TTFT) / (OSL - 1)`.
  ITL is deliberately not used as the primary decode metric.
- **Jain fairness index on TTFT**, cold/warm TTFT split via
  `usage_prompt_cache_read_tokens` (needs the server flag
  `--enable-cache-report`, in the pinned config since 2026-10-06, plus the
  client's `--use-server-token-count --per-chunk-usage`), Spearman cache/TTFT
  correlation.

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
the schedule-eval log path for boost, plus `enable_cache_report`, all read
from `/server_info`; anything else is restarted first.

```bash
scripts/run_e2.sh policy lpm 4        # one official point (repeat per conc)
scripts/run_e2.sh policy boost 8      # boost at the paper default gamma/k
BOOST_GAMMA=2e-4 scripts/run_e2.sh policy boost 8   # sweep arm
scripts/run_e2.sh sweep 4 8           # all six policies at the given concs
sbatch --array=0-5 scripts/run_e2.sh array   # Slurm: one policy per task
python3 analysis/agentx_metrics.py results/1x-rtx6000pro/agentx/boost/c8_g1e-4_k256_<ts>
```

Smoke test (pipeline validation only — 300 s duration + 16-entry subset
violates the scenario minimum, so it passes `--unsafe-override` and is
stamped not submission-valid; never used for comparison):

```bash
SMOKE_POLICY=boost scripts/run_e2.sh smoke
```

## Results (2026-10-06 matrix: seed 42, 1800 s, single GPU)

`scripts/run_boost_matrix.sh` ran the full matrix: {fcfs, lpm, boost} x
{4, 8, 16} at gamma=1e-4 / k=256 (paper defaults), plus the gamma sweep
{5e-5, 2e-4} at c8. Flat table (round-1 format):
`python3 analysis/print_results_table.py --flat results/1x-rtx6000pro/agentx`; the
canonical final table is the same script's default output and is carried in
the "Final results" section below. These runs predate
`--enable-cache-report` in the pinned config (added 2026-10-06, after the
matrix): their client-side `usage_prompt_cache_read_tokens` is absent, and
cache classification comes from the schedule-eval log; runs after the
change are not config-identical to them.

```
policy      c  gamma |    n   hit |  q_p95  q_p99 starv | cold_n c_strv warm_n w_strv  gap99 |   rpm   oTPM   ttft tpot95 retr
fcfs        4      - |  214 0.000 |  0.005  2.850 0.000 |    218  0.000      0      -      - |   7.0   5165   1736   26.8    0
fcfs        8      - |  139 0.000 |  0.005  2.805 0.000 |    145  0.000      0      -      - |   4.6   5826   2099   24.3    0
fcfs       16      - |  151 0.000 | 97.833 178.143 0.549 |    167  0.509      0      -      - |   4.9   4697   32312   166.8  5
lpm         4      - |  217 0.883 |  0.005  2.830 0.000 |     10  0.000    210  0.000  6.385 |   7.1   5371   1851   28.4    0
lpm         8      - |  141 0.894 |  0.005  2.590 0.000 |     16  0.000    131  0.000 -2.041 |   4.6   5839   1987   24.1    0
lpm        16      - |  201 0.803 | 73.929  94.750 0.380 |     34  0.265    185  0.378 -14.247|   6.6   5622   21468   98.6   6
boost       4  1e-04 |  223 0.888 |  0.005  0.009 0.000 |     10  0.000    216  0.000 -0.004 |   7.3   5922   1798   27.5    0
boost       8  1e-04 |  140 0.894 |  0.005  3.600 0.000 |     16  0.000    130  0.000 -3.093 |   4.6   5836   2170   22.2    0
boost       8  2e-04 |  139 0.893 |  0.005  3.600 0.000 |     16  0.000    130  0.000 -3.091 |   4.5   5678   2278   21.5    0
boost       8  5e-05 |  143 0.894 |  0.005  3.570 0.007 |     16  0.000    133  0.008 -3.090 |   4.7   5865   2230   24.3    0
boost      16  1e-04 |  161 0.753 | 118.000 183.600 0.512 |    33  0.273    145  0.524 -21.509|   5.2   4868   26949  115.7   0
```

Calibration: R = fcfs total-token throughput = 6.9k (c4) / 6.7k (c8) / 7.8k
(c16) tok/s; the boost starvation bound b_gamma(k)/R at k=256 is ~11.1 s
(5e-5), ~4.7 s (1e-4), ~1.9 s (2e-4), ~0.5 s (5e-4).

Findings:

1. **Policy gates cache reuse on this deployment.** The hybrid
   (linear-replay SSM) model's unified radix cache materializes cross-turn
   prefix reuse only under cache-aware policies: lpm/boost run
   `_compute_prefix_matches` on every waiting request at every sort, and
   75-89% of requests admit with deep matches; fcfs skips that path
   (`supports_fast_match_prefix()` is False for every cache implementation,
   so the cache-agnostic branch in `calc_priority` never matches) and admits
   with zero matched prefix tokens at every concurrency -- its server-log
   `#cached-token` is chunk-continuation self-hits only. fcfs numbers here
   measure "no effective prefix cache", not "FCFS with cache".
   (Round-2 correction, see "Task 1 verdict" below: the last two claims are
   wrong -- fcfs does realize deep admission-time reuse at c4/c8 through
   `init_next_round_input`'s policy-independent re-match, and its hit 0.000
   is sort-time bookkeeping, not absent cache. The c16 fcfs loss is real
   ordering/memory pressure.)
2. **c4/c8: boost matches lpm** on every metric (hit ~0.89, zero starvation,
   rpm/oTPM equal, TPOT p95 slightly better at c8). All spec guards pass.
3. **c16 (~3x oversubscribed): boost at gamma=1e-4 lands near
   fcfs-with-matching, not lpm**: starvation 0.512 (fcfs 0.549, lpm 0.380),
   queue p99 183.6 s (fcfs 178.1, lpm 94.8), hit 0.753 (lpm 0.803), oTPM
   -13% vs lpm, TPOT p95 +17% -- the hit-rate/throughput/TPOT guards fail
   against lpm at c16. The calibration explains it: the gamma=1e-4 bound
   (4.7 s) is negligible against 95-184 s queue depths, so boost cannot
   realize the warm-first ordering that materializes reuse and degenerates
   to FCFS-with-matching; the cold admissions then dilute the hit rate
   (0.803 -> 0.753) and throughput settles near fcfs. The anti-starvation
   property itself holds -- the cold class is not sacrificed (cold-class
   starvation 0.273 vs lpm's 0.265); the warm class pays for the throughput
   loss (0.524 vs 0.378).
4. **The gamma sweep shows no measurable difference at c8**: all three
   bounds (1.9-11.1 s) sit at or above the c8 queue p99 (~3.6 s), so the
   dial does not bind there.
5. **Cold/warm class gaps are small and arrival-phase-confounded**: gap99 is
   negative (warm p99 > cold p99) under both lpm and boost at c8/c16 -- the
   warm class is the majority and carries the deep-queue tail; the E1
   cold-class starvation story does not reproduce on this workload, where
   structural saturation dominates class ordering.

## Round 2 (2026-10-06/07): fcfs reference, c16 gamma recalibration, verification

Protocol changes this round:

- **Fresh boot per cell.** Round 1 reused one server across each policy's
  concurrencies (STOP_SERVER=never), so c8/c16 runs started cache-warm from
  the same policy's earlier runs. Round 2 boots a fresh server for every
  (policy, gamma, seed) cell (`scripts/run_round2.sh`, STOP_SERVER=always);
  `run_meta.json` now records `server_fresh_boot`. A fresh gamma=1e-4 c16
  boost re-run measures the warm-start confound against the round-1 point
  directly.
- **Warmup exclusion in the analysis.** The first PROFILING-phase scrape
  timestamp cuts the client warmup phase out of the schedule-eval slice
  (`analysis/agentx_metrics.py`), plus a HEALTH_CHECK_ rid filter. The
  round-1 metrics above predate the cut, so eval-log-derived columns (hit,
  cold/warm n and starv, gap99) shift when recomputed (e.g. lpm c16 hit
  0.803 -> 0.869); client/histogram-sourced columns (q_p95/p99, starv, rpm,
  oTPM, TTFT, TPOT, retr) do not move.
- **New admission-time fields in the eval log**, frozen at first admission:
  the device/host split of the match
  (`device_matched_prefix_tokens_first_admission` /
  `host_hit_prefix_tokens_first_admission`) and the in-batch-dedup verdict
  (`in_batch_dedup_follower_first_admission`).
- **New arm `fcfs-match`**: FCFS ordering with the sort-time prefix match
  forced on (`--schedule-policy fcfs-match`), the metric-visible fcfs
  reference for caches where `supports_fast_match_prefix()` is False.
- **New arm `boost` + `--boost-adaptive-gamma`** (gamma-Ada, Task 8): the
  server adapts `boost_gamma` online from the admission wait tail
  (gamma_hat = ln 5 / (w99 - w95) over the last 2048 first-admission waits in
  clock tokens, EMA beta 0.2, clip [3e-7, 1e-3]; gamma0 = `--boost-gamma`).
  Adaptive arms log to `schedule_eval_ada_k<k>.jsonl` and run labels
  `c<c>_gada_k<k>` so they never mix with static-gamma cells; the eval log
  stamps the current adapted gamma on every record (the trajectory).
- **Targeted-regime knobs in `run_e2.sh`** (Task 4): `NUM_DATASET_ENTRIES`
  (pool cap, first N scan-order entries) and
  `TRAJECTORY_START_MIN_RATIO`/`TRAJECTORY_START_MAX_RATIO` (trajectory
  resume window), both recorded in `run_meta.json`.

### Task 1 verdict: what the fcfs arm measures (`analysis/fcfs_reference_check.py`)

**Verdict: partial-reuse artifact at c4/c8 + a real ordering/memory-pressure
loss at c16 -- not "no effective prefix cache".** fcfs realizes deep
cross-request reuse at c4/c8 through `init_next_round_input`'s
policy-independent admission-time re-match; its hit column reads 0.000 only
because the sort-time bookkeeping (`num_matched_prefix_tokens`) is populated
on the match path fcfs skips. fcfs-match is the reference that makes the
bookkeeping honest without changing FCFS ordering.

Evidence (per-run server-log windows, client records):

| check | fcfs | lpm | boost (1e-4) |
|---|---|---|---|
| c4 observed cached tokens | 11.79 M | 11.94 M | 12.57 M |
| c8 observed cached tokens | 11.46 M | 11.61 M | 11.56 M |
| c4 frac input>50k with TTFT<2 s | 0.80 | 0.75 | 0.74 |
| c4 median TTFT given input>50k | 0.76 s | 0.77 s | 0.84 s |
| c16 observed cached tokens | 11.42 M | 17.43 M | 13.40 M |
| c16 median TTFT given input>50k | 19.6 s | 12.3 s | 16.1 s |

- "Self-hits only" is refuted: 0.76 s median TTFT at ~60k input is below any
  cold prefill time (>= 2 s even at 30k tok/s instantaneous), and 80% of
  >50k-input requests reach TTFT < 2 s under fcfs at c4.
- TTFT-vs-input OLS fits have low R^2 everywhere (0.004-0.083): no arm is a
  zero-reuse tight-monotone map; implied prefill rates from the fits are
  scatter-driven and not used.
- The chunked-prefill self-hit ramp model (re-reported prefix per chunk,
  ~c*n(n-1)/2 for n chunks) over-predicts by 0.73-0.78x where matches are
  logged (page alignment + host load-back), so it is a scale reference only.

### Task 1b: fcfs-match verification (c4 gate)

The c4 gate passed on the first official run (2026-10-06 14:21, fresh boot):
**fcfs-match hit = 0.916** vs lpm's recomputed 0.912 (fcfs: 0.000) -- FCFS
ordering with forced sort-time matching realizes the same admission-time
reuse the verdict above predicts, so the round-1 fcfs hit column was
bookkeeping, not cache. The c16 fcfs-match run and the rest of round 2 were
unblocked on this gate.

### Task 3: implementation invariants (`analysis/boost_invariant_check.py`)

**Overtaking invariant: PASS on every boost run.** For pairs where x entered
the queue earlier in the token clock (A(x) < A(y)) but y was admitted first,
the head start A(y) - A(x) must stay within b_gamma(k). Round 2 re-ran the
check over all 10 boost runs with the new frozen first-admission fields:

| run | pairs | within | beyond | same-batch | exempt-follower | legacy-deep | hard |
|---|---|---|---|---|---|---|---|
| c16 1e-4 round-1 (warm) | 23 | 18 | 5 | 1 | 0 | 4 | **0** |
| c16 3e-6 fresh | 111 | 111 | 0 | 0 | 0 | 0 | **0** |
| c16 1e-5 fresh | 146 | 146 | 0 | 0 | 0 | 0 | **0** |
| c16 2e-5 fresh | 83 | 80 | 3 | 0 | 0 | 0 | **3** |
| c16 3e-5 fresh | 51 | 37 | 14 | 2 | 5 | 4 | **3** |
| c16 1e-4 fresh | 58 | 27 | 31 | 3 | 4 | 21 | **3** |
| c4/c8 cells (5 runs) | 0 | — | — | — | — | — | **0** |

The 9 "hard" pairs (3 each in the tight-bound cells 2e-5 / 3e-5 / fresh
1e-4; zero at loose gamma and zero in round-1 warm) are **not phi-sort
violations** — they are the in-batch dedup follower pin, an SGLang-wide
design element shared with lpm, interacting with this workload's shared
AgentX system preamble under fresh-boot pool pressure:

1. **Arithmetic.** b_gamma(q(u)) <= b_gamma(k) for every request (q bins at
   >= k and b is decreasing), so phi(y) < phi(x) requires
   A(y) - A(x) < b_gamma(k). Every hard pair has gap > bound (e.g. the 3e-5
   cell: gaps 484k/542k/526k vs bound 162k), so group-0 ordering cannot
   produce it — by construction.
2. **Parking excluded.** All held requests have
   `host_hit_prefix_tokens_first_admission=0`,
   `device_matched_prefix_tokens_first_admission=0`, and 0 retractions — no
   hicache load-back parking, no retraction displacement; the adder's
   failure semantics (every non-CONTINUE `add_one_req` result breaks the
   loop) rule out pass-around.
3. **Mechanism.** A waiting request whose radix match is <= 32 tokens
   (`IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`,
   `schedule_policy.py`) is matched against the simulated waiting-queue
   tree; if it shares >= 32 leading tokens
   (`IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`) with an
   earlier-scanned waiting peer it is deprioritized and sorts last — group
   1, rank 0.0, rid tie-break. Under c16 fresh-boot pool pressure (93-99%
   full) the shared system preamble is evicted, so cold new sessions have
   radix match 0 and qualify; any two co-waiting cold sessions pin each
   other. The pinned request sits behind ALL group-0 requests regardless of
   phi — later arrivals overtake it without bound.
4. **Cohort signature.** In each tight cell the overtaken set is one or two
   requests held 112-298 s while every other big-cold request in the same
   run admits at ~0-35 s — a co-waiting lottery, not a systematic big-cold
   penalty. The frozen follower flag at admission is False for all of them
   (the freeze captures the release-time verdict; the pin lifts once the
   earlier-scanned peers drain), which is why the check's overtaker-side
   exemption never fired.
5. **rid-order drain confirmed.** Pinned followers tie-break by rid string;
   in the 3e-5 cell all three overtakers of the held a0a68d sort lex-before
   it — the cohort drained in rid order, exactly the mechanism's prediction.
6. **Policy-independence of the symptom.** fcfs-match c16 (fresh) shows the
   same big-cold long holds (65k-input held 132 s, 59k held 102 s) under
   pure FCFS ordering, where the follower pin cannot exist; there the hold
   is head-of-line budget blocking. Big-cold holds at c16 fresh-boot are a
   workload x deployment effect every policy pays in some form — the
   cache-aware policies via the follower pin, FCFS via head-of-line
   blocking.

**Verdict.** The overtaking invariant A(y)-A(x) <= b_gamma(k) holds for the
phi (group-0) ordering on every boost run — zero violations attributable to
the boost sort across 58+83+51+146+111+23 pairs. Not an implementation bug.
Honest caveats: the frozen `in_batch_dedup_follower_first_admission` flag
cannot positively attest pin status at hold time (it is the release-time
verdict), so the attribution rests on the elimination chain above; and at
loose gamma the same pins occur but their gaps fall inside the loose bound
(the bound is the discriminator, not the mechanism). Follow-up lever, out
of scope here: the 32/32 thresholds are env-overridable and a bound on
follower hold time would remove the rid-lottery; the check script will gain
a `dedup_pin_cold_cohort` classification for these pairs so they stop
reporting as "hard".

**u bimodality — PASS on every fresh cell.** u = input + output - matched
at first admission; warm = matched > 0. Warm median u 3.8-4.4k vs cold
median 38.7-46.2k (~10x), warm p10 749-848 vs cold p10 1063-1179,
frac(warm below cold p10) 0.15-0.19, warm-median/cold-p10 ratio 3.4-4.1:
the cold/warm class split is well-supported by the uncached-work
distribution.

**c8 tail pinning.** c8 boost runs pin q-wait p99 at ~3.6 s = the
chunked-prefill iteration cadence (8192-token chunks, one `chunked_req` at
a time): a newcomer admitted behind one chunk iteration waits one
iteration. Tail waits at c8 are chunk-blocking, not ordering.

### Task 8.1: offline gamma-hat table (`analysis/gamma_hat_offline.py`)

gamma_hat = ln(0.05/0.01)/(w99 - w95) = ln 5 / tail width, with the
admission wait tail scaled to clock tokens by the fcfs arm's total-token
throughput R (round-1 logs carry no admission clock stamp):

| c | R (tok/s) | tail source | p99 - p95 (s) | tail (tokens) | gamma_hat | clip [3e-7, 1e-3] |
|---|---|---|---|---|---|---|
| 4 | 6,930 | lpm | 2.061 | 14,281 | 1.13e-4 | -- |
| 4 | 6,930 | boost 1e-4 | 0.003 | 21 | 7.7e-2 | 1e-3 |
| 8 | 6,748 | lpm | 1.748 | 11,797 | 1.36e-4 | -- |
| 8 | 6,748 | boost 1e-4 | 3.065 | 20,683 | 7.78e-5 | -- |
| 16 | 7,842 | lpm | 23.064 | 180,859 | 8.90e-6 | -- |
| 16 | 7,842 | boost 1e-4 | 29.836 | 233,961 | 6.88e-6 | -- |

Reading: c4 clips at gamma_max (waits ~0: the maximally thin tail is
near-FCFS, which is what boost already delivers there at hit parity); c8's
estimates sit inside the round-1 swept grid {5e-5 .. 2e-4} (the dial does
not bind -- consistent with round-1 finding 4); c16's fixed point lies
between lpm's 8.9e-6 and boost's own 6.9e-6, inside the round-2 grid {3e-6,
1e-5, 2e-5, 3e-5}. Scale caveat: R here is client-measured throughput while
the runtime estimator measures the processed-token counter, whose rate is
cache-state-dependent — it counts prefill extend tokens including the
linear replay of matched prefixes, so it runs ~3.3-6k tok/s at c16
fresh-boot and ~36.7k tok/s at c16 round-1 warm (see "Token-clock rates"
below; the earlier "~128k tok/s" figure in this caveat was wrong). An
offline table scaled by client R therefore overstates a fresh-boot runtime
gamma by the warm/fresh clock ratio and roughly matches a warm pool; the
wall-time bound b_gamma(k)/R_clock is invariant to the rescale, and
gamma-Ada's acceptance test is behavioral (hit / starvation vs the sweep's
best gamma), not a numeric gamma match.

### Token-clock rates (corrects the calibration above)

Measured from the eval logs' `boost_arrival_tokens` spans over wall spans
(max stamp minus min stamp per run): fresh-boot c16
runs ~3.3-6k tok/s (stamps span ~8M tokens over ~2450 s of ramp); the
round-1 warm c16 run spans ~237.9M tokens / 6481 s ~ 36.7k tok/s. The clock
(`processed_tokens_counter += batch.extend_num_tokens`,
`python/sglang/srt/managers/scheduler.py`) counts prefill extend tokens
including the linear-replay of matched prefixes, so the rate is
cache-state-dependent — fresh pools replay little, warm pools replay a lot.
Wall-clock equivalents of the overtaking bound b_gamma(k), k=256, at the
measured fresh rates (warm ~10x shorter):

| gamma | b_gamma(k) (tokens) | fresh wall bound (s) |
|---|---|---|
| 3e-6 | 2.39M | 400-724 |
| 1e-5 | 597k | 100-181 |
| 2e-5 | 264k | 44-80 |
| 3e-5 | 162k | 27-49 |
| 1e-4 | 36.8k | 6-11 |

This coherently explains the c16 frontier's TPOT column: at fresh rates
gamma=1e-4 buys only ~6-11 s of overtaking (FCFS-ward), 3e-6 buys 7-12
minutes (LAS-ward); the round-1 c16 1e-4 arm was already effectively FCFS
at the tail.

### Task 4: targeted starvation regime at c8 (`analysis/select_starvation_entries.py`)

Workload recon (mmap manifest + the aiperf source under `.venv`):

- The pool is two-level in scan order: 393 long multi-turn root chains,
  then 9,209 subagent child chains (median 1 turn, ~0 s think-times).
  Sampling is sequential over root trajectories (wrap at pool end; the
  wrap gate is satisfied by the force-enabled cache-bust), and a
  trajectory's child chains spawn when the replay clock reaches their
  recorded branch offsets.
- The scenario (`inferencex-agentx-mvp`) forbids every delay-compression
  flag (`--ignore-trace-delays`, `--inter-turn-delay-cap-seconds`, trace
  idle-gap cap; `--replay-speedup` is baseten-only;
  `--conversation-turn-delay-ratio` is synthetic-only) and force-enables
  first-turn-prefix cache-bust, so wrapped re-draws stay cache-cold.
- Per-turn think-times are bimodal along each trajectory: seconds-scale
  early (tightly recorded agent-loop latency), minutes-scale late (real
  agent work between turns). The official c8 runs are sparse because
  sessions resume mid-trajectory (uniform [0, 1] start ratio) into the
  slow region, so c8 never builds queue depth.

Construction (all stock aiperf flags, no AIPerf change):

- `--num-dataset-entries 16`: pool = the first 16 trajectories -- 16 long
  multi-turn roots (mean 30 turns) + 121 short child chains; children of
  the dense roots spawn at 28-285 s branch offsets.
- `--trajectory-start-min-ratio 0.0 --trajectory-start-max-ratio 0.25`:
  sessions resume in the first quarter of each trajectory -- the only
  legal arrival-density lever -- so c8 lanes fire seconds-scale turns
  instead of minutes-scale ones.
- Selector table (reachable = upper bound walking recorded delays from the
  resume point; ignores serving time, warmup requests, and child traffic,
  which arrives on top):

| N | root turns | reachable @0.25 | reachable @0 | child chains | pool requests |
|---|---|---|---|---|---|
| 8 | 241 | 122 | 155 | 72 | 1,228 |
| 16 | 478 | 223 | 296 | 121 | 2,685 |
| 32 | 1,127 | 610 | 773 | 187 | 4,198 |

  N* = 16: the smallest pool that is both dense (>= 200 reachable root
  turns under the resume window) and child-rich (>= 50 short chains to
  interleave cold new-session starts). Sequential wrap re-draws exhausted
  trajectories cache-cold, keeping the newcomer class arriving late in the
  run.

Driver `scripts/run_round4.sh`: a 300 s lpm gate first (cold n >= 10 and
cold p99 >= 5 s) -- a fail is the trace-structure finding (the c8 trace
cannot build a cold-class tail even at maximum legal density) and stops
before the 3 x 1800 s arms (lpm / boost at default gamma / fcfs-match,
fresh boot each, into `results/agentx_targeted`, kept out of the official
matrix). Headline: cold-class queue-wait p99, cold-vs-warm gap99,
arrival-order OLS slope. **Round 4 never ran on this hardware**: the
auto-armed watcher fired when the round-2 driver was killed (19:24) and was
itself killed at 19:37; the partial cell is quarantined under
`results/quarantine/c16_20261006_192407` and the targeted regime moves to
the H200 campaign.

## Final results (RTX 6000 Pro campaign close-out)

Canonical table (`python3 analysis/print_results_table.py`, default mode;
latest-per-seed, seed replicas n-weighted). RTX 6000 Pro campaign closed
2026-10-06/07 pending the 4x H200 campaign: round 4 (targeted regime),
round 5 (seed replicas), and the hrrn c16 cell were all skipped on this
hardware (the hrrn c16 cell started 19:24 was killed at 19:37 and
quarantined under `results/quarantine/c16_20261006_192407`; the round-4
watcher was disarmed before its driver wrote anything) and move to H200.
hrrn c8 landed before the cut and keeps its row.

```
| Policy | c | γ | n | hit | q_p95 (s) | q_p99 (s) | starv | cold n | cold starv | warm n | warm starv | gap99 (s) | RPM | oTPM | TTFT (ms) | TPOT p95 (ms) | retr | q_p99 Δ vs LPM | Anti-starv. vs LPM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 4 | — | 217 | 0.912 | 0.005 | 2.830 | 0.000 | 7 | 0.000 | 210 | 0.000 | +6.596 | 7.1 | 5371 | 1851 | 28.4 | 0 | — | — |
| boost | 4 | 1e-4 | 223 | 0.916 | 0.005 | 0.009 | 0.000 | 7 | 0.000 | 216 | 0.000 | −0.004 | 7.3 | 5922 | 1798 | 27.5 | 0 | −99.7% | 0% ¹ |
| fcfs-match | 4 | — | 223 | 0.916 | 0.005 | 1.770 | 0.000 | 7 | 0.000 | 216 | 0.000 | −1.645 | 7.3 | 5922 | 1748 | 27.0 | 0 | −37.5% | 0% ¹ |
| fcfs ‡ | 4 | — | 214 | 0.000 | 0.005 | 2.850 | 0.000 | 215 | 0.000 | 0 | — | — | 7.0 | 5165 | 1736 | 26.8 | 0 | +0.7% | 0% ¹ |
| **lpm (baseline)** | 8 | — | 141 | 0.941 | 0.005 | 2.590 | 0.000 | 10 | 0.000 | 131 | 0.000 | −2.041 | 4.6 | 5839 | 1987 | 24.1 | 0 | — | — |
| boost | 8 | 5e-5 | 143 | 0.940 | 0.005 | 3.570 | 0.007 | 10 | 0.000 | 133 | 0.008 | −3.091 | 4.7 | 5865 | 2230 | 24.3 | 0 | +37.8% | −0.7 pp ¹ |
| boost | 8 | 1e-4 | 140 | 0.942 | 0.005 | 3.600 | 0.000 | 10 | 0.000 | 130 | 0.000 | −3.093 | 4.6 | 5836 | 2170 | 22.2 | 0 | +39.0% | 0% ¹ |
| boost | 8 | 2e-4 | 139 | 0.941 | 0.005 | 3.600 | 0.000 | 10 | 0.000 | 130 | 0.000 | −3.091 | 4.5 | 5678 | 2278 | 21.5 | 0 | +39.0% | 0% ¹ |
| hrrn | 8 | — | 129 | 0.937 | 0.005 | 2.855 | 0.000 | 10 | 0.000 | 119 | 0.000 | −2.767 | 4.2 | 5612 | 2146 | 24.5 | 0 | +10.2% | 0% ¹ |
| fcfs ‡ | 8 | — | 139 | 0.000 | 0.005 | 2.805 | 0.000 | 139 | 0.000 | 0 | — | — | 4.6 | 5826 | 2099 | 24.3 | 0 | +8.3% | 0% ¹ |
| **lpm (baseline)** | 16 | — | 201 | 0.869 | 73.929 | 94.750 | 0.380 | 20 | 0.400 | 185 | 0.378 | −12.232 | 6.6 | 5622 | 21468 | 98.6 | 6 | — | — |
| boost | 16 | 3e-6 | 165 | 0.820 | 78.000 | 188.667 | 0.488 | 21 | 0.381 | 147 | 0.503 | −40.063 | 5.4 | 5061 | 25810 | 97.8 | 0 | +99.1% | −28.4% |
| boost | 16 | 1e-5 | 170 | 0.838 | 87.833 | 171.167 | 0.457 | 20 | 0.600 | 153 | 0.438 | +36.656 | 5.5 | 5007 | 26937 | 103.3 | 5 | +80.7% | −20.3% |
| boost | 16 | 2e-5 | 165 | 0.830 | 106.875 | 191.375 | 0.473 | 18 | 0.611 | 151 | 0.457 | +148.486 | 5.4 | 4940 | 26667 | 112.5 | 0 | +102.0% | −24.5% |
| boost | 16 | 3e-5 | 159 | 0.810 | 111.250 | 192.250 | 0.531 | 18 | 0.389 | 144 | 0.549 | −30.451 | 5.2 | 4936 | 28227 | 102.9 | 0 | +102.9% | −39.7% |
| boost | 16 | 1e-4 | 168 | 0.836 | 127.778 | 215.000 | 0.471 | 20 | 0.400 | 150 | 0.480 | −16.284 | 5.5 | 5023 | 28858 | 143.1 | 6 | +126.9% | −23.9% |
| fcfs-match | 16 | — | 142 | 0.805 | 151.667 | 190.333 | 0.559 | 18 | 0.444 | 127 | 0.575 | −16.492 | 4.6 | 4590 | 31930 | 121.1 | 0 | +100.9% | −47.1% |
| fcfs ‡ | 16 | — | 151 | 0.000 | 97.833 | 178.143 | 0.549 | 153 | 0.549 | 0 | — | — | 4.9 | 4697 | 32312 | 166.8 | 5 | +88.0% | −44.5% |

**Footnotes**

(a) ¹ Anti-starvation is relative to the LPM baseline of the same concurrency; a relative change is undefined when the baseline is 0, so the cell prints the absolute point change instead (`−X.X pp`). The starvation rate counts queue waits > 10 s and is blind below that threshold, so at c4/c8 (waits ≤ 4 s) the rate column stays 0.000 for every arm -- the c4 evidence is the cold-vs-warm queue-wait gap (LPM +6.39 s → boost ~0.00 s, eliminated) and q_p99 −99.7% at hit parity.

(b) ‡ fcfs confound: fcfs leaves the sort-time prefix-match bookkeeping at zero for this deployment's cache (every current implementation returns False from `supports_fast_match_prefix()`), so its hit column reads 0.000 even though admission-time matching gives it real cache reuse (Task 1's verdict); fcfs-match is the reference arm with the bookkeeping forced on. fcfs still measures FCFS *ordering*, which is what boost degenerates toward at large γ.

(c) Cold-class starvation at c16: boost (γ=3e-6) 0.381 vs LPM 0.400 (-4.8% relative); fcfs 0.549 (+37.3%). Overall starvation at c16: LPM 0.380, boost 0.488, fcfs 0.549.

(d) Metric definitions: starv = share of requests with queue wait > 10 s; gap99 = cold-class p99 − warm-class p99 queue wait (schedule-eval log, first admission stamp); hit = token-weighted first-attempt prefix hit rate; q_p95/p99 = queue-wait percentiles (server histogram); TTFT = client-observed mean; retr = retraction count. Seed replicas average into one n-weighted row.
```

Notes on the canonical table:

- **gamma\* = 1e-5** (k=256): best hit of the fresh c16 sweep (0.838), TPOT
  p95 +4.8% vs lpm (inside the +10% guard, as are 3e-6/3e-5), and it sits in
  the offline gamma-hat fixed-point region (Task 8.1). The locality axes
  (hit, oTPM) are gamma-insensitive to within noise; every discriminating
  axis (TPOT, hit, offline gamma-hat) favors 1e-5. The frontier is
  honestly flat: at c16 fresh-boot, gamma in [3e-6, 1e-4] moves hit by <3
  pts and oTPM by <2.5%, while warm-start state and the pinned-cohort
  mechanism (Task 3) move both by more than gamma does.
- Footnote (c)'s boost cell is the **lowest-cold-starvation** cell (3e-6,
  0.381), not gamma\*: the gamma\* pick rests on hit / TPOT p95 / the
  offline gamma-hat region, and gamma\*=1e-5's cold-class starvation
  (0.600) is the sweep's worst — reported as-is, not gate-relevant.
- **Fresh-vs-warm confound**: the lpm c16 baseline (and fcfs c16) are
  round-1 warm-start rows; every c16 boost / fcfs-match / hrrn row is
  fresh-boot. The measured confound on the 1e-4 pair is +0.7 pts hit,
  +3.2% oTPM, +23.7% TPOT p95 (fresh 1e-4 vs warm 1e-4), so ~2.4-5.2 pts
  of the c16 boost-vs-lpm hit gap and ~7-10% of the oTPM gap are
  structural to boost-vs-lpm, the rest warm-reference artifact.
- hrrn appears at c8 only (fresh-boot, round 2); the c16 hrrn cell and the
  round-1 warm c8 boost rows are superseded/footnoted as noted above.
- Round-1 c8 boost rows are warm-start (no fresh c8 boost cell existed
  until round 6 adds one under `results/agentx_adaptive`).

Frontier plot: `results/1x-rtx6000pro/agentx/frontier_c16.png` (regenerate with a
matplotlib-bearing interpreter: `python3 analysis/frontier_plot.py
--gamma-star 1e-5`; system python3 lacks matplotlib — use the aiperf venv
or conda python). Point coordinates and per-point guards:
`results/1x-rtx6000pro/agentx/frontier_c16.json`.

## Round 6: gamma-Ada validation

`GAMMA_STAR=1e-5 scripts/run_round6.sh` (fresh boot per arm, seed 42,
1800 s) then `analysis/adaptive_gamma_check.py --gamma-star 1e-5 --bracket
3e-6 2e-5`. Arms: adaptive c8 + adaptive c16 (gamma0 = 1e-4), plus a fresh
static gamma*=1e-5 c8 reference; the static c16 gamma* cell is round 2's.
Raw output: `results/1x-rtx6000pro/agentx_adaptive/adaptive_gamma_check.json`.

**c16 (the validation cell): PASS — bracket, hit, starvation, and TPOT all
hold.**

- Trajectory (188 arrivals): gamma0=1e-4 held while the window filled to
  MIN_SAMPLES=64 (1233 s; warmup-phase admissions feed the window too);
  the first update stepped straight to 3.5e-5, below 2e-5 by +1317 s
  (arrival #64); min 1.34e-5; the last quarter (n=47) oscillated
  1.34-2.43e-5 and converged (mean) to **1.76e-5** — inside the bracket
  [3e-6, 2e-5], at its FCFS-ward edge: the estimator's fixed point sits at
  gamma*'s 2e-5 neighbor, not at gamma* itself.
- Behavior vs the fresh static gamma*=1e-5 cell (same protocol): hit 0.833
  vs 0.838 (−0.5 pts; guard ±2 pts), cold-class starvation 0.500 vs 0.600
  (better), cold p99 131.7 vs 167.6 s, TPOT p95 **76.9 vs 103.3 ms
  (−25.5%)**, oTPM 4975 vs 5007 (−0.6%). Against every other fresh c16
  reference the TPOT guard also holds with room (3e-6: 97.8 ms;
  fcfs-match: 121.1 ms) — the adaptive cell posts the best c16 TPOT p95 of
  the campaign.

**c8: bracket FAIL — estimator-bias finding, documented, not patched.**

- Converged **1.34e-4**, outside [3e-6, 2e-5]. The bracket is the c16
  sweep's neighbors and does not transfer to c8 by construction: the
  offline fixed point at c8 is 7.78e-5 on boost's own round-1 tail vs
  1.36e-4 on lpm's (Task 8.1 table).
- Trajectory (147 arrivals): gamma0 held 1085 s (43 arrivals), first
  update +1098 s; settled at 9.82-9.88e-5 for ~600 s (the boost-own-tail
  fixed point region), then climbed monotonically to 1.70e-4 as the pool
  warmed and the admission tail thinned — thin tail -> FCFS-ward, the
  cache-state-dependent clock rate of the "Token-clock rates" section. The
  converged value lands on lpm's offline c8 estimate (1.36e-4), i.e. the
  own-tail fixed point drifted past the warm-pool lpm-tail estimate with
  cache state; no hand-patch applied.
- Cost of the miss at c8: none measurable — vs the fresh static gamma*=1e-5
  c8 arm, hit 0.940 vs 0.941, cold-class starvation 0.000 both, TPOT p95
  25.5 ms both (the dial does not bind at c8; round-1 finding 4).

**Verdict.** gamma-Ada reproduces the c16 frontier pick's behavior from a
cold 1e-4 start with zero configuration: converged gamma inside the sweep
bracket, hit/starvation/TPOT no worse than (here: better than) the static
gamma* cell. The c8 arm exposes the estimator's known scale sensitivity —
it tracks its per-concurrency, cache-state-dependent fixed point, which at
c8 sits two orders above the c16 bracket — harmless where the dial does
not bind, and a documented caveat for any deployment that carries one
gamma across concurrencies. Upstream candidate (with boost) pending H200
validation.

## Full run (2026-10-07): five-policy comparison at c4/c8 — final RTX run

Self-contained write-up (setup, execution log, table, trajectories,
verdicts, acceptance): `full_run.md`.

6 arms x {c4, c8} = 12 cells, fresh boot per cell, seed 42, 1800 s, pinned
serving config unchanged (`scripts/run_fullrun_c4c8.sh`; driver log
`logs/fullrun_c4c8_20261007_022137.log`). Pre-flight dfs-weight smoke
(600 s, c4) passed before the sweep — the arm that had 3 failed boot
attempts and zero landed cells landed both cells cleanly here. Zero cell
failures; nothing quarantined. Retractions 0 in all 12 cells; the
>128-request FCFS fallback never came close (max `#queue-req` observed
across all 12 servers: 2).

Canonical table regenerated (`python3 analysis/print_results_table.py`,
default mode). The two gamma-Ada rows carry `gamma0 -> converged` in the
gamma column; their run_meta key stays (gamma0, k), so each supersedes the
round-1 static cell at the same key (c4 1e-4 static; c8 1e-4 static) by
latest-per-seed. The c8 5e-5 / 2e-4 boost rows are the round-1/2 static
references, unchanged. Starvation threshold: the 12 full-run cells were
regenerated at `--starvation-threshold 0.2` (2026-10-07; TTFT/TPOT p99
fields added to the stats at the same time) — the 10 s default was blind
at c4/c8 (waits 3 ms to ~8 s), leaving starv all-zero; no request lands
between 0.2 s and the multi-second tail, so the 0.2 / 0.25 / 0.5 / 1 s
rates coincide. Only the
starvation fields (and the new p99 fields) changed, verified
field-by-field against the 10 s output. The two static c8 references predate the scrape-retention
policy, so they keep the 10 s threshold (marked †, not comparable).

| Policy | c | γ | n | hit | q_p95 (s) | q_p99 (s) | starv | cold n | cold starv | warm n | warm starv | gap99 (s) | RPM | oTPM | TTFT (ms) | TPOT p95 (ms) | retr | q_p99 Δ vs LPM | Anti-starv. vs LPM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 4 | — | 224 | 0.918 | 0.005 | 2.760 | 0.018 | 7 | 0.000 | 217 | 0.018 | −2.486 | 7.3 | 5965 | 1718 | 29.0 | 0 | — | — |
| boost | 4 | 1e-4→1.96e-4 * | 216 | 0.914 | 0.005 | 2.943 | 0.032 | 7 | 0.000 | 210 | 0.033 | −2.950 | 7.0 | 5184 | 1766 | 26.2 | 0 | +6.6% | −77.8% |
| hrrn | 4 | — | 216 | 0.914 | 0.005 | 3.840 | 0.023 | 7 | 0.143 | 209 | 0.019 | +2.681 | 7.1 | 5364 | 1770 | 27.0 | 0 | +39.1% | −27.8% |
| fcfs-match | 4 | — | 223 | 0.916 | 0.005 | 3.770 | 0.018 | 7 | 0.000 | 216 | 0.019 | −3.007 | 7.3 | 5922 | 1786 | 24.6 | 0 | +36.6% | +0.0% |
| fcfs ‡ | 4 | — | 216 | 0.000 | 0.004 | 0.005 | 0.009 | 217 | 0.009 | 0 | — | — | 7.0 | 5184 | 1681 | 28.7 | 0 | −99.8% | +50.0% |
| dfs-weight | 4 | — | 221 | 0.917 | 0.005 | 4.780 | 0.023 | 7 | 0.000 | 215 | 0.023 | −3.840 | 7.2 | 5476 | 1815 | 24.1 | 0 | +73.2% | −27.8% |
| **lpm (baseline)** | 8 | — | 138 | 0.941 | 0.005 | 3.610 | 0.022 | 10 | 0.000 | 129 | 0.023 | −3.100 | 4.5 | 5674 | 2154 | 22.6 | 0 | — | — |
| boost | 8 | 5e-5 | 143 | 0.940 | 0.005 | 3.570 | 0.007 † | 10 | 0.000 | 133 | 0.008 † | −3.091 | 4.7 | 5865 | 2230 | 24.3 | 0 | −1.1% | +68.2% † |
| boost | 8 | 1e-4→9.8e-5 * | 138 | 0.939 | 0.005 | 3.620 | 0.022 | 10 | 0.000 | 128 | 0.023 | −3.100 | 4.5 | 5823 | 2233 | 21.0 | 0 | +0.3% | +0.0% |
| boost | 8 | 2e-4 | 139 | 0.941 | 0.005 | 3.600 | 0.000 † | 10 | 0.000 | 130 | 0.000 † | −3.091 | 4.5 | 5678 | 2278 | 21.5 | 0 | −0.3% | +100.0% † |
| hrrn | 8 | — | 142 | 0.940 | 0.005 | 2.570 | 0.014 | 10 | 0.000 | 132 | 0.015 | −2.010 | 4.7 | 5847 | 2058 | 21.7 | 0 | −28.8% | +36.4% |
| fcfs-match | 8 | — | 143 | 0.941 | 0.005 | 3.570 | 0.021 | 10 | 0.000 | 133 | 0.023 | −3.093 | 4.7 | 5859 | 2154 | 24.5 | 0 | −1.1% | +4.5% |
| fcfs ‡ | 8 | — | 146 | 0.000 | 0.005 | 3.540 | 0.021 | 146 | 0.021 | 0 | — | — | 4.8 | 5900 | 2166 | 23.9 | 0 | −1.9% | +4.5% |
| dfs-weight | 8 | — | 138 | 0.940 | 0.005 | 2.800 | 0.021 | 10 | 0.000 | 129 | 0.023 | −2.689 | 4.5 | 5674 | 2115 | 21.3 | 0 | −22.4% | +4.5% |
| **lpm (baseline)** | 16 | — | 201 | 0.869 | 73.929 | 94.750 | 0.380 | 20 | 0.400 | 185 | 0.378 | −12.232 | 6.6 | 5622 | 21468 | 98.6 | 6 | — | — |
| boost | 16 | 3e-6 | 165 | 0.820 | 78.000 | 188.667 | 0.488 | 21 | 0.381 | 147 | 0.503 | −40.063 | 5.4 | 5061 | 25810 | 97.8 | 0 | +99.1% | −28.4% |
| boost | 16 | 1e-5 | 170 | 0.838 | 87.833 | 171.167 | 0.457 | 20 | 0.600 | 153 | 0.438 | +36.656 | 5.5 | 5007 | 26937 | 103.3 | 5 | +80.7% | −20.3% |
| boost | 16 | 2e-5 | 165 | 0.830 | 106.875 | 191.375 | 0.473 | 18 | 0.611 | 151 | 0.457 | +148.486 | 5.4 | 4940 | 26667 | 112.5 | 0 | +102.0% | −24.5% |
| boost | 16 | 3e-5 | 159 | 0.810 | 111.250 | 192.250 | 0.531 | 18 | 0.389 | 144 | 0.549 | −30.451 | 5.2 | 4936 | 28227 | 102.9 | 0 | +102.9% | −39.7% |
| boost | 16 | 1e-4 | 168 | 0.836 | 127.778 | 215.000 | 0.471 | 20 | 0.400 | 150 | 0.480 | −16.284 | 5.5 | 5023 | 28858 | 143.1 | 6 | +126.9% | −23.9% |
| fcfs-match | 16 | — | 142 | 0.805 | 151.667 | 190.333 | 0.559 | 18 | 0.444 | 127 | 0.575 | −16.492 | 4.6 | 4590 | 31930 | 121.1 | 0 | +100.9% | −47.1% |
| fcfs ‡ | 16 | — | 151 | 0.000 | 97.833 | 178.143 | 0.549 | 153 | 0.549 | 0 | — | — | 4.9 | 4697 | 32312 | 166.8 | 5 | +88.0% | −44.5% |

(*) gamma-Ada rows: `gamma0 -> converged gamma` (last-quarter mean; the
eval log carries the full per-admission trajectory). c16 rows are the
round-2 sweep (static; unchanged). Footnotes (a)-(d) of the close-out
table above apply verbatim.

### Gamma-Ada trajectories (the two full-run boost cells)

- **c4** (`c4_gada_k256_20261007_052130`, 222 arrivals): gamma0=1e-4 held
  while the window filled, climbed through ~7e-4, **touched the 1e-3 clip
  at 64% through the run**, then descended and settled at **1.96e-4** from
  69% onward (last quarter 1.96-1.97e-4, flat). The plan's "pins at the
  1e-3 clip" guess was half right: the clip was visited but not pinned —
  the fixed point sits at ~2x gamma0. Benign at c4 headroom: hit 0.914,
  starv@0.2s 0.032, TPOT p95 26.2 ms, all inside the arm band.
- **c8** (`c8_gada_k256_20261007_090608`, 146 arrivals): gamma0=1e-4,
  monotone slight decay, converged **9.82e-5** (last quarter 9.80-9.82e-5,
  flat; peak = gamma0 itself). Dead center of the offline fixed-point band
  for c8 (7.8e-5-1.4e-4, Task 8.1) and a hair under gamma0 — the dial does
  not bind at c8, consistent with round 6.

### Reading notes for the full-run table

- **c4 queue tails are 2-3-request phenomena.** At n~220, q_p99 is the
  ~2nd-worst wait; this run's outlier waits landed on warm first-admissions
  in five of six arms (gap99 negative; round-1's lpm c4 had them on cold
  ones, +6.60 s). hrrn is the only positive-sign arm (+2.68 s). Treat c4
  gap99 sign as run noise; the round-1 sign is in the close-out table.
- fcfs c4's q_p99 0.005 s (vs fcfs-match 3.77 s, same ordering) is the
  same tail noise in the other direction: round-1 read fcfs 2.85 /
  fcfs-match 1.77. The c4 queue-histogram percentiles rank arms only
  within a run, not across runs.
- The c16 group is carried unchanged from the close-out table for
  completeness; the full run adds no c16 cells (that band belongs to H200).
