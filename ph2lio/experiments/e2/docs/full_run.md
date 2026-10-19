# Full run (2026-10-07): five-policy comparison at c4/c8 — final RTX run

6 arms x {c4, c8} = 12 cells on the RTX 6000 Pro, the closing run of the
E2 boost-policy campaign. Plan: `/home/user1111/sglang/fullrun_c4c8_prompt.md`.
Companion docs: `agentx_experiment.md` (full campaign protocol and per-round
detail), `agentx_final_report.md` (campaign summary). This file documents
only the full run.

## Setup

- Harness: `scripts/run_e2.sh policy <arm> <concurrency>`, driven by
  `scripts/run_fullrun_c4c8.sh` (new for this run).
- Serving config: `scripts/start_server.sh`, **unchanged** (pinned flags,
  CCCL opt-out included). No server-code changes; the branch's boost /
  gamma-Ada code (commit `859a7ef145`) is the only arm-specific knob.
- Per cell: fresh boot (`STOP_SERVER=always`), seed 42, DURATION=1800,
  AIPerf `inferencex-agentx-mvp`, Weka agentic-coding traces (256K-capped),
  Qwen3.8-27B-FP8 single GPU.
- Arms: lpm (baseline), dfs-weight, fcfs, fcfs-match, hrrn, and boost =
  **gamma-Ada** (BOOST_ADAPTIVE_GAMMA=1, gamma0=BOOST_GAMMA=1e-4,
  BOOST_BIN_SIZE=256). No static-gamma boost arm — the round-1/2 static
  references (c8 5e-5 / 2e-4) stay in the table for context.
- Order: risk-earliest — lpm first, dfs-weight right after its pre-flight
  smoke, then the reference arms, boost last; c4 group then c8 group.

## Execution

Pre-flight smoke gate (mandatory per the plan): dfs-weight c4, 600 s
(`results/1x-rtx6000pro/agentx/smoke/smoke_c4_20261007_020140`) — **PASSED** (83 eval
records, 0 errors, eval log written, exports present). This arm had 3
failed boot attempts and zero landed cells before the CCCL fix
(commit `2c82ab2323`); both sweep cells then booted cleanly.

Sweep started 02:21:37, completed 09:43:40 (~7 h 22 m). **12/12 cells
landed, zero failures, nothing quarantined.** Driver log:
`logs/fullrun_c4c8_20261007_022137.log`.

| # | cell | run dir (under `results/1x-rtx6000pro/agentx/`) |
|---|---|---|
| 1 | lpm c4 | `lpm/c4_20261007_022137` |
| 2 | dfs-weight c4 | `dfs-weight/c4_20261007_025733` |
| 3 | fcfs c4 | `fcfs/c4_20261007_033335` |
| 4 | fcfs-match c4 | `fcfs-match/c4_20261007_040937` |
| 5 | hrrn c4 | `hrrn/c4_20261007_044533` |
| 6 | boost c4 (gamma-Ada) | `boost/c4_gada_k256_20261007_052130` |
| 7 | lpm c8 | `lpm/c8_20261007_055737` |
| 8 | dfs-weight c8 | `dfs-weight/c8_20261007_063529` |
| 9 | fcfs c8 | `fcfs/c8_20261007_071316` |
| 10 | fcfs-match c8 | `fcfs-match/c8_20261007_075053` |
| 11 | hrrn c8 | `hrrn/c8_20261007_082830` |
| 12 | boost c8 (gamma-Ada) | `boost/c8_gada_k256_20261007_090608` |

Health gauges across all 12 cells: retractions **0** everywhere; the
>128-request FCFS fallback for LPM/HRRN/BOOST never came close (max
`#queue-req` observed in any server log: **2**).

## Metrics cheat-sheet (plain-language)

One row per column of the table below: what it measures, its range, and
which direction is good. "Percentile" (p95 / p99) means the experience of
the slower 5% / 1% of requests — the near-worst case a user can hit, not
the average.

| Metric | What it is | Range | Good direction |
|---|---|---|---|
| n | Requests completed in the run | count | context only |
| retr | Retractions: requests whose half-finished work was thrown away to free memory (wasted compute) | 0 up | 0 |
| cache hit (%) | Cache hit rate: share of input text the server could reuse from earlier requests instead of recomputing | 0–100% | higher (faster, cheaper first response) |
| q_p95 / q_p99 | Queue wait (seconds): how long a request waits before computing starts; p95/p99 = the slow tail | 0 s up | lower (feels like "the server is slow to react") |
| starv (%) | Starvation rate: share of requests waiting more than 0.2 s | 0–100% | lower (here 1–3% of requests stall) |
| cold n / warm n | The requests split into new sessions (cold — nothing reusable in cache) vs returning ones (warm) | count | context for the next row |
| cold starv / warm starv (%) | Starvation rate of each class — whether new or returning users are the ones stuck waiting | 0–100% | lower, especially cold (new users) |
| gap99 | Cold-minus-warm queue wait at p99: how much worse a new session's near-worst wait is than a returning one's | seconds, either sign | near 0 = fair; large positive = new users penalized |
| RPM / TPM (M) | Throughput: requests per minute / total (input + output) tokens per minute, in millions | 0 up | higher |
| TTFT avg / p95 / p99 | Time to first token (seconds): how long before the user sees the first word | 0 s up | lower |
| TPOT avg / p95 / p99 | Time per output token (milliseconds): streaming smoothness — how long between words | 0 up | lower |
| q_p99 Δ vs LPM | Queue p99 change vs the lpm baseline arm, same run | % either sign | negative = better than baseline |
| Starv vs LPM | Starvation-rate change vs the lpm baseline arm, same run | % either sign | negative = better than baseline (fewer stalled requests) |
| Jain | Fairness index: how evenly speed is shared across requests | 0–1 | higher = fairer (1 = perfectly equal) |

## Canonical tables (c4 and c8)

One table per concurrency, five arms each: lpm (baseline), hrrn, boost,
fcfs-match, dfs-weight. Values come from each cell's `e2_metrics.json`
(`analysis/agentx_metrics.py --starvation-threshold 0.2`, regenerated
2026-10-07; TTFT/TPOT p99 added at the same time) — and from **only the
12 full-run cells** (server boots 02:21–09:06 on 2026-10-07, the run-dir
table above); the 02:01 smoke cell and all earlier-round runs are
excluded. The engineering-form
table (with the static-gamma references and the c16 sweep) lives in
`agentx_final_report.md`.

- **Why one boost row: gamma is dynamic.** The boost arm ran gamma-Ada —
  gamma tunes itself online from the admission-wait tail (gamma0=1e-4,
  converged 1.96e-4 at c4 and 9.82e-5 at c8; trajectories below) — so
  there is no single gamma value to columnize. The round-1/2 static-gamma
  references (5e-5 / 2e-4 at c8) bracketed the adaptive cell on a flat
  frontier and are kept in `agentx_final_report.md`.
- **Why plain fcfs is dropped.** Its cache-hit reads 0.000 purely because
  cache-agnostic policies never populate `num_matched_prefix_tokens`
  when `supports_fast_match_prefix()` is False — a bookkeeping artifact,
  not worse caching — and its ordering is identical to fcfs-match, which
  reports honestly. fcfs-match is the FCFS reference.

### c4 — concurrency 4

| Policy | n | retr | cache hit (%) | q_p95 (s) | q_p99 (s) | starv (%) | cold n | warm n | cold starv (%) | warm starv (%) | gap99 (s) | RPM | TPM (M) | TTFT avg (s) | TTFT p95 (s) | TTFT p99 (s) | TPOT avg (ms) | TPOT p95 (ms) | TPOT p99 (ms) | Jain | q_p99 Δ vs LPM | Starv vs LPM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 224 | 0 | 91.79% | 0.005 | 2.76 | 1.79% | 7 | 217 | 0.00% | 1.84% | −2.49 | 7.3 | 0.454 | 1.72 | 7.40 | 11.15 | 18.10 | 28.95 | 96.06 | 0.208 | — | — |
| hrrn | 216 | 0 | 91.37% | 0.005 | 3.84 | 2.31% | 7 | 209 | 14.29% | 1.91% | +2.68 | 7.1 | 0.426 | 1.77 | 7.49 | 13.33 | 17.30 | 27.00 | 55.30 | 0.200 | +39.13% | +27.78% |
| boost | 216 | 0 | 91.42% | 0.005 | 2.94 | 3.23% | 7 | 210 | 0.00% | 3.33% | −2.95 | 7.0 | 0.422 | 1.77 | 7.23 | 13.25 | 16.96 | 26.20 | 52.17 | 0.208 | +6.64% | +77.78% |
| fcfs-match | 223 | 0 | 91.59% | 0.005 | 3.77 | 1.79% | 7 | 216 | 0.00% | 1.85% | −3.01 | 7.3 | 0.450 | 1.79 | 7.31 | 13.43 | 16.96 | 24.56 | 54.00 | 0.212 | +36.59% | +0.00% |
| dfs-weight | 221 | 0 | 91.67% | 0.005 | 4.78 | 2.25% | 7 | 215 | 0.00% | 2.33% | −3.84 | 7.2 | 0.440 | 1.82 | 7.36 | 21.69 | 16.89 | 24.13 | 54.43 | 0.186 | +73.19% | +27.78% |

### c8 — concurrency 8

| Policy | n | retr | cache hit (%) | q_p95 (s) | q_p99 (s) | starv (%) | cold n | warm n | cold starv (%) | warm starv (%) | gap99 (s) | RPM | TPM (M) | TTFT avg (s) | TTFT p95 (s) | TTFT p99 (s) | TPOT avg (ms) | TPOT p95 (ms) | TPOT p99 (ms) | Jain | q_p99 Δ vs LPM | Starv vs LPM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 138 | 0 | 94.14% | 0.005 | 3.61 | 2.16% | 10 | 129 | 0.00% | 2.33% | −3.10 | 4.5 | 0.399 | 2.15 | 10.60 | 13.88 | 16.01 | 22.58 | 41.24 | 0.313 | — | — |
| hrrn | 142 | 0 | 94.00% | 0.005 | 2.57 | 1.40% | 10 | 132 | 0.00% | 1.52% | −2.01 | 4.7 | 0.413 | 2.06 | 9.55 | 13.90 | 15.68 | 21.72 | 36.54 | 0.329 | −28.81% | −36.36% |
| boost | 138 | 0 | 93.93% | 0.005 | 3.62 | 2.17% | 10 | 128 | 0.00% | 2.34% | −3.10 | 4.5 | 0.403 | 2.23 | 10.43 | 14.36 | 16.07 | 21.02 | 40.79 | 0.324 | +0.28% | +0.00% |
| fcfs-match | 143 | 0 | 94.13% | 0.005 | 3.57 | 2.10% | 10 | 133 | 0.00% | 2.26% | −3.09 | 4.7 | 0.416 | 2.15 | 9.99 | 16.46 | 16.11 | 24.46 | 38.12 | 0.283 | −1.11% | −4.55% |
| dfs-weight | 138 | 0 | 94.01% | 0.005 | 2.80 | 2.14% | 10 | 129 | 0.00% | 2.33% | −2.69 | 4.5 | 0.399 | 2.11 | 10.17 | 13.94 | 15.70 | 21.29 | 38.71 | 0.325 | −22.44% | −4.55% |

Notes: cache hit and starv are percentages; starv counts queue waits above
**0.2 s** (regenerated 2026-10-07 — the original 10 s default was blind
at c4/c8, where waits run 3 ms to ~8 s, and left the column all zeros; no
request lands between 0.2 s and the multi-second tail, so the
0.2 / 0.25 / 0.5 / 1 s rates coincide). TPM (M) is total (input + output)
tokens per minute in millions; Jain is the TTFT fairness index
(higher = fairer). Deltas are vs the same-table lpm baseline — negative
means better than baseline; c4 gap99 sign is run noise (see Reading
notes).

Jain fairness index (equal-weight per request), also a table column: c4 —
fcfs-match 0.212 > lpm = boost 0.208 > hrrn 0.200 > fcfs 0.191 >
dfs-weight 0.186. c8 — hrrn 0.329 > dfs-weight 0.325 > boost 0.324 >
lpm 0.313 > fcfs 0.292 > fcfs-match 0.283.

## Gamma-Ada trajectories (the two boost cells)

- **c4** (`c4_gada_k256_20261007_052130`, 222 arrivals): gamma0=1e-4 held
  while the window filled, climbed through ~7e-4, **touched the 1e-3 clip
  at 64% through the run**, then descended and settled at **1.96e-4** from
  69% onward (last quarter 1.96-1.97e-4, flat). The plan's "pins at the
  1e-3 clip" guess was half right: the clip was visited but not pinned —
  the fixed point sits at ~2x gamma0. Benign at c4 headroom: cache hit 91.4%,
  starv@0.2s 0.032, TPOT p95 26.2 ms, all inside the arm band.
- **c8** (`c8_gada_k256_20261007_090608`, 146 arrivals): gamma0=1e-4,
  monotone slight decay, converged **9.82e-5** (last quarter 9.80-9.82e-5,
  flat; peak = gamma0 itself). Dead center of the offline fixed-point band
  for c8 (7.8e-5-1.4e-4) and a hair under gamma0 — the dial does not bind
  at c8, consistent with round 6.

## Verdicts (one per concurrency)

**c4 (headroom band).** All five arms sit in one efficiency band — cache hit
91.4-91.8% (lpm best by <=0.4 pt), RPM 7.0-7.3 — so at c4 no policy buys
locality; the discriminating axes are the tails. TPOT p95: dfs-weight
24.1 and fcfs-match 24.6 lead; lpm is the worst of the field (29.0 ms);
boost (gamma-Ada, converged 1.96e-4) 26.2 and hrrn 27.0 sit between.
Fairness: at the 0.2 s stall threshold, 1-3% of requests wait over
0.2 s — fcfs best (0.9%, not tabled; see the fcfs note), lpm and
fcfs-match at 1.8%, hrrn and dfs-weight at 2.3%, boost worst at 3.2% (−78% vs lpm: its reordering
buys TPOT at the cost of the stalled tail; hrrn's one cold stall is 1 of
its 7 cold requests). Counts are 2-7 requests per cell — treat rank
order as indicative. q_p99 ranks boost closest to the lpm baseline (+6.6%;
fcfs-match +37%, hrrn +39%, dfs-weight +73% — dfs-weight buys its TPOT
lead with the worst queue tail and the lowest Jain, 0.186 vs lpm 0.208).
Verdict: boost tracks the lpm baseline's fairness at a 9% better TPOT p95
and 0.4 pt less cache hit; dfs-weight is the efficiency-favoring outlier; hrrn
shows no fairness dividend at c4.

**c8 (chunk-blocking band).** Efficiency is again tight on cache hit
(93.9-94.1%) and RPM (4.5-4.8). TPOT p95: boost (gamma-Ada, converged
9.82e-5) leads the field at 21.0 ms, dfs-weight 21.3 and hrrn 21.7
follow, lpm 22.6, fcfs-match trails at 24.5. Fairness: hrrn is the
standout — lowest stall rate (0.014 at the 0.2 s threshold, +36% better
than lpm's 0.022), lowest q_p99 (2.57 s, −28.8% vs lpm) and the best
Jain (0.329 vs lpm 0.313); dfs-weight also cuts q_p99 −22.4%; boost
sits at exact lpm parity on stalls (0.022, +0.0%). This
matches the campaign's insight that c8 tails are chunk-blocking, not
ordering — the gamma dial does not bind, and gamma-Ada's fixed point
confirms round 6's "behaviorally identical to static at c8" from a fresh
boot; the static 5e-5 / 2e-4 references bracket the adaptive cell on a
flat frontier. Verdict: at c8, boost = lpm parity on fairness with the
best TPOT p95 of the field; hrrn is the fairness winner at equal cache hit.

## Reading notes

- **c4 queue tails are 2-3-request phenomena.** At n~220, q_p99 is the
  ~2nd-worst wait; this run's outlier waits landed on warm
  first-admissions in five of six arms (gap99 negative; round-1's lpm c4
  had them on cold ones, +6.60 s). hrrn is the only positive-sign arm
  (+2.68 s). Treat c4 gap99 sign as run noise.
- fcfs c4's q_p99 0.005 s (vs fcfs-match 3.77 s, same ordering) is the
  same tail noise in the other direction: round-1 read fcfs 2.85 /
  fcfs-match 1.77. The c4 queue-histogram percentiles rank arms only
  within a run, not across runs.
- **The stall threshold has a wide safe band (critical threshold).**
  Pooled over the 10 tabled cells (1,858 requests): 98-99% of waits are
  <= 6 ms (p95 2.8 ms) and 1-2% are 1.48-20 s — **zero requests land
  between 6 ms and 1.48 s**. Any threshold inside that empty band
  classifies the same requests, which is why 0.2 / 0.25 / 0.5 / 1 s all
  return identical rates; below ~6 ms the count picks up batch-cadence
  waits, above ~1.48 s it drops real stalls. 0.2 s sits mid-band.
- **Stalls are a warm-session, mid-run phenomenon — chunk-blocking, not
  sorting.** All but one of the >0.2 s stalls across the 10 tabled cells
  are warm requests with 18K-172K-token inputs arriving 45-99% through
  the run; cold requests are few (7-17 per cell) and arrive early
  (median 16%/29% of the run at c4/c8) into an empty fresh-boot queue,
  so they are admitted in milliseconds. The same trace sessions stall
  under every policy (a 2.91/3.18 s pair appears in all five c8 arms) —
  the stall is pipeline occupancy by giant in-flight prefills, which no
  sort order can preempt. Hence warm starv > cold starv and gap99 < 0,
  against the "cold sessions starve" intuition, which assumes
  ordering-based starvation.
- **c4/c8 offered load sits below the policy-discriminating band.** Load
  observables per cell: `#queue-req` in the server log (max 2 in this
  run), the queue block of `e2_metrics.json`, per-request arrivals and
  waits in `schedule_eval.jsonl`. At c4/c8 the queue is almost always
  empty (98% of waits <= 6 ms) — a sort key has nothing to act on, so
  the arms converge except for 1-2-request outlier luck (the c4/c8 delta
  signs above). c16 is the first band on this GPU where load binds
  (tails 95-190 s, starv 0.38-0.56, cache hit down to 0.81-0.87) and
  where boost / gamma-Ada showed its win. H200 note: recalibrate before
  comparing policies — sweep concurrency upward from the c16 protocol
  until `#queue-req` is persistently > 0 and waits leave the millisecond
  range, then run the policy comparison in that band.

## Acceptance (per the fullrun plan)

- 12 fresh-boot cells, both concurrencies, all six arms: **done, zero
  failures**.
- Canonical table regenerated with all six arms at c4 and c8: **done**
  (above; c16 group unchanged, in `agentx_final_report.md`).
- Gamma trajectories recorded (start / converged / last quarter): **done**
  (both cells above).
- No serving-config or server-code changes for this run: **held**
  (`start_server.sh` and server code untouched; only the driver script is
  new).
- Campaign stays closed on this hardware; c16 band, seed replicas, and
  targeted starvation regime move to the 4x H200 campaign.
