# E2 final report — UniBoost boost-policy campaign (RTX 6000 Pro)

Seed-42 campaign, 2026-10-05/07. Companion to `agentx_experiment.md`
(protocol and per-task detail) and `agentx_round2_findings.md` (results
compendium); this file is the self-contained summary. Harness:
`ph2lio/experiments/e2`, AIPerf `inferencex-agentx-mvp`, Weka agentic-coding
traces (256K-capped), Qwen3.8-27B-FP8 single GPU, DURATION=1800, fresh boot
per cell.

**Campaign status: closed on RTX 6000 Pro, pending the 4x H200 campaign**
(which inherits round 4 targeted regime, round 5 seed replicas, and the
hrrn c16 cell).

## Round-6 verdict — gamma-Ada validation (the one remaining GPU run)

`GAMMA_STAR=1e-5 scripts/run_round6.sh` (adaptive c8 + adaptive c16 from a
cold gamma0=1e-4, plus a fresh static gamma*=1e-5 c8 reference), then
`analysis/adaptive_gamma_check.py --gamma-star 1e-5 --bracket 3e-6 2e-5`.

| arm | converged gamma | bracket [3e-6, 2e-5] | hit | cold starv | TPOT p95 (ms) |
|---|---|---|---|---|---|
| boost gamma-Ada c16 | **1.76e-5** | **PASS** | 0.833 | 0.500 | **76.9** |
| boost static 1e-5 c16 (round 2) | 1e-5 (set) | — | 0.838 | 0.600 | 103.3 |
| boost gamma-Ada c8 | 1.34e-4 | FAIL (see below) | 0.940 | 0.000 | 25.5 |
| boost static 1e-5 c8 (fresh ref) | 1e-5 (set) | — | 0.941 | 0.000 | 25.5 |

- **c16 PASS on every criterion.** Converged gamma lands inside the sweep
  bracket, at its FCFS-ward edge (the estimator's fixed point is gamma*'s
  2e-5 neighbor). Behavior vs the fresh static gamma* cell: hit −0.5 pts
  (guard ±2), cold-class starvation 0.500 vs 0.600 (better), TPOT p95
  −25.5% — the best c16 TPOT p95 of the campaign (fresh references: 3e-6
  97.8 ms, fcfs-match 121.1 ms).
- **Trajectories.** c16: gamma0 held while the 2048-window filled to 64
  first admissions (1233 s; warmup admissions feed the window too), first
  update stepped straight to 3.5e-5, below 2e-5 by +1317 s, min 1.34e-5,
  last quarter oscillating 1.34-2.43e-5. c8: gamma0 held 1085 s, settled
  9.82-9.88e-5 for ~600 s, then climbed monotonically to 1.70e-4 as the
  pool warmed and the admission tail thinned (thin tail -> FCFS-ward).
- **c8 bracket FAIL — estimator-bias finding, documented, not patched.**
  The bracket is the c16 sweep's neighbors and does not transfer to c8:
  the offline c8 fixed point is 7.78e-5 (boost own-tail) to 1.36e-4
  (lpm-tail), and the runtime converged 1.34e-4 — on lpm's offline
  estimate. The own-tail fixed point drifted with cache state as the pool
  warmed. Costless at c8: hit/starv/TPOT identical to the static gamma*
  arm (the dial does not bind there).

**Reading:** gamma-Ada reproduces the c16 frontier pick from a cold 1e-4
start with zero configuration; its known scale sensitivity (per-concurrency,
cache-state-dependent fixed point) is a documented caveat for deployments
carrying one gamma across concurrencies.

## Canonical final table

`python3 analysis/print_results_table.py` (default mode; latest-per-seed,
seed replicas n-weighted; lpm baseline bold per concurrency group):

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

Footnotes and reading notes (full text in `agentx_experiment.md` "Final
results"): (a) anti-starvation deltas are vs the same-concurrency lpm
baseline; the >10 s starvation rate is blind below c16-scale waits. (b) ‡
fcfs hit=0.000 is a sort-time bookkeeping artifact (cache-agnostic
policies never populate `num_matched_prefix_tokens` when
`supports_fast_match_prefix()` is False); fcfs has real admission-time
reuse, and fcfs-match is the honest reference. (c) the footnote's boost
cell is the lowest-cold-starvation cell (3e-6), **not** gamma* — gamma*
(1e-5) was picked on hit / TPOT p95 / the offline gamma-hat region, and
its cold starvation (0.600) is the sweep's worst. (d) the lpm/fcfs c16
baselines are round-1 warm-start rows; every c16 boost / fcfs-match /
hrrn row is fresh-boot, with the measured fresh-vs-warm confound
(+0.7 pts hit, +3.2% oTPM, +23.7% TPOT p95 on the 1e-4 pair) leaving
~2.4-5.2 pts of the hit gap structural to boost-vs-lpm.

Frontier plot: `results/agentx/frontier_c16.png` (gamma* = 1e-5 ringed).

## Insights

1. **The fcfs arm was measuring its own bookkeeping.** Plain fcfs's
   hit=0.000 is an artifact: the sort-time match path it skips is never
   compensated in reporting, while admission-time re-matching gives it
   real reuse (fcfs-match c4 hit 0.916 vs lpm 0.912). At c16, ordering
   itself matters — fcfs-match is the worst cache-aware arm (hit 0.805,
   oTPM 4590): matching fixes the bookkeeping, not the locality. Round-1's
   fcfs-vs-lpm gap conflated both effects.
2. **gamma* = 1e-5, but the frontier is honestly flat.** At c16
   fresh-boot, gamma in [3e-6, 1e-4] moves hit by <3 pts and oTPM by
   <2.5%; warm-start state and the pinned-cohort mechanism move both by
   more than gamma does. Every discriminating axis (TPOT p95, best hit,
   offline gamma-hat) favors 1e-5.
3. **The overtaking invariant holds; the 9 beyond-bound pairs are the
   in-batch dedup follower pin.** Zero violations attributable to the
   boost sort across 472 pairs. The "hard" pairs are co-waiting cold
   sessions pinning each other through the shared AgentX system preamble
   under fresh-boot pool pressure — an SGLang-wide design element shared
   with lpm; fcfs pays the same big-cold holds via head-of-line blocking.
   Follow-up lever: bound follower hold time (the 32/32 thresholds are
   env-overridable).
4. **The token clock is cache-state-dependent.** The bound b_gamma(k) is
   denominated in a processed-token counter that includes linear replay
   of matched prefixes: ~3.3-6k tok/s fresh vs ~36.7k warm at c16. At
   fresh rates, gamma=1e-4 buys only ~6-11 s of overtaking (the round-1
   c16 arm was already effectively FCFS at the tail); 1e-5 (~100-181 s)
   is the cell whose bound matches the ~95-190 s queue tail.
5. **c8 tail waits are chunk-blocking, not ordering.** q-wait p99 pins at
   ~3.6 s = the chunked-prefill iteration cadence; the gamma sweep is
   flat there because every bound sits at or above the queue depth.
6. **gamma-Ada works where it matters.** From a cold 1e-4 start it
   converges into the c16 sweep bracket and beats the static gamma* cell
   on TPOT p95 (76.9 vs 103.3 ms) at matched hit and starvation. Its
   fixed point is per-concurrency and drifts with cache state (the c8
   arm climbed to 1.7e-4 as the pool warmed) — carry a per-concurrency
   gamma, or accept FCFS-ward drift as pools warm.
7. **Upstream candidates** (PR only after H200 validation): boost +
   gamma-Ada as the feature; separately, the cache-agnostic
   `num_matched_prefix_tokens` bookkeeping bug (fcfs hit metrics read
   0.000 for every current cache implementation) as a small standalone
   fix. fcfs-match itself stays a local experiment alias.

## Skipped and why (all moved to the 4x H200 campaign)

- **Round 4 (targeted starvation regime)**: the auto-armed watcher fired
  when the round-2 driver was killed (19:24) and was itself killed at
  19:37 before its driver wrote anything; partial cell quarantined under
  `results/quarantine/c16_20261006_192407`.
- **Round 5 (seed replicas 43/44)**: never started on this hardware.
- **hrrn c16**: started 19:24, killed 19:37 (hardware retirement cut),
  quarantined; hrrn c8 landed before the cut and keeps its row.

## Commits (branch `uniboost-boost-policy`)

1. `859a7ef145` — fcfs-match arm + gamma-Ada server code + unit checks
   (off-by-default; no harness paths in server code)
2. `25251db11d` — e2 round 2-6 harness + analysis scripts
3. `2c82ab2323` — `arch.py` CCCL toolkit-check opt-out (own commit)
4. `4f1a287b22` — docs + close-out

Results tree intentionally untracked. Repro: see `RUN.md` and the
"Repro" section of `agentx_round2_findings.md`.
