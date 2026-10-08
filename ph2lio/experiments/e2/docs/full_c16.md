# c16 validation sweep (2026-10-08): six-arm post-fix comparison — RTX 6000 Pro

6 arms × c16 = 6 cells, fresh boot per cell — the acceptance sweep for the
four boost-scheduler fixes (wall-clock boost mode, unit-switched gamma-Ada,
the 60 s in-batch follower-pin bound, and fcfs admission-time prefix-match
bookkeeping; commits `bdc67a5720` scheduler, `0083303206` harness + this
driver). Every cell runs the fixed scheduler and the config the H200
campaign inherits. Companion docs: `agentx_final_report.md` (landed campaign
summary and landed c16 table), `full_run.md` (the c4/c8 full run),
`agentx_experiment.md` (protocol), `RUN.md` (how to run).

**Scope: this report uses only the 2026-10-08 c16 cells.** Landed
(2026-10-06, pre-fix) numbers appear only inside the pre-registered checks
that require a comparator, each tagged *landed*; nothing from c4/c8.

## Setup

- Driver: `scripts/run_fullrun_c16.sh` (new; continue-on-failure,
  self-logging) over `scripts/run_e2.sh policy <arm> 16`; at sweep end the
  driver prints the canonical tables (`analysis/print_results_table.py`,
  counted per full run — this sweep's block prints first). Serving config
  `scripts/start_server.sh` **unchanged**.
- Qwen3.8-27B-FP8, single RTX 6000 Pro, DURATION=1800, seed 42, fresh boot
  per cell (`STOP_SERVER=always`), AIPerf `inferencex-agentx-mvp`, Weka
  agentic traces (256K-capped). All six cells stamp
  `sglang_commit 0083303206`.
- Fixes active in every cell: follower-pin bound 60 s
  (`in_batch_prefix_caching_max_pin_s`, the default) and fcfs admission
  bookkeeping. The boost arm additionally runs the wall clock with
  gamma-Ada: `BOOST_CLOCK=wall`, gamma0 `BOOST_GAMMA=0.1` 1/s (the wall
  default), `BOOST_ADAPTIVE_GAMMA=1`, `BOOST_BIN_SIZE=256`.
- Arm order (risk-earliest): lpm, dfs-weight, fcfs, fcfs-match, hrrn, boost
  last.
- Pre-flight smoke gate (2026-10-07): boost/wall/gamma-Ada c4 smoke
  **PASSED** (`smoke/smoke_c4_20261007_164514`; 70-line eval log via the new
  clock-suffixed path `schedule_eval_ada_wall_k256.jsonl`, run_meta
  boost-clock/gamma stamps verified) — this gated the new `run_e2.sh`
  plumbing before a ~4.5 h sweep.

## Execution

Sweep 01:18:52 → 05:43:20 (~4 h 25 m). **6/6 cells landed, zero failures.**
Driver log: `logs/fullrun_c16_20261008_011852.log`.

| # | cell | run dir (under `results/1x-rtx6000pro/agentx/`) | window |
|---|---|---|---|
| 1 | lpm c16 | `lpm/c16_20261008_011852` | 01:18:52–02:03:00 |
| 2 | dfs-weight c16 | `dfs-weight/c16_20261008_020300` | 02:03:00–02:47:04 |
| 3 | fcfs c16 | `fcfs/c16_20261008_024704` | 02:47:04–03:31:08 |
| 4 | fcfs-match c16 | `fcfs-match/c16_20261008_033108` | 03:31:08–04:15:17 |
| 5 | hrrn c16 | `hrrn/c16_20261008_041517` | 04:15:17–04:59:17 |
| 6 | boost c16 (wall, gamma-Ada) | `boost/c16_gada_wall_k256_20261008_045917` | 04:59:17–05:43:20 |

hrrn c16 is the first hrrn c16 cell ever landed (the landed attempt was
killed 2026-10-06 19:37 at hardware retirement). Health gauges: retractions
0 (lpm, boost), 1 (hrrn), 6 (dfs-weight, fcfs, fcfs-match); queue depth mean
2.10–2.28, max 5–7; the >128-request FCFS fallback never engaged.

## Results — the six 2026-10-08 c16 cells

The c16 block of this sweep as the table script counts it
(`analysis/print_results_table.py`: results are counted **per full run** —
the cells sharing a sweep's `sglang_commit` stamp, one block per full run,
latest first — in the `full_run.md` table format). The landed Oct 5/6/7 c16
runs sit in their own blocks and never mix into this table:

### c16 — concurrency 16

| Policy | γ | n | retr | cache hit (%) | q_p95 (s) | q_p99 (s) | starv (%) | cold n | warm n | cold starv (%) | warm starv (%) | gap99 (s) | RPM | TPM (M) | TTFT avg (s) | TTFT p95 (s) | TTFT p99 (s) | TPOT avg (ms) | TPOT p95 (ms) | TPOT p99 (ms) | Jain | q_p99 Δ vs LPM | Starv vs LPM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | — | 178 | 0 | 85.81% | 79.750 | 183.800 | 40.88% | 17 | 164 | 52.94% | 39.63% | +155.69 | 5.8 | 0.599 | 22.80 | 93.46 | 127.21 | 35.77 | 92.67 | 199.70 | 0.378 | — | — |
| hrrn | — | 175 | 1 | 84.07% | 90.250 | 174.429 | 43.58% | 21 | 158 | 57.14% | 41.77% | +46.18 | 5.7 | 0.571 | 25.20 | 98.26 | 153.07 | 38.44 | 126.63 | 222.36 | 0.352 | −5.10% | +6.60% |
| boost | 1e-1 † | 160 | 0 | 82.25% | 93.600 | 167.200 | 48.17% | 18 | 146 | 38.89% | 49.32% | +12.23 | 5.2 | 0.512 | 27.74 | 100.99 | 133.67 | 46.97 | 168.14 | 279.23 | 0.391 | −9.03% | +17.83% |
| fcfs-match | — | 150 | 6 | 80.12% | 91.750 | 174.500 | 58.82% | 21 | 132 | 52.38% | 59.85% | −9.41 | 4.9 | 0.467 | 28.22 | 96.83 | 138.50 | 466.18 | 128.90 | 1459.59 | 0.462 | −5.06% | +43.88% |
| fcfs | — | 149 | 6 | 78.49% | 91.333 | 169.600 | 55.92% | 20 | 132 | 45.00% | 57.58% | +45.23 | 4.9 | 0.455 | 29.61 | 100.71 | 122.86 | 471.82 | 129.65 | 1309.15 | 0.436 | −7.73% | +36.79% |
| dfs-weight | — | 157 | 6 | 82.99% | 95.000 | 340.000 | 38.75% | 17 | 143 | 35.29% | 39.16% | +153.98 | 5.1 | 0.525 | 29.12 | 98.86 | 318.20 | 38.18 | 104.72 | 319.89 | 0.183 | +84.98% | −5.21% |

Columns (the table script's footnotes (a)–(e)): cache hit = token-weighted
first-attempt prefix hit rate; starv = share of requests with queue wait >
the cell's recorded threshold — **10 s** in these cells (c16 waits run tens
of seconds; the c4/c8 tables in `full_run.md` use 0.2 s); q_p95/p99 =
queue-wait percentiles (server histogram); gap99 = cold-class p99 −
warm-class p99 queue wait (schedule-eval log, first admission stamp); RPM =
requests/min; TPM (M) = total (input + output) tokens/min in millions;
TTFT/TPOT = client-observed avg/p95/p99 (TTFT in s, TPOT in ms); Jain = TTFT
fairness index (higher = fairer); retr = retraction count. Δ columns are vs
the same-table lpm baseline — negative = better. **†** the boost row's gamma
is wall-clock, in 1/s: arrival is stamped in wall seconds and work is scaled
by the measured token rate, so it is not comparable with per-token gammas —
the cross-mode identity is γ_wall = γ_tokens · R. **‡** (no row here needs
it) marks pre-fix fcfs rows whose hit reads ~0.00% — the bookkeeping
artifact this sweep's fcfs row retires (78.49%, unmarked).

Reading of the field:

- **hit**: lpm 85.81% best of six (field 78.49–85.81%); boost 82.25%
  mid-field.
- **queue tail (q_p99)**: boost 167.2 s best (−9.03% vs lpm); fcfs 169.6,
  hrrn 174.4, fcfs-match 174.5, lpm 183.8; dfs-weight 340.0 worst (+84.98%).
- **cold-class starvation**: dfs-weight 35.29% and boost 38.89% lead; lpm
  52.94% and fcfs-match 52.38% trail.
- **new-session penalty (gap99)**: fcfs-match −9.41 s and boost +12.23 s
  lead; lpm +155.69 s and dfs-weight +153.98 s trail — the wall-clock boost
  arm nearly closes the cold-vs-warm p99 gap the lpm baseline carries.
- **TPOT p95**: lpm 92.67 ms best; boost 168.14 ms worst — the sweep's one
  regression flag (check 4).
- **fairness (Jain)**: fcfs-match 0.462 and fcfs 0.436 lead; boost 0.391
  sits just over lpm 0.378; hrrn 0.352; dfs-weight 0.183 is the outlier low,
  riding its queue tail.
- **throughput**: TPM (M) 0.455 (fcfs) – 0.599 (lpm); boost 0.512 mid-field
  (oTPM 4837).

## The six pre-registered checks

1. **gamma-Ada wall convergence — PASS.** gamma0 0.1 1/s; equal-count
   quarters over the cell's 179 load admissions (eval log; the 1
   health-check record excluded): **0.100 / 0.074 / 0.065 / 0.061 1/s**,
   last-quarter mean 0.061, range 0.045–0.100 — monotone down, no FCFS-ward
   climb (the failure mode the landed tokens-clock cells showed as the cache
   warmed — *landed*). Inside the pre-registered wall bracket
   **[0.017, 0.62] 1/s**; the wall-unit clip [1e-3, 10] 1/s never visited.
   Offline estimate over the same waits: γ̂ = ln5/(w99−w95) = **0.091 1/s**
   (p95 93.1 s, p99 110.7 s) — the converged 0.061 sits at 0.67× γ̂, same
   order. Cross-mode identity at the cell's measured total-token rate
   R = 8.54k tok/s: γ_tokens = 0.061/R ≈ **7e-6**, inside the landed
   tokens-clock bracket [3e-6, 2e-5] (*landed*).
2. **Follower-pin bound (60 s) — PASS.** Sweep-wide in-batch dedup follower
   first-admission waits: boost 4/179 (2.2%) at 0.2–1.7 s; hrrn 4 (two at
   55.5/58.8 s); dfs-weight 1 (53.7 s); lpm/fcfs/fcfs-match 0. **Nothing
   exceeds 60 s** — the hrrn/dfs waits sit just under the bound, i.e. the
   bound is doing the capping; unbounded (*landed*, γ=1e-4 fresh cell) the
   same stamp ran to 108.3 s (7 followers, p50 12.4 s).
3. **fcfs admission bookkeeping — PASS.** fcfs hit reads **78.49%** — real
   reuse, confirmed by its eval log (mean matched prefix ≈ 67k tokens, 79%
   of admissions nonzero). fcfs-match 80.12% stays the honest sort-time
   reference. The landed pre-fix fcfs row read 0.00% — the artifact this
   fix retires (*landed*); the table script now scopes the ‡ marker to
   those rows.
4. **Boost wall vs the best landed c16 static cell (γ=3e-6, the
   cold-starvation pick) — MIXED.** Comparator (*landed*): hit 82.0%, cold
   starv 38.1%, cold p99 111.0 s, q_p99 188.7 s, TPOT p95 97.8 ms. This
   sweep's wall cell: hit 82.25%, cold starv 38.89%, cold p99 116.1 s,
   q_p99 167.2 s, TPOT p95 168.14 ms. Matched on hit (±0.2 pts) and cold
   starvation (+0.8 pts); cold p99 +5.1 s; q_p99 better by 21.5 s
   cross-boot and −9.03% vs the same-run lpm baseline. **TPOT p95
   regressed**: 168.14 ms is the sweep's worst by +29.7% over the
   next-worst arm (fcfs 129.65) and sits beyond the landed same-config
   fresh-boot spread (γ=1e-4 pair 115.7 ↔ 143.1 ms; *landed*) and beyond
   every landed c16 boost cell (97.8–143.1). Context that isolates the
   effect: the wall arm's reuse is intact (≈74.6k tokens mean matched
   prefix, 82% of admissions nonzero) and TPM 0.512 (oTPM 4837) is
   mid-field — a decode-tail effect, not a throughput collapse.
   **Follow-up, inherited by the H200 campaign: a static-wall arm at the
   offline γ̂ ≈ 0.09 1/s** — if the static arm shows the same tail, the
   clock mode is implicated; if not, gamma-Ada's transient is.
5. **lpm hit — PASS (no regression).** lpm 85.81% is the sweep's best
   (field 78.49–85.81%). Landed context (*landed*): the landed lpm c16 row
   86.9% is a round-1 warm-start row (the landed report measured the
   fresh-vs-warm confound at +0.7 pts hit); against it the fresh 85.81% is
   −1.1 pts — the same order as that confound, i.e. explained by
   warm-start state and boot variance.
6. **Canonical table regeneration — PASS.** The table script now counts
   results **per full run** (cells grouped by their sweep's commit stamp,
   one block per full run, latest first) and both full-run drivers print it
   at sweep end; this sweep's block carries all six rows and nothing else —
   the landed Oct 5/6/7 c16 runs sit in their own blocks, never mixed in.
   The wall boost row is clock-labeled (γ = 1e-1 †, footnote e — wall and
   per-token gammas never mixed unlabeled) and ‡ is scoped to pre-fix
   hit≈0% rows (footnote b). The landed snapshot remains in
   `agentx_final_report.md`.

## gamma-Ada wall trajectory (boost cell detail)

- Method: the cell's schedule-eval log holds 180 records = 179 load
  admissions + 1 harness health check; all stats above exclude the health
  check. Admissions are back-loaded (session-tree fan-out: 11/25/24/119
  admissions per 450 s of the load window), so quarters are taken over the
  admission sequence, not wall time. The table's n counts completed
  requests (160 for boost); the eval log stamps admissions (179) — the
  difference is work still in flight at window close.
- gamma0 0.1 1/s held while the estimator window filled, then stepped down
  monotonically to 0.061 (min 0.045); no FCFS-ward climb at any point — the
  wall clock's cache-state invariance (the fix's design goal) held over the
  whole run.
- Offline fixed point 0.091 1/s (ln5/(w99−w95) over the 179 first-admission
  waits, p95 93.1 s / p99 110.7 s); converged 0.061 = 0.67× offline.

## Verdict

The four fixes hold at c16, the only RTX band where load binds. The
wall-clock gamma-Ada converges monotonically from 0.1 into the bracket
(0.061 1/s, 0.67× the offline fixed point) with no FCFS-ward drift; the
60 s follower-pin bound caps every follower wait in the sweep (worst 58.8 s,
vs 108.3 s unbounded landed) while the boost arm's own followers wait
≤ 1.7 s; fcfs reads real cache reuse (78.5%); lpm keeps the best hit of the
field (85.8%). The wall-clock boost arm posts the sweep's best queue tail
(q_p99 167.2 s, −9.03% vs lpm — the first c16 boost row to come in under the
lpm baseline of its own table) at matched hit and cold-class starvation,
and nearly closes the new-session p99 gap (gap99 +12.23 s vs lpm's
+155.69 s). One open flag: **TPOT p95 168.14 ms** — the sweep's worst by
30%, beyond the landed same-config boot spread, with reuse and throughput
intact: a decode-tail effect. The H200 campaign inherits the follow-up
(static-wall arm at γ̂ ≈ 0.09 1/s) alongside the wall-clock config this
sweep validated.

## Reading notes

- **TPOT p95 flag**: see check 4. Read TPOT by p95 only — the TPOT avg
  column at c16 is outlier-dominated (fcfs/fcfs-match 471.82/466.18 ms
  against p95s of 129.65/128.90 ms; p99s 1309/1460 ms) and is not a usable
  column.
- **dfs-weight tails**: q_p99 340.0 s and cold p99 355.5 s — the sweep's
  worst by far (next-worst q_p99: lpm 183.8 s) with a mid-field TPOT p95
  (104.72 ms) and the field's lowest Jain (0.183): the arm's known
  TPOT-for-tails trade, now measured at c16.
- **Single-boot caveat**: one boot per cell, one seed; single-cell TPOT p95
  deltas at c16 carry boot variance (the landed same-config fresh pair
  spread 115.7 ↔ 143.1 ms bounds it — *landed*). The flag's 168.14 ms sits
  beyond that bound; the queue-tail and hit results are cross-arm
  consistent and not boot-sensitive.
- **TTFT**: client-observed avg 22.80–29.61 s, p95 93.46–100.99 s, p99
  122.86–318.20 s across arms (dfs-weight's p99 rides its queue tail) —
  c16 is the load-binding band on this GPU (see `full_run.md`'s reading
  note); queue depth mean 2.10–2.28, max 5–7.
- The landed round-6 tokens-clock adaptive c16 cell (converged 1.76e-5) has
  no run dir on disk (cleaned; its server logs
  `boost/server_20261006_19*.log` remain) — its numbers are recorded in
  `agentx_final_report.md` and were not used here.

## Acceptance (per the sweep plan)

- 6/6 cells under commit `0083303206`, fresh boots, seed 42, zero
  failures: **done**.
- Six pre-registered checks answered: **done** — five PASS, one MIXED
  (TPOT p95) with the prescribed follow-up.
- Canonical table regenerated with the clock-labeled wall row and the
  scoped ‡: **done** — the table script now counts results per full run and
  both full-run drivers print it at sweep end; this file quotes this
  sweep's c16 block.
- Held to the plan: no targeted-regime knobs, no second seed, no
  tokens-clock boost arm, no c4/c8 repeats.
- RTX wall-clock numbers are never compared against H200 numbers; the H200
  campaign recalibrates the pressure band first and lands in its own
  per-hardware results root.
