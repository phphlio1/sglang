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
  self-logging) over `scripts/run_e2.sh policy <arm> 16`. Serving config
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

Regenerated canonical rows (`analysis/print_results_table.py`; the full
output also retains the landed Oct 6 static-gamma boost rows and the c4/c8
groups — not quoted here):

| Policy | c | γ | n | hit | q_p95 (s) | q_p99 (s) | starv | cold n | cold starv | warm n | warm starv | gap99 (s) | RPM | oTPM | TTFT (ms) | TPOT p95 (ms) | retr | q_p99 Δ vs LPM | Anti-starv. vs LPM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 16 | — | 178 | 0.858 | 79.750 | 183.800 | 0.409 | 17 | 0.529 | 164 | 0.396 | +155.689 | 5.8 | 5517 | 22799 | 92.7 | 0 | — | — |
| boost | 16 | 1e-1 † | 160 | 0.822 | 93.600 | 167.200 | 0.482 | 18 | 0.389 | 146 | 0.493 | +12.233 | 5.2 | 4837 | 27742 | 168.1 | 0 | −9.0% | −17.8% |
| hrrn | 16 | — | 175 | 0.841 | 90.250 | 174.429 | 0.436 | 21 | 0.571 | 158 | 0.418 | +46.177 | 5.7 | 5062 | 25198 | 126.6 | 1 | −5.1% | −6.6% |
| fcfs-match | 16 | — | 150 | 0.801 | 91.750 | 174.500 | 0.588 | 21 | 0.524 | 132 | 0.598 | −9.411 | 4.9 | 5017 | 28224 | 128.9 | 6 | −5.1% | −43.8% |
| fcfs | 16 | — | 149 | 0.785 | 91.333 | 169.600 | 0.559 | 20 | 0.450 | 132 | 0.576 | +45.231 | 4.9 | 3822 | 29614 | 129.6 | 6 | −7.7% | −36.7% |
| dfs-weight | 16 | — | 157 | 0.830 | 95.000 | 340.000 | 0.388 | 17 | 0.353 | 143 | 0.392 | +153.981 | 5.1 | 5087 | 29119 | 104.7 | 6 | +85.0% | +5.1% |

Column definitions: starv = share of requests with queue wait > 10 s; hit =
token-weighted first-attempt prefix hit rate; q_p95/p99 = queue-wait
percentiles (server histogram); gap99 = cold-class p99 − warm-class p99
queue wait (schedule-eval log, first admission stamp); TTFT = client-observed
mean; retr = retraction count. **†** the boost row's gamma is wall-clock, in
1/s: arrival is stamped in wall seconds and work is scaled by the measured
token rate, so it is not comparable with per-token gammas — the cross-mode
identity is γ_wall = γ_tokens · R. **‡** (no row here needs it) marks pre-fix
fcfs rows whose hit reads ~0.000 — the bookkeeping artifact this sweep's
fcfs row retires (0.785, unmarked).

Reading of the field:

- **hit**: lpm 0.858 best of six (field 0.785–0.858); boost 0.822 mid-field.
- **queue tail (q_p99)**: boost 167.2 s best (−9.0% vs lpm); fcfs 169.6,
  hrrn 174.4, fcfs-match 174.5, lpm 183.8; dfs-weight 340.0 worst (+85.0%).
- **cold-class starvation**: dfs-weight 0.353 and boost 0.389 lead; lpm
  0.529 and fcfs-match 0.524 trail.
- **new-session penalty (gap99)**: fcfs-match −9.4 s and boost +12.2 s lead;
  lpm +155.7 s and dfs-weight +154.0 s trail — the wall-clock boost arm
  nearly closes the cold-vs-warm p99 gap the lpm baseline carries.
- **TPOT p95**: lpm 92.7 ms best; boost 168.1 ms worst — the sweep's one
  regression flag (check 4).
- **throughput**: oTPM 3822 (fcfs) – 5517 (lpm); boost 4837.

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
3. **fcfs admission bookkeeping — PASS.** fcfs hit reads **0.785** — real
   reuse, confirmed by its eval log (mean matched prefix ≈ 67k tokens, 79%
   of admissions nonzero). fcfs-match 0.801 stays the honest sort-time
   reference. The landed pre-fix fcfs row read 0.000 — the artifact this
   fix retires (*landed*); the table script now scopes the ‡ marker to
   those rows.
4. **Boost wall vs the best landed c16 static cell (γ=3e-6, the
   cold-starvation pick) — MIXED.** Comparator (*landed*): hit 0.820, cold
   starv 0.381, cold p99 111.0 s, q_p99 188.7 s, TPOT p95 97.8 ms. This
   sweep's wall cell: hit 0.822, cold starv 0.389, cold p99 116.1 s, q_p99
   167.2 s, TPOT p95 168.1 ms. Matched on hit (±0.002) and cold starvation
   (+0.008); cold p99 +5.1 s; q_p99 better by 21.5 s cross-boot and −9.0%
   vs the same-run lpm baseline. **TPOT p95 regressed**: 168.1 ms is the
   sweep's worst by +29.7% over the next-worst arm (fcfs 129.6) and sits
   beyond the landed same-config fresh-boot spread (γ=1e-4 pair
   115.7 ↔ 143.1 ms; *landed*) and beyond every landed c16 boost cell
   (97.8–143.1). Context that isolates the effect: the wall arm's reuse is
   intact (≈74.6k tokens mean matched prefix, 82% of admissions nonzero)
   and oTPM 4837 is mid-field — a decode-tail effect, not a throughput
   collapse. **Follow-up, inherited by the H200 campaign: a static-wall
   arm at the offline γ̂ ≈ 0.09 1/s** — if the static arm shows the same
   tail, the clock mode is implicated; if not, gamma-Ada's transient is.
5. **lpm hit — PASS (no regression).** lpm 0.858 is the sweep's best
   (field 0.785–0.858). Landed context (*landed*): the landed lpm c16 row
   0.869 is a round-1 warm-start row (the landed report measured the
   fresh-vs-warm confound at +0.7 pts hit); against it the fresh 0.858 is
   −1.1 pts — the same order as that confound, i.e. explained by
   warm-start state and boot variance.
6. **Canonical table regeneration — PASS.** The table regenerates with all
   six new rows; the wall boost row is clock-labeled (γ = 1e-1 †, footnote
   e — wall and per-token gammas never mixed unlabeled) and ‡ is scoped to
   pre-fix hit≈0 rows (footnote b rewritten). Per latest-per-seed semantics
   the new rows supersede the landed same-cell rows (c16 lpm/fcfs/
   fcfs-match are now the post-fix fresh rows); the landed snapshot remains
   in `agentx_final_report.md`.

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
≤ 1.7 s; fcfs reads real cache reuse (0.785); lpm keeps the best hit of the
field (0.858). The wall-clock boost arm posts the sweep's best queue tail
(q_p99 167.2 s, −9.0% vs lpm — the first c16 boost row to come in under the
lpm baseline of its own table) at matched hit and cold-class starvation,
and nearly closes the new-session p99 gap (gap99 +12.2 s vs lpm's +155.7 s).
One open flag: **TPOT p95 168.1 ms** — the sweep's worst by 30%, beyond the
landed same-config boot spread, with reuse and throughput intact: a
decode-tail effect. The H200 campaign inherits the follow-up (static-wall
arm at γ̂ ≈ 0.09 1/s) alongside the wall-clock config this sweep validated.

## Reading notes

- **TPOT p95 flag**: see check 4. Read TPOT by p95 only — at c16 the TPOT
  *mean* is outlier-dominated (fcfs/fcfs-match means read 466–472 ms
  against p95s of 129/128 ms) and is not a usable column.
- **dfs-weight tails**: q_p99 340.0 s and cold p99 355.5 s — the sweep's
  worst by far (next-worst q_p99: lpm 183.8 s) with a mid-field TPOT p95
  (104.7 ms): the arm's known TPOT-for-tails trade, now measured at c16.
- **Single-boot caveat**: one boot per cell, one seed; single-cell TPOT p95
  deltas at c16 carry boot variance (the landed same-config fresh pair
  spread 115.7 ↔ 143.1 ms bounds it — *landed*). The flag's 168.1 ms sits
  beyond that bound; the queue-tail and hit results are cross-arm
  consistent and not boot-sensitive.
- **TTFT**: client-observed mean 22.8–29.6 s, p95 93.5–101.0 s across arms —
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
  scoped ‡: **done** (this file quotes the six new rows; the script's full
  output remains the canonical artifact).
- Held to the plan: no targeted-regime knobs, no second seed, no
  tokens-clock boost arm, no c4/c8 repeats.
- RTX wall-clock numbers are never compared against H200 numbers; the H200
  campaign recalibrates the pressure band first and lands in its own
  per-hardware results root.
