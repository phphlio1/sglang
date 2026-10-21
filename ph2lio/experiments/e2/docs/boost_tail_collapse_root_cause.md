# Root cause: boost-arm TTFT tail collapse at seed 77188 (fair sweep, 2026-10-20)

## Symptom

At seed 77188 (c16, pin=0, 30-min arrival window), all three boost arms
collapse on aiperf TTFT p99 while fcfs/lpm stay flat:

| arm          | TTFT p99 | admitted-side wait p99 | starvers (wait > 1000 s) |
|--------------|----------|------------------------|--------------------------|
| fcfs         | 461 s    | 450 s                  | 0                        |
| lpm          | 716 s    | 785 s                  | 0                        |
| hrrn         | 893 s    | -                      | -                        |
| dfs-weight   | 1123 s   | -                      | -                        |
| boost g5.0   | 1530 s   | 1472 s                 | 5                        |
| boost g15.0  | 1750 s   | 1646 s                 | 5                        |
| boost ada-j60| 1511 s   | 1512 s                 | 2                        |

The tail is dominated by a handful of starved requests, and the SAME
requests (same seed = same mix) starve only under boost:

- tiny `in=514, match=0` request: <= 453 s (fcfs), 693 s (lpm),
  **1953 s (g5.0)**, 1616 s (g15.0). Admitted t+2964 s, deep in grace,
  after ~70 later arrivals passed it.
- giant `in=122858`: 412 s (fcfs) vs **1472 s** (g5.0); sibling giants
  `in=148967/134881` wait 1790/1450 s under g5.0.

## Mechanism (all three links code-confirmed)

1. **Demotion to group 1.** `_compute_prefix_matches`
   (schedule_policy.py:799) classifies every waiting request each sort
   pass: a request whose cached match is <= 32 tokens
   (IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD) that shares >= 32 tokens of
   prefix (DEPRIORITIZE_THRESHOLD) with an already-inserted queued
   request is added to `temporary_deprioritized`. All cache-aware sorts
   (lpm, hrrn, boost, spf) put this group AFTER every group-0 request
   (schedule_policy.py:880) - i.e. behind every future arrival. Under
   the agentx multi-turn mix, later turns of a session share the whole
   prior history with an earlier queued turn, so later turns (tiny
   continuations AND sequential giants) are demoted; fcfs never runs
   the classification (agnostic path clears the flags).

2. **No time bound.** `_pin_hold_expired` (schedule_policy.py:858)
   returns False when `in_batch_prefix_caching_max_pin_s <= 0`, and the
   campaign runs pin=0, so a demoted follower is never released. The
   60 s default is exactly the anti-starvation valve this campaign
   disabled (chosen to keep arms comparable; it made every cache-aware
   arm's tail measure the starvation mechanism instead of the ranking).

3. **The admission scan never reaches group 1.** `get_new_batch_prefill`
   (scheduler.py:4027-4110) scans the sorted queue front-to-back, admits
   while the budget fits, and BREAKS at the first misfit. The queue was
   giant-logged the whole window (11-18 queued, 1.0-1.8 M pending
   tokens for ~36 min, pending never shrinking), so group 0 kept
   budget-consuming requests ahead and the scan broke long before the
   group-1 tail. A demoted request is only reached when group 0 fully
   drains - end of window + backlog, hence waits ~1500-1950 s.

Under boost, group-0 order is ~arrival order (b <= 0.8 s at g5.0,
rate ~ 66 k tok/s), so the demoting leaders (first-turn giants) sit at
the front chunk-draining for the whole window while their later turns
stay unreachable behind every new arrival. Under lpm the sort is
longest-match-first, which puts the leaders at the very front: the
leader is admitted, the dedup pair dissolves, and followers re-enter
group 0 within minutes (4 recorded followers, worst 785 s). fcfs has
no demotion at all.

Adaptive gamma is orthogonal to all three links, which is why ada-j60
collapses like the static arms (2 starvers vs 5 only because its live
gamma ~ 0.10-0.25 helps the non-demoted smalls). No gamma setting can
fix this tail.

Note: `in_batch_dedup_follower_first_admission` only records the
verdict at the admission pass. T was admitted after its last demoter
had drained, so the flag reads False - the eval records hide the 1950 s
of demotion. The flag-True followers under lpm (62/164/687/785 s) are
the direct evidence the mechanism fires in this workload.

## Implications

- The g5.0-vs-fcfs TTFT p99 gap at this seed is NOT the boost ranking;
  the ranking behaved exactly as designed (verified: all records
  gamma=5.0, bin=256, wall clock; phi-order would put `in=514` ~14th).
  The gap is the in-batch dedup demotion interacting with a
  giant-logged overloaded queue and pin=0.
- Old single-seed verdicts ("boost g5.0 best") came from regimes where
  leaders drained fast; under sustained overload the demotion
  assumption (leader admitted promptly) breaks. Regime-dependent, not
  a property of gamma.
- Cheap validation cell: rerun boost g5.0 seed 77188 with
  IN_BATCH_PREFIX_CACHING_MAX_PIN_S=60 (the default). Prediction: the
  1500-1950 s starvers disappear, TTFT p99 drops toward the lpm/hrrn
  band while keeping the higher cache hit (28.7% vs fcfs 16.1%).
- Policy-code fix candidate for the H200 deployment: bound the
  demotion by sorting followers just behind their leader (phi_leader -
  epsilon) instead of a global group-1 tail, preserving the dedup hit
  without starvation behind future arrivals.

## Fix (branch boost-audit-fixes, 2026-10-20)

Implemented the fix candidate: `_compute_prefix_matches` returns
(followers, follower -> leader) -- simulated-tree insert values are now
per-insert sequence numbers, so the deepest matched node identifies the
leader -- and `_sort_by_boost` places a follower at phi(leader) + eps,
group 0: behind its leader (dedup win preserved), ahead of every later
arrival (starvation gone; worst case a follower waits for one leader).
lpm/hrrn/shortest-prefill keep the frozen group-1 behavior so the
campaign arms stay comparable.

Validation: scripts/boost_jgrid.sh runs boost-g5.0 and ada J in
{60, 15, 5} at seed 77188, c16, pin=0, all with the fix -- the g5.0 and
ada-j60 cells pair 1:1 against the no-fix campaign cells at the same
seed. Prediction: the 1500-1950 s starvers disappear (TTFT p99 drops
toward the fcfs/lpm band) while the cache hit stays near 28.7%.

### Validation result (J-grid, 2026-10-20 evening, commit 8d3823e95f)

| arm            | TTFT p99 | TTFT mean | starvers >1k s | hit % | tok/s | Jain  |
|----------------|----------|-----------|----------------|-------|-------|-------|
| fcfs (ref)     | 461 s    | 310 s     | 0              | 16.1  | 64.6  | -     |
| lpm (ref)      | 716 s    | 107 s     | 0              | 74.5  | 75.7  | -     |
| g5.0 no-fix    | 1530 s   | 306 s     | 5              | 28.7  | 34.8  | 0.500 |
| g5.0 fix       | 1699 s   | 261 s     | 3              | 31.2  | 36.3  | 0.437 |
| ada-j60 no-fix | 1511 s   | 249 s     | 2              | 41.4  | 41.2  | 0.452 |
| ada-j60 fix    | 1186 s   | 227 s     | 4              | 35.9  | 40.9  | 0.412 |
| ada-j15 fix    | 964 s    | 259 s     | 1              | 38.8  | 38.5  | 0.537 |
| ada-j5 fix     | 1500 s   | 317 s     | 7              | 31.6  | 37.3  | 0.436 |

- The follower class left the tail. The no-fix starvers were the
  follower class: the tiny continuation (in=514, match=0 after its
  demoter drained) and match-rich later-turn giants (in=134881
  match=123518, in=111697 match=43478). With the fix, the g5.0 tail is
  match=0 fresh giants only (in=101426/41062/52049); ada-j60 keeps two
  match=23076 siblings that queue behind their own session's giant
  leader (bounded displacement: a follower waits for its leader, no
  longer for the whole group-0 prefix). rids are not stable across
  runs, so the pairing is by class, not by rid.
- The tail that remains is still demotion, through a different door: the
  fix bounds a follower behind one leader per pass, but the next leader is
  the min-rid family member, not the earliest arrival (see the rid tie-break
  section below). Class-level, boost-fix beats fcfs at p50/p90 in every size
  class (giants >= 100k: p50 115-136 s vs 190 s, p90 278-310 s vs 370 s);
  the tail is 1-3 rid-unlucky outliers per cell, not a fresh-giant class
  effect.
- J under overload: j15 is the best boost arm (p99 964 s, 1 starver,
  best Jain 0.537). Live floors matched the computed ones (j60
  0.116-0.122, j15 0.27-0.40, j5 0.62-1.05) and pinned=55-100% in the
  trajectory logs confirms J, not the tail fit, governed throughout --
  finding 2 of the paper audit, now measured.

### Reading the numbers (closed-loop load, distribution view)

aiperf holds 16 requests in flight and issues a new one per freed slot,
so per-arm request counts are goodput, not a fixed arrival set
(client-side request_count: fcfs 84, g5.0-fix 96, ada-j60-fix 110,
ada-j15-fix 113, lpm 180). Every arm completed all the requests it took
(0 null-ttft records at this seed); fcfs's 461 s p99 is over the ~99 it
served while lpm served ~194 with p99 716 s -- under saturation the
arms differ in what they sacrifice (fcfs: goodput, hit 16%; lpm: tail
spread, Jain 0.25), not only in tail latency.

The fix improved or held every body percentile of both paired arms
(scheduler-side, warmup-excluded): g5.0 mean 290->236 s, p50 202->166,
p90 515->406, p95 897->616, >600 s count 9->6; ada-j60 mean 229->209,
p90 582->310, >300 s count 26->13, >600 s 12->7. The apparent g5.0 p99
regression in the aiperf export (1530->1699 s) does not survive the
scheduler-side recomputation (1807->1671 s) -- at n~100 the p99 is a
worst-2 statistic. Costs of the fix: ada-j60 p10 19->35 s (fastest
small requests wait behind freed followers) and TTFT Jain -0.03
(0.434->0.407 g5.0, 0.427->0.390 ada-j60); mechanism not pinned down.

### Root cause of the remaining tail: rid tie-break head rotation (2026-10-20 night)

The fix's "worst case a follower waits for one leader" holds per pass but
not across head rotations. Elimination ladder on the g5.0-fix cell (seed
77188), every step measured on existing data:

- Retraction: the three worst outliers (waits 1870/1680/1483 s,
  in=101426/41062/52049, all match=0, follower=False) record
  num_retractions=0; queue_wait stamps survive retraction by definition
  (first admission minus first entry, stamped once).
- Chunk-limit: engaged in only 13% of prefill steps while the queue was
  non-empty (781 steps, new-token p50 = 8192, 103 capped) -- minor at most.
- Budget misfit / scan break: 77 requests arrived after the worst outlier
  and were admitted before it, 55 of them with uncached > 8192 (a fresh
  122366-token request waited 131 s) -- large fresh requests fit
  throughout.
- Sort displacement at gamma=5: b <= 0.64 s per jump -> ~50 s cumulative
  over ~75 later arrivals vs ~1400 s measured excess.
- Stuck earlier leader: every pre-873 s arrival drained by 963 s while the
  outlier waited to 2743 s -- no earlier leader to inherit the wait from.

Mechanism (code path: scheduler.py calc_priority at the top of every
get_new_batch_prefill pass -> schedule_policy.py):

1. calc_priority sorts waiting_queue in place each pass; followers of a
   shared-prefix family all tie at phi(leader)+eps, broken by rid
   (_sort_by_boost key = (group, phi, rid)).
2. Next pass, _compute_prefix_matches iterates that order: the first free
   family member -- once the head drains, the min-rid follower -- inserts
   and becomes the new leader; the family now sorts at phi(new leader) =
   the ARRIVAL of whichever member holds the smallest rid.
3. Under continuous same-prefix arrivals an early member with an unlucky
   rid never wins headship: every later arrival with a smaller rid passes
   it, for as long as arrivals continue. pin=0 (campaign) disables the
   only other escape (_pin_hold_expired).

Record-level fingerprint (no other mechanism produces it): the final drain
of the g5.0-fix cell admits the last five requests in exact rid-lexicographic
order while ignoring arrival -- 97fd(arr 2406) 2711 s, bdd1(arr 2417)
2730 s, ea93(arr 873) 2743 s, f8ee(arr 1287) 2770 s, fd3b(arr 1099)
2779 s. The stuck trio are all e/f-prefixed rids (bottom of the rid space)
and drain only after arrivals stop (window end 2475 s + backlog).

This supersedes the j5-open-question reading in boost_paper_fidelity
finding 2: the mechanism is gamma-independent (it lives in the dedup
tie-break layer, not the ranking), so j5's 7 starvers and ada-j15's single
starver (a tiny in=513 that waited 1510 s) are the same signature, not
chunk-limit effects.

Pre-registered validation, written before any code change: tie-break
followers by arrival stamp (first_queue_entry_time wall /
boost_arrival_tokens tokens) instead of rid in _sort_by_boost.
Predictions for one g5.0 seed-77188 cell:

- starvers (>1000 s) -> 0; TTFT p99 -> the fcfs/lpm band (<= ~700 s); the
  ex-trio waits -> the 43-160 s cohort band (the pre-873 s family drained
  by 963 s).
- body percentiles (mean/p50/p90/p95) unchanged within noise; cache hit
  stays ~31% (one head at a time is preserved).
- any residual late tie admits in ARRIVAL order, not rid order.

Kill criteria: starvers persist -> mechanism incomplete, next suspect is
the chunk-limit interaction (one g5.0 cell with the limit disabled);
starvers gone but a body percentile regresses >10% -> trade-off rejected,
redesign before adopting.

### Validation result (arrival tie-break, 2026-10-21 04:17, commit 413514ef3d)

Cell c16_g5.0_wall_k256_20261021_032242 (run_meta sglang_commit
413514ef3d; warm-cache parity restored after the ENOSPC night: fresh
FP8 + workload-dataset download, 0 dangling links; log
logs/boost_arrival_tiebreak_g5.0_77188_retry.log). Scheduler-side
warmup-excluded, n=104, vs the g5.0-fix pair 20261020_181256 (n=110),
same recipe both cells.

| prediction                           | verdict  | numbers |
|--------------------------------------|----------|---------|
| starvers (>1000 s) -> 0              | PASS     | 0 starvers, 0 >600 s, max wait 457 s (old: 3 at 1483-1870 s) |
| TTFT p99 -> <= ~700 s                | PASS     | aiperf 1699 -> 474 s (scheduler p99 469, max 501); fcfs 461 / lpm 716 |
| ex-trio waits -> 43-160 s            | PARTIAL  | same-class waits 160/284/298 s (old 1870/1680/1483); nothing >600 s anywhere, but 2 of 3 sit above the band -- they arrive later (1260/1287 s) into a deeper queue |
| body (mean/p50/p90/p95) unchanged    | FAIL     | mean 214->244 (+14%), p50 144->297 (+106%), p90 379->407 (+7%), p95 590->422 (-28%); same-arrival-band medians 114->189 / 218->368 / 306->386 s -- real, not mix |
| cache hit stays ~31%                 | FAIL     | 31.2 -> 22.3% scheduler-side (aiperf first-attempt 37.3 -> 26.9%); not a top-2 artifact (excl top-2: 26.6 -> 17.7%) |
| residual ties admit in arrival order | PASS     | last 8 admits strictly arrival-monotone; the rid-order drain fingerprint is gone |

Kill criterion fired (p50 +106%, mean +14%): per pre-registration the
trade-off is rejected as-is -- redesign before adopting. What the cell
does settle: the rid tie-break mechanism is dead (starvers, p99 and the
drain fingerprint all cleared, Jain 0.437 -> 0.843), and part of the old
cell's speed was the bug itself -- the stuck families' prefill work was
deferred past the window edge, and their end-drain bursts (9 siblings x
22,622 matched tokens admitted 2711-2779 s) inflated the old hit rate.

Open, and what blocks adoption: the in-window hit gap (31.6% old vs
22.1% new, drain admits excluded) is NOT explained by the drain bursts
alone; old cells show same-match fan-out clusters of 8-9 members, new
cells 3-5. Near the saturation knee a 9-pp hit drop (-6% total
throughput, 3565 -> 3360 tok/s) roughly doubles the median wait. n=1
per arm, so before redesigning: (a) one repeat cell of this exact
config to bound run variance, and/or (b) trace one fan-out family
through both cells' server_metrics_export.jsonl to pin the
cluster-shrink mechanism. Campaign view on the clean base: boost g5.0
p99 474 s ~ fcfs 461, < lpm 716 << dfs-weight 1123, 0 starvation, best
Jain -- the e2 tail goal is met; the open question is the body cost.

### H200 phase-1 contamination scan (2026-10-21, server HEAD a9d2b9ecd8)

Question: the H200 phase-1 gamma sweep ran on a base WITHOUT the arrival
tie-break fix (a9d2b9ecd8 has the leader-relative fix but not 413514ef3d)
-- does the rid bug contaminate those cells? Verdict: the bug is INERT in
that regime; the gamma-flat conclusion is usable, not reference-only. Scan
of all 6 cells' schedule_eval.jsonl (warmup-excluded, same recipe as above;
server path results/glm53/agentx_dynamo/policies_v2/phase1_gamma/boost/):

| cell            | n    | >1k s | >600 s | max qw | qw p99 | hit % |
|-----------------|------|-------|--------|--------|--------|-------|
| g1.0            | 1212 | 0     | 0      | 305 s  | 81 s   | 87.7  |
| g3.0            | 1155 | 0     | 0      | 190 s  | 120 s  | 87.2  |
| g5.0            | 1211 | 0     | 0      | 315 s  | 85 s   | 87.7  |
| g7.0            | 1239 | 0     | 0      | 467 s  | 101 s  | 84.3  |
| g10.0           | 1176 | 0     | 0      | 208 s  | 80 s   | 87.9  |
| g15.0 (crashed) | 420  | 0     | 0      | 131 s  | 53 s   | 77.5  |

Scheduler-side max waits (305/190/315/467/208) match the aiperf warm-max
column exactly; p99 ordering matches (g3.0 worst, g1.0/g10.0 best). The
g15.0 row is the survivor subset (420 of ~1210 served before the crash) --
invalid as a policy cell, scanned only for completeness.

Why inert, three independent reasons:

1. pin=60 on the H200 cells (run_meta in_batch_prefix_caching_max_pin_s
   =60.0), not the local campaign's pin=0: the 60 s valve bounds any
   head-rotation wait even if the mechanism engaged.
2. The regime never builds the giant-logged queue the mechanism needs: qw
   p50=1 s, p90=8-13 s, hit 87-88% -- 8xH200 drains the queue in seconds
   (the local RTX c16 cell held 1.0-1.8 M pending tokens for 36 min). The
   workload's fan-out families exist (same-match clusters of 6-9 members,
   matched 28k-108k) but arrive against a warm cache -- the leader drained
   long before -- so the in-batch dedup classification never binds: 0
   follower records in every cell (local cells record them routinely).
3. Fingerprints absent: 0 starvers, 0 waits >600 s, and the final drain is
   neither rid-lexicographic (the bug) nor otherwise anomalous.

Cross-rig consistency: gamma-flat on H200 matches the local validation --
the tail lives in the dedup layer, not the ranking; gamma is not the tail
knob on either rig. Known config delta for any cross-rig read: local
campaign pin=0, H200 pin=60. The g15.0 crash is a separate defect (hard
pool exhaustion -> OOM backout spin -> CUDA IMA; g15_oom_crash_root_cause.md).

### Variance-bound repeat cell (pre-registered before launch, 2026-10-21)

P4 (qw p50 144 -> 297 s) and P5 (hit 31.2 -> 22.3%) rest on n=1 per arm.
This cell repeats the 032242 config exactly (seed 77188, gamma 5.0, bin 256,
wall clock, pin=0, c16, commit 413514ef3d) to bound run-to-run noise before
any redesign decision. Comparison recipe identical (drop HEALTH, drop the
first 60 s of arrivals).

Predictions, written before launch:
- R1 starvers stay 0, max wait < 600 s.
- R2 qw p50 in [223, 371] s (297 +/- 25%), p90 in [305, 509] s (407 +/- 25%).
- R3 scheduler-side hit in [17, 28]% (22.3 +/- 5pp).
- R4 final drain admits arrival-monotone.

Decision rule, pre-registered:
- Inside R2/R3 and outside the old bands (p50 ~144, hit ~31): the body/hit
  costs are real on n=2 -> the redesign gate opens (split family admission
  per pass).
- Back in the old bands (p50 <= 190 s or hit >= 29%): the 032242 regressions
  were run-to-run noise -> P4/P5 reopen, no redesign.
- Otherwise mixed: one more seed before deciding.

Parallel analysis while the cell runs (no new run needed): trace one
fan-out family through both cells (181256 vs 032242) to pin the
cluster-shrink mechanism -- hypothesis: with arrival tie-break the family
admits spread out in arrival order, and members admitted while the head is
still chunk-prefilling match only the committed prefix (partial, differing
values), while the old rid-order drain admitted the whole family at once
after the head finished (uniform full match).

### Fan-out family trace (2026-10-21, passes 1-4, before the repeat cell completed)

schedule_eval per-request pairing (same seed, same inputs) plus the
server_metrics series. Two results: a correction to the P5 reading, and
the cluster-shrink mechanism.

Correction: the "9 siblings x 22,622 admitted 2711-2779 s" drain-burst
claim above is wrong for this cell. The >2500 s drain admits are all
match=0 (the stuck trio plus two fresh); the 22,622 cluster is 8 members
admitted IN-WINDOW at 1575-2514 s with full matches and 113-371 s waits.
The old cell's hit advantage is real in-window dedup efficiency, not a
drain artifact. (The other half of "old speed was the bug" still holds:
the trio's ~195k of fresh prefill was deferred past the window edge.)

Mechanism, evidence per link:

1. Trunk timing seeds everything. Session trunk in=30569 admits 923 (OLD)
   vs 1037 (NEW) -- 114 s apart with the pool at usage 0.97-0.99 (first
   giant phase). The trunk is a head (match=0, no dedup): its sort position
   is identical in both cells, so the gap is admission-budget jitter at
   the knee, not the tie-break. Its early followers then straddle the
   trunk's prefill: the in=30186 twins admit 964/964 (OLD -- trunk
   committed, full match 27373 x2) vs 1041/1044 (NEW -- trunk mid-flight,
   partial 23431 and 27685 three seconds apart: progressive page commits).
2. Partial matches cost real prefill: each early follower pays its
   unmatched 4-17k tokens fresh instead of hitting cache.
3. The extra uncached work raises KV pressure and the radix tree evicts
   idle session prefixes: mid-session continuations that matched 23-25k in
   OLD match 0 in NEW (in=27268 [OLD 23622], 25773 [24576], 28611 [24673],
   30476 [24576 -- its own session matched 24-28k at 2300-2400, zero by
   2629: the prefix was evicted in the ~200 s between turns), and family-B
   branches lose their 22k matches wholesale (in=93625, 103742, 69877).
   Retraction is ruled out (OLD 3, NEW 1). Both cells run the same 545k
   pool at the knee (usage 0.8-0.99, evictable often <50k; NEW touches
   evictable=0 at t=901/1041/2222).
4. Totals: NEW prefilled ~7.49M uncached tokens vs OLD ~6.98M (+7%) while
   serving 0.5M LESS total input -- at the knee that is the -6% throughput,
   the ~2x median wait (P4) and the -9pp hit (P5).

Reading: the tie-break fix does not CAUSE the body cost -- it removes an
accidental admission timing that happened to land on the lucky side of a
noise-seeded cascade (n=1). The old cell's real edge was followers that
(by that run's budget jitter) admitted just after their trunks committed,
giving full matches. The systematic fix makes that timing deterministic:
gate a follower's admission on its leader's KV commit (hold set released
by the commit event) instead of placing it behind the leader's first
chunk -- match completeness stops depending on budget jitter, the dedup
win is recovered by construction, and the displacement bound is one
leader prefill duration (seconds), with the pin valve bounding the
hold against later arrivals. This supersedes the "smaller family batches
per pass" sketch: batch the family at the leader's commit boundary.
Caveat: eviction severity is rig-config-dependent -- the H200 production
config has hicache 128G write-back, which would soften link 3 (evicted
prefixes reload from host instead of missing).

R5 (pre-registered before the repeat cell completes): count same-match
clusters (m>=16k) and mid-session zero-matches (continuations whose
session has earlier served turns). Repeat cell materially different from
032242 (top cluster >=7 or mid-session zeros <=1) -> the cascade seed is
run-to-run noise; family batching is a robustness fix. Same pattern
reproduces (top <=5, several mid-session zeros) -> the tie-break
systematically biases admission toward the cascade; the fix is required
regardless.
