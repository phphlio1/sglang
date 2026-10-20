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
