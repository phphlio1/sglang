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
