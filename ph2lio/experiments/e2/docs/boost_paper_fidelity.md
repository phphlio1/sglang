# Boost policy: paper-fidelity audit and resolutions (2026-10-20)

Fresh full re-audit of the boost implementation (schedule_policy.py at
0229372590..bec18cc62a) against the paper ("Beyond Prediction: Tail-Aware
Scheduling for LLM Inference" = the UniBoost/TRAIL paper, Li et al.,
ICML 2026; arxiv-linked from yl3469.github.io/uniboost-icml26). Scope: hunt
implementation bugs that make adaptive boost worse. Result: no wiring or
math bug in the controller; five findings, all resolved below. Fixes land
on branch `boost-audit-fixes`.

## Verified exact matches (no change)

- `gamma_boost` == paper b_gamma(w) = -ln(1 - e^{-gamma w})/gamma
  (numerically verified across the domain, incl. the underflow tail).
- `geometric_bin` == paper q(u) MemGuard quantization
  k * 2^floor(log2(max(u, k)/k)).
- Jump-floor round trip: b_{gamma_floor}(bin_size / rate) == max_jump_s
  exactly (bisection root of g(u) = J/tau); locked by
  `test_jump_floor_bounds_max_boost` / `test_jump_floor_tracks_rate_not_gamma`.
- Controller wiring: scheduler `record_completed_latencies` feeds E2E
  latencies (completion - scheduler recv) + works; EMA(0.2) -> clip ->
  `schedule_eval_logger.boost_gamma` mirror; verified end to end.

## Finding 1 -- the paper is internally inconsistent on Eq. 9 (39x)

Main text Eq. 9: gamma_hat = ln(Fbar(t95)/Fbar(t99))/Delta = ln 5 / Delta
(CCDF ratio; theoretically coherent -- the log-linear tail slope).
Algorithm 1 UPDATEGAMMA line 17: gamma_hat = (log 99 - log 95)/Delta
= ln(99/95)/Delta ~= 0.0412/Delta -- a CDF mixup, 39x smaller.

Resolution: keep the main-text form (already implemented); anchor the
discrepancy in a comment at `AdaptiveBoostGamma.LN5`
(schedule_policy.py). No code change.

## Finding 2 -- under overload, J (not the tail fit) sets gamma

Live ada-j60 @ seed 77188: survivor E2E (n=122) t95 = 721.2 s,
t99 = 1428.0 s -> Delta = 706.8 s -> gamma_hat = 0.0023, vs floor
0.1154 (k=256, rate ~30k tok/s, J=60). The controller sat at the floor
the whole run (logged gamma p50 ~ 0.120). Consequence: "adaptive" under
sustained overload degenerates to a static gamma = J-derived floor with
J-second jumps; the fit never moves it. This is the contract (fat tail ->
more aggressive), not a bug -- but it makes J the real tuning knob, and it
was invisible in the logs.

Resolution:
- Trajectory log now prints `pinned=%` (share of updates whose raw tail
  fit is at/below the floor; the EMA target is the floored estimate, so
  gamma itself only approaches the floor asymptotically -- the raw fit is
  the pinned signal). A floored run now reads `fit << floor, pinned=100%`.
- J-grid experiment (scripts/boost_jgrid.sh): J in {60, 15, 5} at seed
  77188 under the leader-relative fix, to measure the J gradient that the
  floor-pinned regime actually exposes. Result (evening run): TTFT p99
  1186 / 964 / 1500 s for J = 60 / 15 / 5 -- J=15 best (also best Jain
  0.537, 1 starver); live floors matched the computed ones (0.116-0.122 /
  0.27-0.40 / 0.62-1.05) with pinned=55-100%. Full table in
  boost_tail_collapse_root_cause.md. The ranking is non-monotonic in J;
  the j5 tail (7 starvers despite a near-FCFS sort) is the rid tie-break
  head rotation, not a J or chunk-limit effect -- gamma-independent,
  root-caused and pre-registered for validation in
  boost_tail_collapse_root_cause.md (rid tie-break section).
- Unit tests already lock the floor round trip and the Eq. 9 ln5/Delta
  form (`test_estimator_math_and_ema`).

## Finding 3 -- dedup demotion starvation (the boost-arm tail collapse)

Root cause of the 3.3x TTFT-p99 collapse of all boost arms at seed 77188
(full write-up: boost_tail_collapse_root_cause.md): in-batch dedup
followers were demoted to a global group-1 tail behind every group-0
request -- under a giant-logged queue that is behind every future
arrival, so later session turns starved 1500-1950 s.

Resolution (branch boost-audit-fixes): `_compute_prefix_matches` now
returns (followers, follower->leader) -- simulated-tree node values are
per-insert sequence numbers, so the deepest matched node identifies the
leader -- and `_sort_by_boost` places a follower at phi(leader) + eps
(group 0): still behind its leader (the dedup win), never behind future
arrivals (the starvation gone). lpm/hrrn/shortest-prefill keep the frozen
group-1 behavior for campaign-arm comparability. Bounded displacement
now holds for followers too: worst case they wait for one leader.

## Finding 4 -- size-aware gamma split: deliberate deviation, kept

The paper fits the overall completed-latency tail. We split at median
work and substitute the short-work fit when the long-work tail is fatter.
Reason: under overload the overall tail is inflated by giants starved by
the policy itself (positive feedback: fatter tail -> smaller gamma ->
longer giant waits -> fatter tail); the paper's fit targets exogenous
service variance, not policy-induced starvation. Documented in the
`_size_aware_gamma_hat` docstring; kept as-is.

## Finding 5 -- survivorship bias in the Eq. 9 feed: direction analysis

The fit sees only completed requests; starved requests never complete,
so Delta is underestimated and gamma_hat is overestimated (more
aggressive than the truth). Naive censored-sample fixes backfire under
overload: imputing starved requests' waits at current elapsed inflates
Delta enormously -> gamma_hat -> 0 -> clamps to the same floor
(floor-pinned, unchanged behavior); excluding them biases the other way.
The real remediation is un-starving the tail (finding 3's fix), after
which the completed feed is representative. No estimator change.

## Finding 6 -- deliberate fidelity gaps, documented

- No cross-phase preemption: the paper's Eq. 7 SELECTVICTIM/SWAPOUT
  (evicting a running prefill for a short newcomer) is not implemented;
  boost here is waiting-queue ordering only. Chunked-prefill preemption
  is a much larger scheduler change -- out of scope for this port.
- `shortest_prefill_chunk_limit` (BOOST + SPF): sglang-only addition
  capping the active prefill chunk to reserve budget for shorter waiting
  requests; complements boost under chunked prefill. Kept.
- w~ = input + output - matched (sum) vs paper max(w_i, s_i - h_i):
  identical except retracted requests, where the sum correctly charges
  the re-prefill cost of sglang's retraction-restart semantics. Kept.
