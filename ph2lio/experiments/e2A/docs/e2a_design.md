# e2A: boost-cache — fusing LPM cache reuse into boost's bounded anti-starvation

Status: policy implemented (this doc + the code it describes), RTX sweep
pending. Owner: e2A (this clone). Baselines live in the archived e2 campaign
(`../../archives/`, verdicts in `fixed_e2_rtx6000.md`).

## The gap e2 left

Boost (gamma-boosted arrival, `phi = A - b_gamma(q(u))`) wins the starvation
metrics at overload — Jain, worst-5%, TTFT/ITL p95/p99 — but loses the
throughput family (tok/s, TTFT/ITL p50/mean) to dfs-weight and lpm because
it does not exploit the radix cache the way lpm does. lpm sorts purely by
`-matched_prefix_tokens`: it wins cache reuse and throughput, and starves
unboundedly (no arrival anchor at all).

Two mechanisms behind boost's cache loss:

1. Inside the work bin: every request with `u <= k` ties at the max boost
   `b_gamma(k)` and the tie falls to FCFS — a hot-prefix request that
   arrived later loses to an equal-bin cold one, and the hot pages can be
   evicted before reuse.
2. Cold giants admitted by arrival evict the hot tree region; lpm keeps
   serving the hot region instead.

## The fusion

Everything boost needs for anti-starvation is one inequality: a newcomer
can only cut ahead by its boost, which is bounded by `b_gamma(k)`. The
generalization this policy rests on:

> Any additive term in phi that is (a) bounded above by a constant and
> (b) not growing with request age preserves bounded displacement — the
> bound just becomes `b_gamma(k) + C`.

lpm starves because its cache term dominates arrival unboundedly. So the
fusion adds lpm's signal as a *capped* bonus:

```
phi'(r) = A(r) - b_gamma(q(u(r))) - lambda * min(matched(r), M)
```

- `lambda` in [0, 1] (`--boost-cache-lambda`, default 0.5): how much seniority
  one matched token buys. `lambda = 0` is bit-identical to boost — the
  baseline stays comparable in the same binary.
- `M` (`--boost-cache-cap-tokens`, default 8192): cap on matched tokens that
  earn the bonus; beyond it, nothing. The newcomer jump bound becomes
  `b_gamma(k) + lambda*M` clock-tokens (wall mode: `lambda*M/rate` seconds).
- Units follow `--boost-clock` exactly like the work term: tokens mode adds
  clock-tokens, wall mode converts at the same measured rate (unmeasured
  rate -> bonus 0, pure arrival order).
- gamma-Ada carries over unchanged except its J budget now covers the sum:
  the gamma floor tightens so `b_gamma(bin/rate) + lambda*M/rate <= J`.

Matched tokens are already excluded from `u(r)` — the work term prices the
prefill that must happen; the bonus prices the reuse itself (the service
seconds a hit saves), which is what pulls hot-prefix requests ahead of
equal-bin cold ones and recovers lpm's locality.

## Where the code is

- `python/sglang/srt/managers/schedule_policy.py` — `CacheAwarePolicy
  .BOOST_CACHE`, the `cache_bonus` term in `_sort_by_boost`, the shared
  jump budget in `AdaptiveBoostGamma._gamma_floor`.
- `python/sglang/srt/arg_groups/fields/schedule.py` +
  `validation_hook.py` — the two knobs, validated (lambda in [0,1], cap >= 1).
- `python/sglang/srt/observability/schedule_eval_log.py` — per-record
  `cache_lambda` / `cache_cap_tokens` (arm identity in the eval log).
- Unit tests: `test/registered/unit/managers/test_schedule_policy.py`
  (`TestBoostCacheSort` + the floor-sharing case) — the bound test pins the
  headline invariant: a fully-cached newcomer beats an old cold request only
  within `b_gamma(k) - b_gamma(q(u_cold)) + lambda*M`, no further.

## Experiment plan (RTX 6000 Pro, this clone)

Arms per seed (seeds 77188 overload + 42302 light, same as the e2 campaign;
c16 the standard RTX point, `FORCE_RESTART=1` fresh boot per cell):

1. boost g0.25 wall k256 (lambda=0 reference — identical code path)
2. boost-cache g0.25 wall k256 lambda in {0.25, 0.5, 1.0}, M=8192
3. boost-cache g0.25 wall k256 lambda=0.5, M in {2048, 32768} (cap sensitivity)
4. boost-cache ada J50 (shared-budget floor) lambda=0.5 M=8192
5. lpm + dfs-weight (reference lines for the two target families)

Metrics: the e2 metrics_v2 pipeline (unchanged, cloned with this frame):
starvation family (Jain, worst-5%, TTFT p95/p99) must stay at boost's level;
throughput family (tok/s, TTFT/ITL p50/mean) must close toward lpm/dfs.

## Success criteria

- Keep: Jain / worst-5% / TTFT p95/p99 within noise of boost g0.25 at seed
  77188 (the anti-starvation the user named as the must-keep).
- Gain: TTFT p50 and tok/s strictly better than boost, a measurable share of
  the boost->lpm gap closed (the user named the cache-utilization loss as
  the thing to fix).
- The lambda sweep should show a monotone trade: lambda=0 = boost, larger
  lambda moves the throughput family toward lpm while the tail family
  degrades gracefully (bounded by lambda*M), not cliff-like.
