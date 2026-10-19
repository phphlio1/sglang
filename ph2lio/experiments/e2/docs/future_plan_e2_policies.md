# Future Plan: E2 Policies on 8xH200

**Date**: 2026-10-19
**Current**: 1x RTX 6000 Pro Blackwell 96GB, Qwen/Qwen3.8-27B-FP8
**Target**: 8x H200, GLM-5.3 (production), TP8/DPA8/EP8
**Branch**: `uniboost-boost-policy`

## 1. Implementation status

### Static boost — complete, paper-correct

| Component | Paper ref | Code location | Status |
|-----------|-----------|---------------|--------|
| b_gamma(w) = -ln(1-e^{-gw})/g | Eq. 2 | `gamma_boost()` schedule_policy.py:232 | Correct, uses `expm1` for stability |
| q(u) = k*2^floor(log2(max(u,k)/k)) | Eq. 8 | `geometric_bin()` schedule_policy.py:253 | Correct |
| Phi(r) = A(r) - b_gamma(q(u(r))) | Sec. 2 | `_sort_by_boost()` schedule_policy.py:956 | Correct, sort ascending Phi |
| u(r) = uncached work | — | `_shortest_prefill_work()` schedule_policy.py:856 | `max(1, input+output-matched)` |
| Anti-starvation bound | Thm | b_gamma(bin_size) caps jump | Correct |
| g->0 = SJF, g->inf = FCFS | — | Verified by 20-cell sweep | Correct |

Edge cases all handled: zero work (`max(1,...)`), negative (clamped), overflow (Python big ints), division by zero (guarded), `rate=None` (pure arrival order).

Sort only reorders `waiting_queue` in-place. Does not touch `running_batch`, `tree_cache`, `can_run_list`, or any `Req` fields. Pure function.

### Adaptive boost — complete, paper-correct with known limitation

| Component | Paper ref | Code location | Status |
|-----------|-----------|---------------|--------|
| gamma_hat = ln(5)/(t99-t95) | Eq. 9 | `_tail_gamma_hat()` schedule_policy.py:454 | Correct, linear-interpolation percentile |
| EMA update | — | `(1-beta)*gamma + beta*gamma_hat`, beta=0.2 | Correct (implementation choice, not from paper) |
| Max-jump bound | — | `_gamma_floor()` bisection schedule_policy.py:464 | Correct, 64 iterations, bounds b_gamma(k/rate) <= J |
| Size-aware fix | Extension | `_size_aware_gamma_hat()` schedule_policy.py:421 | Sound, partially breaks feedback loop |

Known limitation: under heavy overload, gamma does not converge to sweet spot (oscillates ~0.10 instead of 5.0). Root cause: tail fat is global under overload, not isolated to giant starvation. This is an algorithm limitation, not a code bug. Use static gamma=5.0 for production; adaptive for experimentation.

### OOM recovery — general reliability fix (not boost-specific)

Code in `scheduler.py`, applies to all policies:
- `scheduler.py:4180`: try/except RuntimeError around `prepare_for_extend()`
- `scheduler.py:4183-4185`: free req pool slots on OOM
- `scheduler.py:4187`: return reqs to waiting_queue
- `scheduler.py:4196-4197`: set grace period + increment OOM counter
- `scheduler.py:3946`: chunk downscaling after 3+ consecutive OOMs
- `scheduler.py:5065-5067`: grace period skips invariant checker

Evidence: 559 OOM events -> server survived, sweep completed 11/11 cells. OOM reduced 96% with chunk downscaling.

### Known bugs (logging only, no behavior impact)

1. `schedule_policy.py:416`: Log field `tail t99-t95` is actually `LN5/gamma_hat` (post-floor/clip/substitution), not raw tail width. Misleading when floor or size-aware substitution binds.
2. `schedule_policy.py:417`: Log field `gamma_hat` shows processed value, not raw Eq. 9 estimate. Misleading when diffing runtime vs offline estimator.
3. `schedule_policy.py:404`: `zip(lat_list, work_list)` silently truncates on length mismatch. Safe in production (built in lockstep), but future callers could misalign.

## 2. Multi-GPU readiness

| Setup | Why safe | Risk |
|-------|----------|------|
| TP8 | Single scheduler -> single policy -> one sort decision | None |
| DP8/DPA8 | Multiple schedulers -> each has own policy -> independent | Each rank computes own adaptive gamma (correct) |
| EP8 | Expert dispatch internal, scheduler unchanged | None |
| OOM recovery | Scheduler is single decision-maker -> all workers skip same batch | None |

Policy is initialized per-scheduler in `init_schedule_policy()` (scheduler.py:1395). Each scheduler has its own `SchedulePolicy` instance with its own `boost_gamma`, `adaptive_gamma`, and `prefill_token_rate`.

## 3. Pre-deploy checklist

### 3.1 Code

```bash
# On 8xH200 server:
cd /path/to/sglang
git fetch origin
git checkout uniboost-boost-policy   # or cherry-pick onto main
pip install -e "python[dev]"
pre-commit run --all-files           # ensure lint passes
```

### 3.2 Environment

```bash
# Required env vars:
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export SGLANG_ENABLE_JIT_DEEPGEMM=0    # if Blackwell; H200 may not need this

# CUDA setup (adjust for H200):
export CUDA_HOME=/path/to/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
```

### 3.3 Verification run (30 min)

Before any sweep, run one cell to confirm the policy works on multi-GPU:

```bash
# Start server with boost gamma=5.0
python -m sglang.launch_server \
  --model-path <model> \
  --schedule-policy boost \
  --boost-gamma 5.0 --boost-bin-size 256 --boost-clock wall \
  --in-batch-prefix-caching-max-pin-s 0 \
  <other server flags...>

# Run a short benchmark (30 min, c16)
# Check: server starts, requests complete, no crash, metrics look reasonable
```

If the verification passes, proceed to sweep.

### 3.4 Unit tests

```bash
python3 test/registered/unit/managers/test_schedule_policy.py
# Expected: 59 tests, all OK
```

## 4. Experiment plan for 8xH200

### 4.1 Phase 1: Static boost sweep (verify sweet spot transfers)

The sweet spot gamma=5.0 was found on 1x RTX6000. On 8xH200 with TP8, the prefill throughput is ~8x higher, so the optimal gamma may shift. Re-sweep:

| gamma | Rationale |
|-------|-----------|
| 1.0 | SJF-ward bound |
| 3.0 | Below previous sweet spot |
| 5.0 | Previous sweet spot |
| 7.0 | Above previous sweet spot |
| 10.0 | FCFS-ward |
| 15.0 | Near-FCFS |

Config: c16 or c32 (match production concurrency), 1800s, seed=42, wall clock, k=256, pin=0.

### 4.2 Phase 2: Policy comparison

Compare boost gamma=5.0 against lpm, hrrn, fcfs on the 8xH200 config. The 1x RTX6000 results showed boost gamma=5.0 best on TTFT p99 + balance; verify this holds on multi-GPU.

| Policy | Why |
|--------|-----|
| lpm | Throughput baseline |
| hrrn | E2E + proportional fairness |
| fcfs | Fairness baseline |
| boost g=5.0 | Previous best |
| boost g=15.0 | Fairness alternative |

### 4.3 Phase 3: Adaptive gamma (experimentation)

Run adaptive gamma with size-aware fix on 8xH200. The higher prefill throughput may reduce the overload pressure, potentially allowing the adaptive gamma to converge better.

```bash
--boost-adaptive-gamma --boost-adaptive-max-jump-s 60 --boost-clock wall
```

Check: does gamma converge closer to the sweet spot with 8x throughput? The feedback loop may be weaker when the system is less overloaded.

### 4.4 Phase 4: Production config (GLM-5.3)

Run the production-exact config: GLM-5.3, TP8/DPA8/EP8, max-running-requests=256, chunked-prefill-size=32768. Compare:
- Current production policy (whatever is deployed)
- boost gamma=5.0
- boost gamma=15.0

Metrics to collect (same as e2_policies.md):
- n done / n wait
- cache hit (%)
- queue mean/p99 (s)
- TTFT mean/p99 (ms)
- ITL mean/p99 (ms)
- tok/s
- Jain
- stretch p99
- worst-5% wait (%)

## 5. Key findings from 1x RTX6000 (reference)

### 5.1 Sweet spot: gamma=5.0

Best TTFT p99 (140270 ms), best Jain (0.400), zero retractions, zero OOM. Balanced on all metrics — no population gets thrown under the bus.

### 5.2 Starvation ranking (corrected)

`starvation_rate` (threshold >10s) is misleading. Use stretch p99, Jain, and worst-5% wait instead. All are threshold-free (no cache-hit/miss classification needed).

| Rank | Policy | stretch p99 | Jain | worst-5% | TTFT p99 (ms) | tok/s |
|------|--------|-------------|------|----------|---------------|-------|
| 1 | boost g=5.0 | 117.90 | 0.400 | 29.20 | 140270 | 68.2 |
| 2 | hrrn | 58.14 | 0.304 | 36.65 | 194054 | 75.2 |
| 3 | boost g=15.0 | 60.65 | 0.419 | 28.26 | 150806 | 67.3 |
| 4 | lpm | 127.81 | 0.318 | 36.92 | 183587 | 75.0 |
| 5 | boost ada | 92.96 | 0.332 | 34.57 | 182250 | 67.7 |
| 6 | fcfs | 56.83 | 0.337 | 33.73 | 226862 | 66.4 |
| 7 | dfs-weight | 64.04 | 0.151 | 60.92 | 421762 | 86.9 |

### 5.3 Boost g=5.0 trade-off: throughput vs latency+fairness, not starvation

On all threshold-free metrics, boost g=5.0 is better than or equal to LPM: stretch p99 (117.90 vs 127.81), Jain (0.400 vs 0.318), worst-5% wait (29.20 vs 36.92), TTFT p99 (140270 vs 183587). The trade-off is throughput (68.2 vs 75.0 tok/s, −9%) for better latency and fairness. Stretch p99 is high because mid-size requests with tiny prefill (0.25s) get high wait/prefill ratios — the cost of SJF bias for better TTFT, not starvation.

A cache-based cold/warm split (cached==0 vs cached>0) was considered but dropped: the binary threshold at 0 is crude (1/10000 matched = "warm"), and the metric can diverge from threshold-free metrics.

### 5.4 Policy recommendations

- **boost g=5.0**: interactive/chat, SLA by TTFT p99, Jain fairness priority. Best TTFT p99 + best Jain.
- **hrrn**: mixed workload, SLA by E2E, proportional fairness (best stretch p99 among reordering policies)
- **boost g=15.0**: fairness priority, near-FCFS with better throughput
- **lpm**: batch/throughput, highest n done (181). Not interactive (TTFT p99=184s, worst stretch p99=127.81).
- **boost ada**: dynamic workload, experimentation (does not converge under overload)
- **fcfs**: light load only, best stretch p99 (56.83) but worst TTFT p99 (227s). "Equally slow for everyone."
- **dfs-weight**: batch/throughput only, highest tok/s (86.9). Worst Jain (0.151), worst TTFT p99 (422s). Never use for interactive.

### 5.5 Adaptive gamma convergence

| Version | gamma final | gamma range | tok/s |
|---------|-------------|-------------|-------|
| Original J=60 | 0.083 | [0.08, 0.09] | 63.5 |
| Original J=120 | 0.053 | [0.05, 0.06] | 59.5 |
| Size-aware J=60 | 0.102 | [0.09, 0.55] | 67.7 |

Size-aware fix improves throughput 63.5 -> 67.7 tok/s and breaks feedback loop partially (gamma occasionally reaches 0.55 vs stuck at 0.083). But does not converge to sweet spot (5.0) under heavy overload.

## 6. OOM recovery details (for reference)

### What it does

When `prepare_for_extend()` raises RuntimeError("out of memory"):
1. Free req pool slots for new reqs (those that don't hold KV yet)
2. Put `can_run_list` back at front of `waiting_queue`
3. Reset `chunked_req` if the OOM was on a chunked prefill
4. Decrement `inflight_middle_chunks` if another chunk is in flight
5. Set grace period (10 steps) to skip invariant checker
6. Increment consecutive OOM counter
7. Return `(None, running_batch)` — skip this prefill step, decode continues

### Chunk downscaling

After 3+ consecutive OOMs, `chunked_prefill_size` is capped to `max(available_memory, page_size)`. This reduced OOM from 559 to 21 events (96% reduction).

### What to watch on 8xH200

- OOM events should be rare (more memory per GPU). If frequent, check mem-fraction-static.
- The `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` env var is required. Without it, the invariant checker will crash on the accounting gap from OOM recovery's `evict_from_tree_cache` side effect.
- The grace period (10 steps) may need tuning if the invariant checker fires after OOM recovery on 8xH200.

## 7. Files modified (for cherry-pick reference)

| File | Changes |
|------|---------|
| `python/sglang/srt/managers/schedule_policy.py` | `AdaptiveBoostGamma` class (size-aware fix), `record_latencies` accepts `works`, `record_completed_latencies` collects work sizes |
| `python/sglang/srt/managers/scheduler.py` | OOM recovery (try/except around `prepare_for_extend`), chunk downscaling, invariant checker grace period, OOM counters |
| `python/sglang/srt/arg_groups/fields/schedule.py` | `boost_adaptive_gamma`, `boost_adaptive_max_jump_s`, `boost_clock` args |
| `test/registered/unit/managers/test_schedule_policy.py` | Updated `_latencies` -> `_samples`, 59 tests pass |
| `ph2lio/experiments/e2/scripts/start_server.sh` | `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` |
| `ph2lio/experiments/e2/docs/e2_policies.md` | Policy comparison results, cold/warm analysis, stretch vs cold wait |
| `ph2lio/experiments/e2/docs/future_plan_e2_policies.md` | This file — 8xH200 deployment plan, findings, recommendations |
| `ph2lio/experiments/e2/scripts/rerun_dfs_fcfs.sh` | Rerun dfs-weight + fcfs with commit 60060211, pin=0 |
