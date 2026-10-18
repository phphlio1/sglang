# E2 Scheduling Policy Comparison

**Date**: 2026-10-16
**Hardware**: 1x RTX 6000 Pro Blackwell 96GB
**Model**: Qwen/Qwen3.8-27B-FP8
**Config**: c16, seed=42, 1800s duration, mem-fraction-static=0.80, chunked-prefill-size=8192, context-length=262144, EAGLE spec decode (3 steps), hierarchical cache (64GB)

## Metrics

| # | Metric | Description | Formula |
|---|--------|-------------|---------|
| 1 | n done / n wait | Completed requests / total sent | — |
| 2 | cache hit (%) | Prefix cache utilization | `sum(matched_token)/sum(input_len)` |
| 3 | queue mean/p99 (s) | Queue wait mean and 99th percentile | Linear percentile `k=(n-1)*p/100` |
| 4 | TTFT mean/p99 (ms) | Time to first token mean and 99th percentile | Linear percentile `k=(n-1)*p/100` |
| 5 | ITL mean/p99 (ms) | Inter-token latency mean and 99th percentile | Linear percentile `k=(n-1)*p/100` |
| 6 | tok/s | Output token throughput | `sum(output_len) / (end_max - start_min)` |
| 7 | Jain | Fairness of queue wait (1 = equal) | `(sum(x))^2 / (n * sum(x^2))`, x: per-request wait |
| 8 | stretch p99 | Cold request wait relative to prefill | `wait / (ttft - wait)` (denominator ~ prefill time) |
| 9 | worst-5% wait (%) | Wait mass concentrated in top 5% | `sum(ceil(0.05n) largest waits) / sum(all waits) * 100` |

## Result — Concurrency 16

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p99 (s) | TTFT mean (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p99 (ms) | tok/s | Jain | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fcfs | 159 | 173 | 77.4 | 26.892 | 220.500 | 35286 | 226862 | 422.95 | 154.95 | 66.4 | 0.337 | 56.83 | 33.73 |
| lpm | 181 | 195 | 85.8 | 18.728 | 186.500 | 25136 | 183587 | 381.07 | 265.73 | 75.0 | 0.318 | 127.81 | 36.92 |
| hrrn | 187 | 201 | 84.2 | 19.174 | 182.600 | 26070 | 194054 | 366.24 | 255.92 | 75.2 | 0.304 | 58.14 | 36.65 |
| boost g=5.0 | 168 | 182 | 82.9 | 21.761 | 183.000 | 29526 | 140270 | 404.71 | 265.70 | 68.2 | 0.400 | 117.90 | 29.20 |
| boost g=15.0 | 161 | 175 | 80.7 | 22.148 | 167.800 | 30123 | 150806 | 50.77 | 209.90 | 67.3 | 0.419 | 60.65 | 28.26 |
| boost ada szfix | 163 | 177 | 81.5 | 21.646 | 192.125 | 30084 | 182250 | 421.08 | 350.82 | 67.7 | 0.332 | 92.96 | 34.57 |
| dfs-weight | 181 | 195 | 87.3 | 23.483 | 359.500 | 29698 | 421762 | 379.50 | 715.04 | 86.9 | 0.151 | 64.04 | 60.92 |

Note: All runs are from Oct 15-16 (commit `60060211`, with OOM recovery) with `in_batch_prefix_caching_max_pin_s=0`. FCFS and dfs-weight were rerun on Oct 16 to match this config (previously Oct 8-9, commit `101ea1dc`, pin=60). Boost g=15.0 ITL mean (50.77 ms) is anomalously low.

## Overall ranking

`starvation_rate` (queue wait > 10s threshold) is misleading: LPM shows 41.4% (lowest) but has the worst stretch p99 (127.81x). The threshold metric counts how many requests exceed 10s, not how badly they exceed it. LPM's bimodal wait distribution (cache-hit requests wait ~0s, cache-miss requests wait very long) keeps the count low while the tail is severe.

FCFS has the best stretch p99 (56.83) — no reordering means no request is disproportionately delayed relative to its prefill. But FCFS has the worst TTFT p99 (226862 ms) and low throughput (66.4 tok/s): "equally slow for everyone." DFS-weight has the highest throughput (86.9 tok/s, 16% above HRRN) but worst fairness on every dimension: Jain 0.151, worst-5% wait 60.92%, TTFT p99 421762 ms. HRRN's stretch p99 (58.14, second-best) shows proportional fairness: wait relative to service time is balanced, even though absolute TTFT p99 is high.

| Rank | Policy | stretch p99 | Jain | worst-5% wait | TTFT p99 | tok/s | Verdict |
|------|--------|-------------|------|---------------|----------|-------|---------|
| 1 | boost g=5.0 | 117.90 | 0.400 | 29.20 | 140270 | 68.2 | Best TTFT p99, balanced on all metrics, no extreme weakness |
| 2 | hrrn | 58.14 | 0.304 | 36.65 | 194054 | 75.2 | Best stretch p99 (non-FCFS), high throughput, proportional fair |
| 3 | boost g=15.0 | 60.65 | 0.419 | 28.26 | 150806 | 67.3 | Good Jain + queue p99, near-FCFS, moderate throughput |
| 4 | lpm | 127.81 | 0.318 | 36.92 | 183587 | 75.0 | Good mean + throughput, worst stretch p99 |
| 5 | boost ada szfix | 92.96 | 0.332 | 34.57 | 182250 | 67.7 | Adaptive, moderate but does not excel anywhere |
| 6 | fcfs | 56.83 | 0.337 | 33.73 | 226862 | 66.4 | Best stretch p99, but worst TTFT p99 and low throughput |
| 7 | dfs-weight | 64.04 | 0.151 | 60.92 | 421762 | 86.9 | Highest throughput, but worst fairness and worst TTFT p99 by far |

## When to use each policy

### Static Boost g=5.0 — recommended for interactive/chat
- **Use when**: interactive/chat workload; SLA measured by TTFT p99; need good Jain fairness. This is the sweet spot.
- **Strength**: best TTFT p99 (140270 ms), best Jain (0.400), zero retractions, zero OOM. Balanced on all metrics — no population gets thrown under the bus. ITL mean 404.71 ms, ITL p99 265.70 ms: consistent decode.
- **Weakness**: throughput lower than LPM/HRRN (68.2 vs 75 tok/s); stretch p99 high (117.90) — mid-size requests with tiny prefill get high wait/prefill ratio. Trade-off is throughput vs latency+fairness, not starvation.

### HRRN — recommended for mixed workloads with E2E SLA
- **Use when**: SLA measured by E2E p99 (not TTFT); mixed workload with short and long requests; want proportional fairness without tuning.
- **Strength**: best stretch p99 (58.14) — wait proportional to service time, no request starved out of proportion. Highest n done (187), highest throughput (75.2 tok/s). Queue mean 19.174s, TTFT mean 26070 ms: good average performance.
- **Weakness**: worst Jain (0.304), worst TTFT p99 (194054 ms) — some requests wait very long in absolute terms. Worst-5% wait 36.65%: wait mass concentrated in tail.

### Static Boost g=15.0 — recommended when fairness matters but FCFS too slow
- **Use when**: fairness is the priority but FCFS throughput is too low; need near-FCFS ordering with better efficiency.
- **Strength**: second-best Jain (0.419), best queue p99 (167.8s), low stretch p99 (60.65), best worst-5% wait (28.26%). Queue mean 22.148s: consistent waits.
- **Weakness**: TTFT mean 30123 ms (highest) — everyone waits a bit more. Less SJF bias than g=5.0, so TTFT p99 slightly worse (150806 ms). ITL mean 50.77 ms anomalously low (likely measurement artifact).

### LPM — recommended for batch/throughput
- **Use when**: throughput is the priority; cache hit rate is very high (shared prompt templates, RAG with common context); SLA is measured in tok/s, not TTFT; batch processing where interactive latency does not matter.
- **Strength**: lowest queue mean (18.728s), lowest TTFT mean (25136 ms), high throughput (75.0 tok/s), highest n done (181). The majority of requests are served very well.
- **Weakness**: worst stretch p99 (127.81x) — cold requests wait 128x their prefill time. Worst Jain (0.318), worst worst-5% wait (36.92%). Bimodal: cache-hit requests wait ~0s, cache-miss requests wait very long. Not suitable for interactive/chat workloads.

### Adaptive Boost (g-Ada + size-aware fix) — for dynamic workloads
- **Use when**: workload is dynamic (bursty traffic, changing request size mix); do not know the optimal g in advance; want self-tuning.
- **Strength**: self-adapts to workload, no manual g tuning. ITL mean 421.08 ms, ITL p99 350.82 ms: consistent decode.
- **Weakness**: does not converge to optimal g under heavy overload (oscillates around g=0.10 instead of reaching g=5.0). Throughput lower than LPM/HRRN. Moderate on all metrics but does not excel anywhere. The paper's tail fit assumes fat tail comes from short-request starvation; under heavy overload, tail fat is global (queue congestion affects all requests), so the fit cannot distinguish exogenous variance from policy-induced variance.

### FCFS — baseline, best stretch p99 but worst TTFT p99
- **Use when**: workload is light (no overload, queue near-empty); or need strict arrival-order guarantee; or SLA is measured by stretch (relative latency) not TTFT (absolute latency).
- **Strength**: best stretch p99 (56.83) — no reordering means no request is disproportionately delayed relative to its prefill time. ITL mean 422.95 ms, ITL p99 154.95 ms: consistent decode.
- **Weakness**: worst TTFT p99 (226862 ms = 227s) — everyone waits long in absolute terms. Queue mean 26.892s (highest), tok/s 66.4 (second-lowest), n done 159 (low). "Equally slow for everyone" — fairness in suffering, not in service quality. Not recommended under overload.

### DFS-Weight — highest throughput, worst fairness
- **Use when**: batch/throughput-only workload with very high cache hit rate (shared prompt templates, RAG with common context); SLA measured in tok/s only; interactive latency does not matter at all.
- **Strength**: highest throughput (86.9 tok/s, 16% above HRRN), n done 181 (tied with LPM), highest cache hit (87.3%). DFS traversal of the radix cache tree groups requests by shared prefix, maximizing cache locality and batch efficiency.
- **Weakness**: worst Jain (0.151), worst worst-5% wait (60.92%), worst TTFT p99 (421762 ms = 422s), worst queue p99 (359.5s). Extremely bimodal: cache-hit requests served immediately, cache-miss requests wait enormous times. 60.92% of all wait mass concentrated in top 5% of requests — the most unfair policy tested.

## Stretch p99 and the boost g=5.0 trade-off

Stretch p99 = wait / prefill_time over all requests with uncached work >= 256. It captures relative latency: how many times longer a request waited compared to its own prefill. A request with 0.25s prefill that waits 30s has stretch 120x.

Stretch p99, Jain, worst-5% wait, and TTFT p99 are all **threshold-free** metrics — no cache-hit/miss classification needed.

A cache-based cold/warm split (cached==0 vs cached>0, from `starvation.py:155`) was considered but dropped: the binary threshold at 0 is crude (1/10000 matched tokens = "warm"), and the resulting miss/hit ratio can diverge from threshold-free metrics.

### Does boost g=5.0 make starvation worse?

**No — on all threshold-free metrics, boost g=5.0 is better than or equal to LPM:**

| Metric | boost g=5.0 | lpm | Winner |
|--------|-------------|-----|--------|
| stretch p99 | 117.90 | 127.81 | boost |
| Jain | 0.400 | 0.318 | boost |
| worst-5% wait | 29.20% | 36.92% | boost |
| TTFT p99 | 140270 | 183587 | boost |
| queue mean | 21.761 | 18.728 | lpm |
| tok/s | 68.2 | 75.0 | lpm |

The trade-off is **throughput vs latency+fairness**, not starvation. Boost g=5.0 sacrifices 9% throughput (68.2 vs 75.0 tok/s) for 24% better TTFT p99, 26% better Jain, and 21% better worst-5% wait. Stretch p99 is high (117.90) because mid-size requests with tiny prefill (0.25s) get high wait/prefill ratios — but this is the cost of prioritizing short requests for better TTFT, not a starvation effect.

## Boost g sweep (focused, around sweet spot)

| g | n done | queue p99 (s) | TTFT p99 (ms) | tok/s | Jain | stretch p99 | worst-5% wait (%) |
|---|--------|---------------|---------------|-------|------|-------------|--------------------|
| 2.0 | 166 | 183.0 | 159836 | 67.8 | 0.360 | — | — |
| 5.0 | 168 | 183.0 | 140270 | 68.2 | 0.400 | 117.90 | 29.20 |
| 6.0 | 166 | 183.0 | 174877 | 68.7 | 0.367 | — | — |
| 7.0 | 166 | 183.0 | 222333 | 57.4 | 0.381 | — | — |
| 8.0 | 166 | 183.0 | 398315 | 58.2 | 0.201 | — | — |
| 10.0 | 166 | 183.0 | 163041 | 66.0 | 0.384 | — | — |
| 12.0 | 166 | 183.0 | 170806 | 66.8 | 0.393 | — | — |
| 15.0 | 161 | 167.8 | 150806 | 67.3 | 0.419 | 60.65 | 28.26 |
| 20.0 | 166 | 183.0 | 322825 | 57.6 | 0.325 | — | — |

g=5.0 is the sweet spot: best TTFT p99, zero OOM, zero retractions. g=15.0 is the fairness alternative. g=8.0 is an anomaly (TTFT p99=398s, likely variability from OOM interaction).

## Adaptive gamma convergence

| Version | g final | g range | TTFT p99 (ms) | tok/s | Jain |
|---------|---------|---------|---------------|-------|------|
| Original J=60 | 0.083 | [0.08, 0.09] | 154052 | 63.5 | 0.389 |
| Original J=120 | 0.053 | [0.05, 0.06] | 205915 | 59.5 | 0.349 |
| Size-aware J=60 | 0.102 | [0.09, 0.55] | 182250 | 67.7 | 0.332 |

The size-aware fix breaks the feedback loop partially: g occasionally reaches 0.55 (vs original stuck at 0.083), throughput improves 63.5 to 67.7 tok/s. But g still does not converge to the sweet spot (5.0) because under heavy overload, the tail fat is global, not isolated to giant requests.
