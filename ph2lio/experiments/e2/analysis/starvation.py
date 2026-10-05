"""Anti-starvation metrics computed from per-request benchmark data.

Starvation = a request waits disproportionately long in the scheduling queue
relative to its peers. LPM (longest-prefix-match) prioritizes warm-cache
requests, which can starve cold-cache requests. HRRN (highest-response-ratio-
next) adds aging to lift starved requests sooner. These metrics quantify that
difference using per-request TTFT, cached_tokens, and input_lens arrays
already captured by --output-details --cache-report in the benchmark output.

All functions are pure: arrays in, scalar metrics out. No I/O.
"""

import math
from typing import Optional


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def jains_fairness_index(values: list[float]) -> float:
    """Jain's fairness index: (sum)^2 / (n * sum_of_squares).

    Range [1/n, 1]. 1 = perfectly fair (all equal). Lower = more unequal.
    """
    n = len(values)
    if n == 0:
        return float("nan")
    s = sum(values)
    sq = sum(v * v for v in values)
    if sq == 0:
        return 1.0
    return (s * s) / (n * sq)


def coefficient_of_variation(values: list[float]) -> float:
    """std / mean. Lower = more uniform (fairer). Dimensionless."""
    n = len(values)
    if n == 0:
        return float("nan")
    mean = sum(values) / n
    if mean == 0:
        return float("nan")
    var = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(var) / mean


def max_mean_ratio(values: list[float]) -> float:
    """max / mean. 1.0 = perfectly fair. Higher = worst-case starvation."""
    n = len(values)
    if n == 0:
        return float("nan")
    mean = sum(values) / n
    if mean == 0:
        return float("nan")
    return max(values) / mean


def p99_p50_ratio(values: list[float]) -> float:
    """p99 / p50. Tail-to-median ratio. Higher = heavier tail (starvation)."""
    if len(values) < 2:
        return float("nan")
    s = sorted(values)
    p50 = _percentile(s, 50)
    p99 = _percentile(s, 99)
    if p50 == 0:
        return float("nan")
    return p99 / p50


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation coefficient.

    For TTFT vs cached_tokens: negative correlation means high cache -> low TTFT
    (cache-aware scheduling working). Near-zero means no cache-based ordering.
    We return correlation of (cached_tokens, ttft) so negative = cache helps.
    """
    n = len(x)
    if n < 3:
        return float("nan")

    def rank(vals: list[float]) -> list[float]:
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def ols_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary-least-squares slope of ys on xs, with R^2.

    Returns (slope, r_squared); (nan, nan) when undefined (n < 2 or zero
    x-variance). For the queue-wait vs arrival-order starvation slope: a
    slope near 0 means no cohort of arrivals waits systematically longer
    (the anti-starvation target); R^2 says how much of the wait the trend
    actually explains.
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return float("nan"), float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    syy = sum((y - mean_y) ** 2 for y in ys)
    if syy == 0:
        # All ys equal: the flat fit is exact, so slope 0 with R^2 1.
        return 0.0, 1.0
    return sxy / sxx, (sxy * sxy) / (sxx * syy)


def cache_hit_rate(cached_tokens: list[int], input_lens: list[int]) -> float:
    """Token-weighted prefix-cache hit rate: sum(cached) / sum(input).

    Token weighting answers "what share of prefill work did the cache save";
    a request-weighted rate would let one long cold request swamp many short
    hits. Arrays must be index-aligned.
    """
    total_input = sum(input_lens)
    if total_input <= 0:
        return float("nan")
    return sum(cached_tokens) / total_input


def cold_warm_ttft_split(
    ttfts: list[float],
    cached_tokens: list[int],
) -> tuple[list[float], list[float]]:
    """Split TTFTs into cold-cache (cached=0) and warm-cache (cached>0) groups."""
    cold = [t for t, c in zip(ttfts, cached_tokens) if c == 0]
    warm = [t for t, c in zip(ttfts, cached_tokens) if c > 0]
    return cold, warm


def compute_starvation_metrics(
    ttfts_sec: list[float],
    cached_tokens: list[int],
    input_lens: Optional[list[int]] = None,
) -> dict:
    """Compute all anti-starvation metrics from per-request data.

    Args:
        ttfts_sec: per-request TTFT in seconds (from benchmark --output-details).
        cached_tokens: per-request cached prefix token count (from --cache-report).
        input_lens: per-request input prompt length (optional, for normalized TTFT).

    Returns:
        Dict of scalar metrics (all TTFT-based values in milliseconds).
    """
    n = len(ttfts_sec)
    if n == 0:
        return {}

    ttfts_ms = [t * 1000.0 for t in ttfts_sec]
    ttfts_sorted = sorted(ttfts_ms)

    result: dict = {}

    # --- TTFT fairness metrics (overall) ---
    result["ttft_fairness_jain"] = jains_fairness_index(ttfts_ms)
    result["ttft_cv"] = coefficient_of_variation(ttfts_ms)
    result["ttft_max_mean_ratio"] = max_mean_ratio(ttfts_ms)
    result["ttft_p99_p50_ratio"] = p99_p50_ratio(ttfts_ms)

    # --- Cache-based starvation metrics ---
    has_cache = any(c > 0 for c in cached_tokens) and any(c == 0 for c in cached_tokens)
    cold, warm = cold_warm_ttft_split(ttfts_ms, cached_tokens)

    if has_cache and cold and warm:
        cold_mean = sum(cold) / len(cold)
        warm_mean = sum(warm) / len(warm)
        result["cold_cache_n"] = len(cold)
        result["warm_cache_n"] = len(warm)
        result["cold_cache_ttft_mean_ms"] = cold_mean
        result["warm_cache_ttft_mean_ms"] = warm_mean
        result["cold_warm_ttft_ratio"] = (
            cold_mean / warm_mean if warm_mean > 0 else float("nan")
        )

        cold_sorted = sorted(cold)
        warm_sorted = sorted(warm)
        result["cold_cache_ttft_p99_ms"] = _percentile(cold_sorted, 99)
        result["warm_cache_ttft_p99_ms"] = _percentile(warm_sorted, 99)
        result["cold_cache_ttft_max_ms"] = cold_sorted[-1]
        result["warm_cache_ttft_max_ms"] = warm_sorted[-1]

        # Spearman correlation: negative = cache helps reduce TTFT (cache-aware
        # scheduling working), near-zero = no cache-based ordering, positive =
        # cache inversely correlated (unusual, possibly pathological).
        result["ttft_cache_correlation"] = spearman_rank_correlation(
            cached_tokens, ttfts_ms
        )
    else:
        # All cold or all warm -- no cache-based split possible.
        result["cold_cache_n"] = len(cold)
        result["warm_cache_n"] = len(warm)
        result["cold_warm_ttft_ratio"] = float("nan")
        result["ttft_cache_correlation"] = float("nan")

    # --- Normalized TTFT (TTFT per input token) fairness ---
    if input_lens:
        norm_ttfts = [t / il if il > 0 else 0.0 for t, il in zip(ttfts_ms, input_lens)]
        norm_ttfts = [v for v in norm_ttfts if v > 0]
        if norm_ttfts:
            result["norm_ttft_fairness_jain"] = jains_fairness_index(norm_ttfts)
            result["norm_ttft_cv"] = coefficient_of_variation(norm_ttfts)

    return result


def load_per_request_data(summary_path: str) -> tuple[list, list, list]:
    """Load per-request arrays from a bench_summary.jsonl file.

    Returns (ttfts_sec, cached_tokens, input_lens). Empty lists if file
    missing or fields absent.
    """
    import json
    from pathlib import Path

    p = Path(summary_path)
    if not p.exists():
        return [], [], []

    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ttfts = d.get("ttfts", [])
            cached = d.get("cached_tokens", [])
            input_lens = d.get("input_lens", [])
            return ttfts, cached, input_lens

    return [], [], []
