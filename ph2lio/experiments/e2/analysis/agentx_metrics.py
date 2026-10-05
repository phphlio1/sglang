"""E2 AgentX metrics computation from AIPerf run outputs.

Reads one run directory (produced by scripts/run_agentx_policy.sh or
run_agentx_smoke.sh) and computes the E2 target metrics:

  - TTFT / TPOT / E2E mean + P95          (per-request, profile_export.jsonl)
  - RPM / TPM                              (aggregate, profile_export_aiperf.json)
  - Queue waiting time mean/P95/P99        (server histogram sglang:queue_time_seconds)
  - Starvation Rate                        (queue wait above --starvation-threshold)
  - Service Difference                     (P99 - P50 of queue wait)
  - Jain fairness on TTFT, cold/warm TTFT  (per-request, cached_tokens split)
  - Per-request queue wait                 (queue_wait_per_request block; only when
                                            the run passed --log-schedule-eval-path):
                                            cold/warm split, gap, per-class starvation
                                            rate + arrival-order slope, first-attempt
                                            cache hit rate, retractions, queue length

Known observability limits (documented, not approximated):
  - Queue waiting time MAX is not derivable: the Prometheus histogram's
    +Inf bucket has no upper bound. (The per-request block below reports a
    real max when the schedule-eval log is present.)
  - Cold-vs-Warm Queue-Wait Gap is not computable from the server side
    alone: queue wait exists only as an aggregate histogram; there is no
    per-request queue wait to join with per-request cache hits. Runs that
    passed --log-schedule-eval-path get the real thing in the
    queue_wait_per_request block. Without the log, the TTFT-based
    cold_warm_ttft_gap_p99_ms below is a clearly-labeled PROXY, not queue
    wait. AIPerf has no queue-time metric; TTFT includes network + queue +
    prefill.
  - Queue-wait percentiles are bounded by the histogram bucket resolution
    (standard Prometheus linear interpolation, not the polynomial method).

Usage:
    python3 analysis/agentx_metrics.py <run_dir> [--starvation-threshold 10]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starvation import (
    _percentile,
    cache_hit_rate,
    cold_warm_ttft_split,
    compute_starvation_metrics,
    jains_fairness_index,
    ols_slope,
)

QUEUE_TIME_FAMILY_CANDIDATES = (
    "sglang:queue_time_seconds",
    "sglang_queue_time_seconds",
)
RETRACTION_COUNTER_FAMILY = "sglang:num_retracted_requests_total"
RETRACTION_INTERVAL_GAUGE_FAMILY = "sglang:num_retracted_reqs"
QUEUE_LEN_FAMILY_CANDIDATES = ("sglang:num_queue_reqs", "sglang_num_queue_reqs")


def _clean(value):
    # NaN -> None so output stays valid JSON (json.dumps writes bare NaN).
    if isinstance(value, float) and value != value:
        return None
    return value


def load_profile_records(run_dir: Path) -> list[dict]:
    """Per-request records from profile_export.jsonl (records export level)."""
    records = []
    path = run_dir / "profile_export.jsonl"
    for line in path.open():
        records.append(json.loads(line))
    return records


def load_aggregate(run_dir: Path) -> dict:
    path = run_dir / "profile_export_aiperf.json"
    return json.loads(path.read_text())


def _metric_value(record: dict, tag: str) -> float | None:
    entry = record.get("metrics", {}).get(tag)
    if entry is None:
        return None
    return entry.get("value")


def extract_per_request(records: list[dict]) -> dict:
    """Aligned per-request arrays (error records excluded from latency stats).

    Records missing any of the five tags are dropped entirely so every array
    stays index-aligned (TTFT/cached/ISL joins must not mis-pair).
    """
    ok = [r for r in records if not r.get("error")]
    rows = []
    for r in ok:
        ttft = _metric_value(r, "time_to_first_token")
        lat = _metric_value(r, "request_latency")
        isl = _metric_value(r, "input_sequence_length")
        osl = _metric_value(r, "output_sequence_length")
        cached = _metric_value(r, "usage_prompt_cache_read_tokens")
        if None in (ttft, lat, isl, osl, cached):
            continue
        rows.append((ttft, lat, isl, osl, int(cached)))

    ttft_s = [r[0] for r in rows]
    lat_s = [r[1] for r in rows]
    isl = [int(r[2]) for r in rows]
    osl = [int(r[3]) for r in rows]
    cached = [r[4] for r in rows]

    tpot_s = [
        (lat - ttft) / (o - 1) for ttft, lat, o in zip(ttft_s, lat_s, osl) if o >= 2
    ]

    return {
        "n_records": len(records),
        "n_error": len(records) - len(ok),
        "n_complete": len(rows),
        "ttft_s": ttft_s,
        "latency_s": lat_s,
        "isl": isl,
        "osl": osl,
        "cached_tokens": cached,
        "tpot_s": tpot_s,
    }


def percentile_stats(values_s: list[float]) -> dict:
    if not values_s:
        return {"n": 0, "mean_ms": None, "p95_ms": None}
    ms = sorted(v * 1000.0 for v in values_s)
    return {
        "n": len(ms),
        "mean_ms": sum(ms) / len(ms),
        "p95_ms": _percentile(ms, 95),
    }


def load_queue_histogram(run_dir: Path) -> dict | None:
    """Delta (first->last PROFILING scrape) of the sglang queue-time histogram.

    Returns {"buckets": {le: count}, "sum": s, "count": c} for the profiling
    window, or None when the export is missing / has no profiling-phase
    scrapes of the family.
    """
    path = run_dir / "server_metrics_export.jsonl"
    if not path.exists():
        return None

    first = None
    last = None
    for line in path.open():
        rec = json.loads(line)
        if rec.get("benchmark_phase") != "profiling":
            continue
        for name, samples in rec.get("metrics", {}).items():
            if not any(c in name for c in QUEUE_TIME_FAMILY_CANDIDATES):
                continue
            sample = samples[0]
            if sample.get("buckets") is None:
                continue
            snap = {
                "buckets": dict(sample["buckets"]),
                "sum": sample.get("sum"),
                "count": sample.get("count"),
            }
            if first is None:
                first = snap
            last = snap
    if first is None or last is None or last["count"] is None or first["count"] is None:
        return None

    delta_count = last["count"] - first["count"]
    if delta_count <= 0:
        return None
    delta_buckets = {
        le: last["buckets"].get(le, 0.0) - first["buckets"].get(le, 0.0)
        for le in last["buckets"]
    }
    return {
        "buckets": {le: c for le, c in delta_buckets.items() if c >= 0},
        "sum": (last["sum"] or 0.0) - (first["sum"] or 0.0),
        "count": delta_count,
    }


def prometheus_quantile(buckets: dict[str, float], q: float) -> float | None:
    """Standard Prometheus histogram_quantile: linear interpolation in the
    bucket that brackets the q-th observation. +Inf gets the highest finite
    bound (the only honest statement for an unbounded bucket)."""
    finite = sorted(
        ((float(le), c) for le, c in buckets.items() if le not in ("+Inf", "inf")),
        key=lambda x: x[0],
    )
    if not finite:
        return None
    total = finite[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_bound, prev_count = 0.0, 0.0
    for bound, count in finite:
        if count >= rank:
            if count == prev_count:
                return bound
            frac = (rank - prev_count) / (count - prev_count)
            return prev_bound + (bound - prev_bound) * frac
        prev_bound, prev_count = bound, count
    return finite[-1][0]


def compute_queue_metrics(hist: dict | None, starvation_threshold_s: float) -> dict:
    if hist is None:
        return {
            "queue_time_available": False,
            "note": "server_metrics_export.jsonl missing or no profiling-phase sglang:queue_time_seconds samples; server needs --enable-metrics",
        }

    count = hist["count"]
    mean_s = hist["sum"] / count if hist["sum"] is not None else None
    buckets = hist["buckets"]

    finite = sorted(
        ((float(le), c) for le, c in buckets.items() if le not in ("+Inf", "inf")),
        key=lambda x: x[0],
    )
    total = finite[-1][1] if finite else 0

    # Share of requests whose queue wait exceeds the threshold: the count in
    # buckets strictly above the first bound >= threshold, over total.
    above = 0
    for bound, c in finite:
        if bound >= starvation_threshold_s:
            above = total - c
            break

    p99 = prometheus_quantile(buckets, 0.99)
    p95 = prometheus_quantile(buckets, 0.95)
    p50 = prometheus_quantile(buckets, 0.50)

    return {
        "queue_time_available": True,
        "queue_time_n": int(count),
        "queue_time_mean_s": mean_s,
        "queue_time_p50_s": p50,
        "queue_time_p95_s": p95,
        "queue_time_p99_s": p99,
        "queue_time_max_s": None,  # +Inf bucket is unbounded; not derivable
        "starvation_threshold_s": starvation_threshold_s,
        "starvation_rate": above / total if total > 0 else None,
        "service_difference_p99_p50_s": (p99 - p50)
        if (p99 is not None and p50 is not None)
        else None,
        "histogram_bucket_resolution_note": "percentiles are linear-interpolated within Prometheus buckets; resolution is bounded by bucket width",
    }


def load_run_meta(run_dir: Path) -> dict:
    path = run_dir / "run_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_schedule_eval_records(run_dir: Path) -> list[dict] | None:
    """This run's per-request schedule-eval records, or None when absent.

    The snapshot schedule_eval.jsonl may carry earlier runs' records from a
    reused server; run_meta.json's schedule_eval.offset_lines slices those
    off so only this run's requests are analyzed.
    """
    path = run_dir / "schedule_eval.jsonl"
    if not path.exists():
        return None
    block = load_run_meta(run_dir).get("schedule_eval") or {}
    offset = block.get("offset_lines")
    offset = offset if isinstance(offset, int) and offset >= 0 else 0
    records = [json.loads(line) for line in path.open() if line.strip()]
    return records[offset:]


def load_profiling_series(run_dir: Path, family: str) -> list[float] | None:
    """One metric family's values across PROFILING-phase scrapes (samples[0])."""
    path = run_dir / "server_metrics_export.jsonl"
    if not path.exists():
        return None
    series: list[float] = []
    for line in path.open():
        rec = json.loads(line)
        if rec.get("benchmark_phase") != "profiling":
            continue
        samples = rec.get("metrics", {}).get(family)
        if samples and samples[0].get("value") is not None:
            series.append(float(samples[0]["value"]))
    return series or None


def _queue_wait_class_stats(
    waits_s: list[float], starvation_threshold_s: float
) -> dict:
    if not waits_s:
        return {
            "n": 0,
            "mean_s": None,
            "p95_s": None,
            "p99_s": None,
            "max_s": None,
            "starvation_rate": None,
        }
    ordered = sorted(waits_s)
    starved = sum(1 for w in ordered if w > starvation_threshold_s)
    return {
        "n": len(ordered),
        "mean_s": sum(ordered) / len(ordered),
        "p95_s": _percentile(ordered, 95),
        "p99_s": _percentile(ordered, 99),
        "max_s": ordered[-1],
        "starvation_rate": starved / len(ordered),
    }


def _retraction_count(run_dir: Path, eval_records: list[dict] | None) -> dict:
    """Retractions during the profiling window.

    Preference order: the monotonic server counter (last - first scrape),
    then the per-interval server gauge, then the schedule-eval log's
    per-request retraction_count sum.
    """
    out: dict = {"value": None, "source": None}
    counter = load_profiling_series(run_dir, RETRACTION_COUNTER_FAMILY)
    if counter is not None:
        out["value"] = counter[-1] - counter[0]
        out["source"] = "server_metrics_counter"
        # The counter family first appears in the export only after some
        # retraction has happened, so the window delta can miss events
        # before its first observation.
    else:
        gauge = load_profiling_series(run_dir, RETRACTION_INTERVAL_GAUGE_FAMILY)
        if gauge is not None:
            # The scheduler resets this gauge after every report, so the
            # window total is the sum of per-interval values, not a
            # last-minus-first delta.
            out["value"] = sum(gauge)
            out["source"] = "server_metrics_interval_gauge_sum"
        elif eval_records:
            out["value"] = sum(r.get("num_retractions") or 0 for r in eval_records)
            out["source"] = "schedule_eval_log"
    if eval_records:
        out["eval_log_cross_check"] = sum(
            r.get("num_retractions") or 0 for r in eval_records
        )
    return out


def _queue_len_stats(run_dir: Path) -> dict | None:
    """Waiting-queue-length gauge over the profiling window, when exported."""
    series = load_profiling_series(run_dir, QUEUE_LEN_FAMILY_CANDIDATES[0])
    if series is None:
        return None
    return {
        "mean": sum(series) / len(series),
        "max": max(series),
        "n_scrapes": len(series),
    }


def compute_queue_wait_per_request(
    run_dir: Path,
    eval_records: list[dict],
    starvation_threshold_s: float,
) -> dict:
    """Per-request queue-wait metrics from the schedule-eval log snapshot.

    Cold = zero prefix match at FIRST admission (first-attempt miss); warm =
    any match. This block is the real cold-vs-warm queue-wait measurement
    and supersedes the TTFT-gap proxy in the fairness block for runs that
    have the log.
    """
    usable = [
        r for r in eval_records if isinstance(r.get("queue_wait_s"), (int, float))
    ]
    cold = [r for r in usable if not r.get("matched_prefix_tokens_first_admission")]
    warm = [r for r in usable if r.get("matched_prefix_tokens_first_admission")]

    cold_stats = _queue_wait_class_stats(
        [r["queue_wait_s"] for r in cold], starvation_threshold_s
    )
    warm_stats = _queue_wait_class_stats(
        [r["queue_wait_s"] for r in warm], starvation_threshold_s
    )

    gap = {
        "p99_s": (
            cold_stats["p99_s"] - warm_stats["p99_s"]
            if cold_stats["p99_s"] is not None and warm_stats["p99_s"] is not None
            else None
        ),
        "mean_s": (
            cold_stats["mean_s"] - warm_stats["mean_s"]
            if cold_stats["mean_s"] is not None and warm_stats["mean_s"] is not None
            else None
        ),
    }

    # Arrival-order starvation slope: queue wait regressed on arrival rank
    # within the cold class. Slope ~0 = no arrival cohort waits
    # systematically longer (the anti-starvation target).
    cold_by_arrival = sorted(cold, key=lambda r: r.get("arrival_wall_s") or 0.0)
    waits = [r["queue_wait_s"] for r in cold_by_arrival]
    slope, r_squared = ols_slope(list(range(len(waits))), waits)

    first_attempt_hits = cache_hit_rate(
        [r.get("matched_prefix_tokens_first_admission") or 0 for r in eval_records],
        [r.get("input_len") or 0 for r in eval_records],
    )

    block = {
        "available": True,
        "n_records": len(eval_records),
        "n_with_queue_wait": len(usable),
        "cold_class": cold_stats,
        "warm_class": warm_stats,
        "cold_warm_queue_wait_gap": gap,
        "starvation_slope": {
            "slope_s_per_arrival": slope,
            "r_squared": r_squared,
            "n": len(waits),
        },
        "first_attempt_cache_hit_rate": first_attempt_hits,
        "retractions": _retraction_count(run_dir, eval_records),
        "waiting_queue_len": _queue_len_stats(run_dir),
        "note": "per-request queue wait from --log-schedule-eval-path; supersedes fairness.cold_warm_ttft_gap_p99_ms_proxy",
    }
    if not eval_records:
        block["note"] = (
            "snapshot has no records after the run offset; server may have "
            "rebooted or logging was disabled mid-run"
        )
    return block


def _agg_value(aggregate: dict, key: str) -> float | None:
    """Aggregate metric value: the export wraps each as {"unit": ..., "avg": ...}."""
    field = aggregate.get(key)
    if isinstance(field, dict):
        return field.get("avg")
    return field


def compute_metrics(run_dir: Path, starvation_threshold_s: float) -> dict:
    records = load_profile_records(run_dir)
    aggregate = load_aggregate(run_dir)
    per_req = extract_per_request(records)
    meta = load_run_meta(run_dir)
    eval_records = load_schedule_eval_records(run_dir)

    result = {
        "run_dir": str(run_dir),
        "n_records": per_req["n_records"],
        "n_error_records": per_req["n_error"],
        "n_complete_records": per_req["n_complete"],
    }

    if meta:
        result["run"] = {
            "policy": meta.get("policy"),
            "concurrency": meta.get("concurrency"),
            "duration_s": meta.get("duration_s"),
            "seed": meta.get("seed"),
            "boost_gamma": meta.get("boost_gamma"),
            "boost_bin_size": meta.get("boost_bin_size"),
        }

    result["ttft"] = percentile_stats(per_req["ttft_s"])
    result["e2e"] = percentile_stats(per_req["latency_s"])
    result["tpot"] = percentile_stats(per_req["tpot_s"])

    rps = _agg_value(aggregate, "request_throughput")
    out_tps = _agg_value(aggregate, "output_token_throughput")
    result["throughput"] = {
        "request_throughput_rps": rps,
        "request_per_min": (rps or 0) * 60,
        "output_token_throughput_tps": out_tps,
        "output_tokens_per_min": (out_tps or 0) * 60,
        "total_token_throughput_tps": _agg_value(aggregate, "total_token_throughput"),
    }

    result["queue"] = compute_queue_metrics(
        load_queue_histogram(run_dir), starvation_threshold_s
    )

    # Real per-request queue wait, only when the run passed
    # --log-schedule-eval-path; absent otherwise (the fairness block's
    # TTFT-gap proxy covers those runs).
    if eval_records is not None:
        result["queue_wait_per_request"] = compute_queue_wait_per_request(
            run_dir, eval_records, starvation_threshold_s
        )

    # Fairness / cold-warm from per-request TTFT + server-reported cached tokens.
    ttft_ms = [v * 1000.0 for v in per_req["ttft_s"]]
    cached = per_req["cached_tokens"]
    starvation = (
        compute_starvation_metrics(per_req["ttft_s"], cached, per_req["isl"])
        if per_req["ttft_s"]
        else {}
    )

    cold, warm = cold_warm_ttft_split(ttft_ms, cached) if ttft_ms else ([], [])
    result["fairness"] = {
        "ttft_fairness_jain": jains_fairness_index(ttft_ms) if ttft_ms else None,
        "ttft_cv": starvation.get("ttft_cv"),
        "cold_cache_n": starvation.get("cold_cache_n", len(cold)),
        "warm_cache_n": starvation.get("warm_cache_n", len(warm)),
        "cold_warm_ttft_ratio": starvation.get("cold_warm_ttft_ratio"),
        # PROXY for the (uncomputable) cold-vs-warm queue-wait gap: per-request
        # TTFT includes queue wait, so the cold/warm TTFT gap upper-bounds the
        # queue-wait gap with prefill time mixed in. Not queue wait itself.
        "cold_warm_ttft_gap_p99_ms_proxy": (
            _percentile(sorted(cold), 99) - _percentile(sorted(warm), 99)
            if cold and warm
            else None
        ),
        "cold_warm_queue_wait_gap_note": "not computable: queue wait is server-histogram-only; no per-request queue wait to join with cache classification",
        "ttft_cache_correlation": starvation.get("ttft_cache_correlation"),
    }

    return {k: _clean_rec(v) for k, v in result.items()}


def _clean_rec(obj):
    if isinstance(obj, dict):
        return {k: _clean_rec(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_rec(v) for v in obj]
    return _clean(obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--starvation-threshold",
        type=float,
        default=10.0,
        help="queue-wait seconds above which a request counts as starved (default 10)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    metrics = compute_metrics(args.run_dir, args.starvation_threshold)
    out_path = args.run_dir / "e2_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")

    if args.json:
        print(json.dumps(metrics, indent=2))
        return

    print(f"metrics written to {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
