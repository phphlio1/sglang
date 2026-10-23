"""E2 metrics v2: threshold-independent, per-request exact.

Companion to agentx_metrics.py (v1, frozen in each run dir's
e2_metrics.json). v2 replaces every threshold- or histogram-dependent
metric with per-request measurements from the schedule-eval log and the
records-level profile export (audit: docs/metrics_v2_audit.md):

  - queue mean/p50/p90/p95/p99 from eval-log queue_wait_s, exact; the v1
    histogram percentiles are bucket-resolution bounded (-40..-68% at
    c4/c8 p95, +13..+33% at c16 p99)
  - Jain on queue_wait_s, not TTFT (TTFT mixes legitimate prefill-time
    variance into "unfairness")
  - ITL from per-chunk gaps (inter_chunk_latency arrays) when present,
    honestly labeled; TPOT-based fallback otherwise
  - stretch (queue wait / own uncached prefill work) and worst-5% wait
    share instead of fixed-threshold starvation and cold/warm splits

Two populations, never joined (the eval log's rid is server-internal,
the profile export's x_request_id is the client UUID; intersection 0):

  P_done: profile_export records, profiling phase, no error, complete
          core tags -- n, TTFT, ITL, tok/s
  P_wait: schedule_eval records after the run offset, minus harness
          health checks, minus warmup arrivals, valid queue_wait_s --
          n, cache hit, queue, Jain, stretch, worst-5%

Writes e2_metrics_v2.json into the run dir. Never touches
e2_metrics.json: the c4/c8 cells lost server_metrics_export.jsonl to
the gzip policy and recomputation cannot reproduce their frozen
histogram block (verified in the audit).

Usage:
    python3 analysis/metrics_v2.py <run_dir> [--json]
    python3 analysis/metrics_v2.py --all [--results-root DIR]
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starvation import _percentile, jains_fairness_index

SCHEMA = "e2_metrics_v2/1"
# Uncached-work floor for stretch, in tokens: the boost bin size k;
# below one bin the "own prefill work" denominator is noise.
STRETCH_UNCACHED_FLOOR_TOKENS = 256
QUEUE_TIME_FAMILY_CANDIDATES = (
    "sglang:queue_time_seconds",
    "sglang_queue_time_seconds",
)


def _clean(value):
    # NaN -> None so output stays valid JSON (json.dumps writes bare NaN).
    if isinstance(value, float) and value != value:
        return None
    return value


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile, k = (n-1)*p/100; None on empty."""
    if not values:
        return None
    return _percentile(sorted(values), p)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def stats(values: list[float], ps: tuple[float, ...]) -> dict:
    """mean + named percentiles over one population's values."""
    out = {"n": len(values), "mean": mean(values)}
    out.update({f"p{int(p)}": percentile(values, p) for p in ps})
    return out


def stretch_values(
    waits_s: list[float],
    ttfts_s: list[float | None],
    uncached_tokens: list[int],
    floor_tokens: int = STRETCH_UNCACHED_FLOOR_TOKENS,
) -> tuple[list[float], int, int]:
    """queue_wait / (ttft - queue_wait) over requests with uncached work
    >= floor (how many times over its own prefill work a request waited).

    Returns (values, n_dropped_denominator, n_dropped_no_ttft): a
    non-positive denominator (ttft <= queue_wait) drops out and is
    counted; a missing ttft_s (request still queued when the window
    closed -- the eval log stamps its wait but never its service) drops
    out and is counted separately. Neither is silently kept.
    """
    values = []
    n_dropped_denominator = 0
    n_dropped_no_ttft = 0
    for wait, ttft, uncached in zip(waits_s, ttfts_s, uncached_tokens):
        if uncached < floor_tokens:
            continue
        if ttft is None:
            n_dropped_no_ttft += 1
            continue
        denominator = ttft - wait
        if denominator <= 0:
            n_dropped_denominator += 1
            continue
        values.append(wait / denominator)
    return values, n_dropped_denominator, n_dropped_no_ttft


def worst5_wait_share_pct(waits_s: list[float]) -> float | None:
    """Sum of the ceil(0.05*n) largest waits over the total wait mass, %.

    Threshold-free tail-concentration measure: 5% of requests holding
    100% of the wait reads 100; an even spread of n >= 20 reads 5.
    """
    n = len(waits_s)
    if n == 0 or sum(waits_s) <= 0:
        return None
    k = math.ceil(0.05 * n)
    top = sum(sorted(waits_s, reverse=True)[:k])
    return 100.0 * top / sum(waits_s)


def jain_queue_wait(waits_s: list[float]) -> float | None:
    """Jain's fairness index on queue waits; None on empty. All-zero
    waits (nothing waited) count as perfectly fair: 1.0."""
    if not waits_s:
        return None
    return jains_fairness_index(waits_s)


def itl_chunk_gaps(rows: list[dict]) -> list[float] | None:
    """Pooled per-chunk gaps (ms) across P_done rows, or None when no row
    carries an inter_chunk_latency array (the TPOT-based fallback then
    applies). Each array value is the gap between two consecutive
    received chunks, including the ttft -> second-token gap."""
    gaps: list[float] = []
    for row in rows:
        gaps.extend(row["chunk_gaps_ms"])
    return gaps or None


def tpot_values_ms(rows: list[dict]) -> list[float]:
    """Per-request (latency - ttft) / (osl - 1) in ms, osl >= 2 only."""
    return [
        (row["latency_s"] - row["ttft_s"]) * 1000.0 / (row["osl"] - 1)
        for row in rows
        if row["osl"] >= 2
    ]


def _metric(record: dict, tag: str) -> dict:
    return (record.get("metrics") or {}).get(tag) or {}


def load_done_rows(run_dir: Path) -> tuple[list[dict], dict]:
    """P_done rows + counts: profiling-phase, non-error profile records
    with the four core tags (unit-aware ms->s; an unrecognized unit drops
    the record visibly instead of mis-scaling it)."""
    rows = []
    counts = {"n_records": 0, "n_error": 0, "n_phase_excluded": 0, "n_incomplete": 0}
    for line in (run_dir / "profile_export.jsonl").open():
        record = json.loads(line)
        counts["n_records"] += 1
        if record.get("error"):
            counts["n_error"] += 1
            continue
        if (record.get("metadata") or {}).get("benchmark_phase") != "profiling":
            counts["n_phase_excluded"] += 1
            continue
        ttft = _metric(record, "time_to_first_token")
        lat = _metric(record, "request_latency")
        isl = _metric(record, "input_sequence_length")
        osl = _metric(record, "output_sequence_length")
        if None in (ttft.get("value"), lat.get("value"), isl.get("value"), osl.get("value")):
            counts["n_incomplete"] += 1
            continue
        gaps = _metric(record, "inter_chunk_latency").get("value")
        rows.append(
            {
                "ttft_s": _ms_to_s(ttft),
                "latency_s": _ms_to_s(lat),
                "isl": int(isl["value"]),
                "osl": int(osl["value"]),
                "chunk_gaps_ms": list(gaps) if isinstance(gaps, list) else [],
                "start_ns": (record.get("metadata") or {}).get("request_start_ns"),
                "end_ns": (record.get("metadata") or {}).get("request_end_ns"),
            }
        )
    return rows, counts


def _ms_to_s(metric: dict) -> float | None:
    """Unit-aware ms->s; None on an unrecognized unit."""
    if metric.get("value") is None:
        return None
    unit = metric.get("unit", "ms")
    if unit == "ms":
        return metric["value"] / 1000.0
    if unit == "s":
        return metric["value"]
    return None


def warmup_cut_s(run_dir: Path, done_rows: list[dict] | None = None) -> tuple[float | None, str]:
    """Warmup cut (epoch s) + its source.

    Primary: the first profiling-phase server scrape timestamp (the v1
    cut). Fallback for cells whose server_metrics_export.jsonl was
    gzip-cleaned: the first profiling request_start_ns -- verified to
    drop the exact same records on all six c16 cells and to recover the
    frozen c4 population (audit section 4).
    """
    export = run_dir / "server_metrics_export.jsonl"
    if export.exists():
        for line in export.open():
            record = json.loads(line)
            if record.get("benchmark_phase") == "profiling" and record.get("timestamp_ns"):
                return record["timestamp_ns"] / 1e9, "server_first_profiling_scrape"
    if done_rows is None:
        done_rows = load_done_rows(run_dir)[0]
    starts = [row["start_ns"] for row in done_rows if row["start_ns"] is not None]
    if starts:
        return min(starts) / 1e9, "profile_first_profiling_request_start"
    return None, "none"


def load_wait_rows(run_dir: Path, done_rows: list[dict] | None = None) -> tuple[list[dict], dict]:
    """P_wait rows + counts: schedule_eval records after the run offset
    (run_meta.schedule_eval.offset_lines slices earlier runs off a reused
    server), minus harness health-check probes, minus warmup arrivals
    (arrival before the warmup cut), valid queue_wait_s only."""
    meta = json.loads((run_dir / "run_meta.json").read_text())
    block = meta.get("schedule_eval") or {}
    offset = block.get("offset_lines")
    offset = offset if isinstance(offset, int) and offset >= 0 else 0
    lines = [json.loads(line) for line in (run_dir / "schedule_eval.jsonl").open() if line.strip()]
    post = lines[offset:]
    load_records = [r for r in post if not str(r.get("rid", "")).startswith("HEALTH_CHECK_")]
    n_health = len(post) - len(load_records)

    cut_s, cut_source = warmup_cut_s(run_dir, done_rows=done_rows)
    in_window = [
        r
        for r in load_records
        if cut_s is None
        or not (isinstance(r.get("arrival_wall_s"), (int, float)) and r["arrival_wall_s"] < cut_s)
    ]
    n_warmup = len(load_records) - len(in_window)

    rows = []
    n_no_wait = 0
    for record in in_window:
        wait = record.get("queue_wait_s")
        if not isinstance(wait, (int, float)):
            n_no_wait += 1
            continue
        rows.append(
            {
                "queue_wait_s": wait,
                "ttft_s": record.get("ttft_s"),
                "input_len": record.get("input_len") or 0,
                "matched_first": record.get("matched_prefix_tokens_first_admission") or 0,
            }
        )
    counts = {
        "n_lines": len(lines),
        "n_offset_sliced": offset,
        "n_health_check": n_health,
        "n_warmup_cut": n_warmup,
        "n_no_queue_wait": n_no_wait,
        "warmup_cut_source": cut_source,
        "warmup_cut_s": cut_s,
    }
    return rows, counts


def tok_per_s(rows: list[dict]) -> tuple[float | None, dict]:
    """Output tokens/s over the profiling window (first P_done request
    start -> last P_done request end), window declared in the output."""
    stamped = [(row["start_ns"], row["end_ns"]) for row in rows if row["start_ns"] and row["end_ns"]]
    if not stamped or sum(row["osl"] for row in rows) <= 0:
        return None, {"window_s": None}
    window_s = (max(end for _, end in stamped) - min(start for start, _ in stamped)) / 1e9
    tokens = sum(row["osl"] for row in rows)
    if window_s <= 0:
        return None, {"window_s": window_s}
    return tokens / window_s, {"window_s": window_s, "output_tokens": tokens}


def cache_hit_rate(rows: list[dict]) -> float | None:
    """Token-weighted first-attempt hit rate over P_wait (v1 definition,
    eval-log source): sum(matched_first) / sum(input_len)."""
    total_input = sum(row["input_len"] for row in rows)
    if total_input <= 0:
        return None
    return sum(row["matched_first"] for row in rows) / total_input


def _frozen_v1(run_dir: Path) -> dict:
    """Old e2_metrics.json, read-only: the cross-check block quotes its
    frozen histogram percentiles and recorded starvation threshold."""
    path = run_dir / "e2_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def compute_metrics_v2(run_dir: Path) -> dict:
    """The v2 metric set for one run dir (see module docstring)."""
    meta = json.loads((run_dir / "run_meta.json").read_text())
    done_rows, done_counts = load_done_rows(run_dir)
    wait_rows, wait_counts = load_wait_rows(run_dir, done_rows=done_rows)

    waits = [row["queue_wait_s"] for row in wait_rows]
    uncached = [row["input_len"] - row["matched_first"] for row in wait_rows]
    stretch, n_stretch_dropped, n_stretch_no_ttft = stretch_values(
        waits_s=waits,
        ttfts_s=[row["ttft_s"] for row in wait_rows],
        uncached_tokens=uncached,
    )
    gaps = itl_chunk_gaps(done_rows)
    if gaps is not None:
        itl = {"source": "per_chunk", "n_chunks": len(gaps), **stats(gaps, (50, 90, 95, 99))}
    else:
        tpot = tpot_values_ms(done_rows)
        itl = {"source": "tpot_based", "n_chunks": None, **stats(tpot, (50, 90, 95, 99))}

    tps, window = tok_per_s(done_rows)
    ttfts_ms = [row["ttft_s"] * 1000.0 for row in done_rows]

    frozen_queue = (_frozen_v1(run_dir).get("queue") or {})
    cold = [row["queue_wait_s"] for row in wait_rows if row["matched_first"] == 0]
    warm = [row["queue_wait_s"] for row in wait_rows if row["matched_first"] > 0]

    result = {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "run": {
            "policy": meta.get("policy"),
            "concurrency": meta.get("concurrency"),
            "seed": meta.get("seed"),
            "sglang_commit": meta.get("sglang_commit"),
            "timestamp": meta.get("timestamp"),
            "boost_gamma": meta.get("boost_gamma"),
            "boost_clock": meta.get("boost_clock"),
            "boost_adaptive_gamma": meta.get("boost_adaptive_gamma"),
            "boost_bin_size": meta.get("boost_bin_size"),
        },
        "populations": {
            # rid (eval log) and x_request_id (profile export) do not join:
            # every table column below names its population.
            "join": "not_joinable",
            "p_done": {
                "n": len(done_rows),
                "definition": "profile_export profiling-phase, non-error, complete core tags",
                **done_counts,
            },
            "p_wait": {
                "n": len(wait_rows),
                "definition": "schedule_eval post-offset, non-health-check, post-warmup-cut, valid queue_wait_s",
                **wait_counts,
            },
        },
        "n": {"p_done": len(done_rows), "p_wait": len(wait_rows)},
        "cache_hit": {"rate": cache_hit_rate(wait_rows), "population": "p_wait"},
        "queue_wait_s": {**stats(waits, (50, 90, 95, 99)), "population": "p_wait"},
        "ttft_ms": {**stats(ttfts_ms, (50, 95, 99)), "population": "p_done"},
        "itl_ms": {**itl, "population": "p_done"},
        "tok_per_s": {"value": tps, "window": window, "population": "p_done"},
        "jain_queue_wait": {"value": jain_queue_wait(waits), "population": "p_wait"},
        "stretch": {
            **stats(stretch, (95, 99)),
            "n_dropped_denominator": n_stretch_dropped,
            "n_dropped_no_ttft": n_stretch_no_ttft,
            "uncached_floor_tokens": STRETCH_UNCACHED_FLOOR_TOKENS,
            "population": "p_wait (uncached >= floor)",
        },
        "worst5_wait_share_pct": {"value": worst5_wait_share_pct(waits), "population": "p_wait"},
        # Below: out of the v2 table by spec; stored for cross-checking.
        "cross_check": {
            "queue_histogram_frozen_v1": {
                "p50_s": frozen_queue.get("queue_time_p50_s"),
                "p95_s": frozen_queue.get("queue_time_p95_s"),
                "p99_s": frozen_queue.get("queue_time_p99_s"),
                "n": frozen_queue.get("queue_time_n"),
            },
            "starvation_rate_at": {
                "0.2_s": _share_above(waits, 0.2),
                "10_s": _share_above(waits, 10.0),
                "recorded_threshold_s": frozen_queue.get("starvation_threshold_s"),
                "recorded_rate": frozen_queue.get("starvation_rate"),
            },
            "cold_warm_queue_wait": {
                "cold": stats(cold, (95, 99)),
                "warm": stats(warm, (95, 99)),
                "gap_p99_s": (
                    percentile(cold, 99) - percentile(warm, 99)
                    if cold and warm
                    else None
                ),
            },
        },
    }
    return _clean_rec(result)


def _share_above(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(1 for v in values if v > threshold) / len(values)


def _clean_rec(obj):
    if isinstance(obj, dict):
        return {k: _clean_rec(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_rec(v) for v in obj]
    return _clean(obj)


def write_metrics_v2(run_dir: Path) -> dict:
    """Compute + write e2_metrics_v2.json (e2_metrics.json never touched)."""
    metrics = compute_metrics_v2(run_dir)
    out_path = run_dir / "e2_metrics_v2.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--all", action="store_true", help="every run dir with run_meta.json")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "1x-rtx6000pro" / "agentx",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    if args.all:
        run_dirs = sorted(p.parent for p in args.results_root.glob("*/*/run_meta.json"))
        if not run_dirs:
            sys.exit(f"no run dirs under {args.results_root}")
        for run_dir in run_dirs:
            metrics = write_metrics_v2(run_dir)
            print(
                f"{metrics['run']['policy']}/c{metrics['run']['concurrency']} "
                f"{run_dir.name}: n_done={metrics['n']['p_done']} n_wait={metrics['n']['p_wait']} "
                f"-> {run_dir / 'e2_metrics_v2.json'}"
            )
        return

    if args.run_dir is None:
        parser.error("give a run_dir or --all")
    metrics = write_metrics_v2(args.run_dir)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(f"metrics written to {args.run_dir / 'e2_metrics_v2.json'}")


if __name__ == "__main__":
    main()
