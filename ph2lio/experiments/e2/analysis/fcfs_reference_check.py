#!/usr/bin/env python3
"""Task-1a decisive checks on the fcfs reference arm (no runs needed).

Resolves the matrix's fcfs "internal tension" from existing logs:

1. TTFT-vs-input fit per arm: under zero reuse TTFT ~ input / prefill_rate,
   tightly monotone; under reuse many points fall far below that line. OLS
   fit TTFT = a + b*input reports R^2 and the implied instantaneous prefill
   rate 1/b; compare across arms and concurrencies.
2. matched_prefix_tokens_first_admission distribution per arm (the sort-time
   bookkeeping fcfs skips) and uncached work u = input + output - matched
   per cold/warm class.
3. Chunk-continuation self-hit check at c4: chunked prefill re-reports each
   finished chunk as cached tokens on the next chunk's admission, so a cold
   request of L tokens at chunk size c generates ~c*n*(n-1)/2 cached-token
   volume (n = ceil(L/c)) with NO cross-request reuse. We predict that ramp
   per request from client input lengths (+ admission matches for arms that
   log them), sum it over the run, and compare against the server log's
   observed #cached-token aggregate sliced to the run window:
     observed ~ predicted  -> self-hits only (no real reuse signal)
     observed >> predicted -> cross-request reuse beyond the ramp
   The ramp is quadratic in input length, so it dominates the aggregate for
   this workload's ~57k-token inputs; the delta is the honest reuse volume.

Writes fcfs_reference_check.json next to the results root and prints the
report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

from agentx_metrics import (
    extract_per_request,
    load_profile_records,
    load_run_meta,
    load_schedule_eval_records,
)
from starvation import _percentile, ols_slope

CHUNK_SIZE = 8192  # --chunked-prefill-size in the pinned serving config

PREFILL_LINE_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Prefill batch.*?"
    r"#new-token: (\d+), #cached-token: (\d+)"
)


def profiling_window_s(run_dir: Path) -> tuple[float, float] | None:
    """[first, last] PROFILING-phase scrape epochs from the metrics export."""
    path = run_dir / "server_metrics_export.jsonl"
    if not path.exists():
        return None
    first = last = None
    for line in path.open():
        rec = json.loads(line)
        if rec.get("benchmark_phase") != "profiling" or not rec.get("timestamp_ns"):
            continue
        t = rec["timestamp_ns"] / 1e9
        first = t if first is None else min(first, t)
        last = t if last is None else max(last, t)
    return (first, last) if first is not None else None


def parse_server_prefill_lines(log_path: Path) -> list[tuple[float, int, int]]:
    """(epoch_s, new_tokens, cached_tokens) per Prefill-batch log line."""
    rows = []
    for line in log_path.open(errors="replace"):
        m = PREFILL_LINE_RE.match(line)
        if not m:
            continue
        ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        rows.append((ts, int(m.group(2)), int(m.group(3))))
    return rows


def find_server_log(policy_dir: Path, window: tuple[float, float]) -> Path | None:
    """The policy's server log whose timestamp range covers the window start."""
    candidates = []
    for path in sorted(policy_dir.glob("server_*.log")):
        rows = parse_server_prefill_lines(path)
        if not rows:
            continue
        lo, hi = rows[0][0], rows[-1][0]
        candidates.append((lo, hi, path))
    # Prefer the log whose range contains the window; ties (one long-lived
    # server reused across concurrencies) resolve to the latest start.
    containing = [c for c in candidates if c[0] - 5 <= window[0] <= c[1] + 300]
    pool = containing or candidates
    return max(pool, key=lambda c: c[0])[2] if pool else None


def observed_cached_in_window(
    log_path: Path, window: tuple[float, float]
) -> dict | None:
    rows = parse_server_prefill_lines(log_path)
    in_window = [r for r in rows if window[0] - 5 <= r[0] <= window[1] + 300]
    if not in_window:
        return None
    return {
        "n_prefill_batches": len(in_window),
        "observed_cached_tokens": sum(r[2] for r in in_window),
        "observed_new_tokens": sum(r[1] for r in in_window),
        "server_log": log_path.name,
    }


def chunk_ramp_cached(input_len: int, matched: int, chunk: int) -> int:
    """Cached-token volume one request's chunked prefill generates.

    With an admission match of m tokens and n = ceil((L - m)/c) chunks, each
    chunk reports its accumulated match: m, m+c, m+2c, ... so the total is
    n*m + c*n*(n-1)/2. The c*n*(n-1)/2 term is the pure self-hit ramp
    (policy-independent); n*m is the cross-request match counted once per
    chunk that re-reports it.
    """
    uncached = max(0, input_len - matched)
    n = max(1, math.ceil(uncached / chunk))
    return int(n * matched + chunk * n * (n - 1) / 2)


def dist_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "max": ordered[-1],
    }


def ttft_input_fit(run_dir: Path) -> dict | None:
    """OLS fit TTFT_s = a + b*input_len over profiling-phase client records."""
    per_req = extract_per_request(load_profile_records(run_dir))
    pairs = [
        (isl, ttft) for isl, ttft in zip(per_req["isl"], per_req["ttft_s"]) if isl > 0
    ]
    if len(pairs) < 8:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    slope, r2 = ols_slope(xs, ys)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    intercept = mean_y - slope * mean_x
    long_inputs = [t for i, t in pairs if i > 50_000]
    return {
        "n": len(pairs),
        "slope_s_per_token": slope,
        "intercept_s": intercept,
        "r_squared": r2,
        "implied_prefill_rate_tps": (1.0 / slope) if slope and slope > 0 else None,
        "n_input_gt_50k": len(long_inputs),
        "median_ttft_input_gt_50k_s": (
            _percentile(sorted(long_inputs), 50) if long_inputs else None
        ),
        "frac_input_gt_50k_ttft_lt_2s": (
            sum(1 for t in long_inputs if t < 2.0) / len(long_inputs)
            if long_inputs
            else None
        ),
    }


def admission_match_dist(run_dir: Path) -> dict | None:
    """matched/u distribution per class from the schedule-eval log."""
    records, _ = load_schedule_eval_records(run_dir)
    if not records:
        return None
    matched = [
        r.get("matched_prefix_tokens_first_admission") or 0
        for r in records
        if isinstance(r.get("input_len"), int)
    ]
    inputs = [r.get("input_len") or 0 for r in records]
    outputs = [r.get("output_len") or 0 for r in records]
    u = [i + o - m for i, o, m in zip(inputs, outputs, matched)]
    warm = [uv for uv, m in zip(u, matched) if m > 0]
    cold = [uv for uv, m in zip(u, matched) if m == 0]
    return {
        "n_records": len(records),
        "matched_dist": dist_stats([float(m) for m in matched]),
        "u_all": dist_stats([float(v) for v in u]),
        "u_cold": dist_stats([float(v) for v in cold]),
        "u_warm": dist_stats([float(v) for v in warm]),
    }


def self_hit_check(run_dir: Path, policy_dir: Path) -> dict | None:
    """Observed vs predicted cached-token volume for one run window."""
    window = profiling_window_s(run_dir)
    if window is None:
        return None
    log_path = find_server_log(policy_dir, window)
    if log_path is None:
        return None
    observed = observed_cached_in_window(log_path, window)
    if observed is None:
        return None

    per_req = extract_per_request(load_profile_records(run_dir))
    records, _ = load_schedule_eval_records(run_dir)
    # Admission matches per client record: the eval log is the only per-request
    # source; join by order is unsafe, so predict from input lengths with
    # matched=0 when the arm does not log matches (fcfs) and with the run's
    # matched multiset otherwise (ramp depends on matched only through u).
    matched_by_input: dict[int, int] = {}
    if records:
        for r in records:
            if isinstance(r.get("input_len"), int):
                matched_by_input[r["input_len"]] = (
                    r.get("matched_prefix_tokens_first_admission") or 0
                )
    predicted = 0
    n_with_match = 0
    for isl in per_req["isl"]:
        matched = matched_by_input.get(isl, 0)
        n_with_match += matched > 0
        predicted += chunk_ramp_cached(isl, matched, CHUNK_SIZE)

    observed_cached = observed["observed_cached_tokens"]
    return {
        **observed,
        "window_start_s": window[0],
        "chunk_size": CHUNK_SIZE,
        "n_requests": len(per_req["isl"]),
        "n_requests_with_logged_match": n_with_match,
        "predicted_self_hit_tokens": predicted,
        "observed_minus_predicted": observed_cached - predicted,
        "observed_over_predicted": (
            observed_cached / predicted if predicted > 0 else None
        ),
    }


def discover_runs(results_root: Path) -> list[tuple[str, Path]]:
    """(policy, run_dir) for every run dir with run_meta.json, latest first."""
    runs = []
    for policy_dir in sorted(results_root.iterdir()):
        if not policy_dir.is_dir() or policy_dir.name in ("smoke",):
            continue
        for run_dir in sorted(policy_dir.glob("c*_*")):
            if (run_dir / "run_meta.json").exists():
                runs.append((policy_dir.name, run_dir))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_root",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "results" / "agentx",
    )
    args = parser.parse_args()

    report: dict = {"results_root": str(args.results_root), "runs": []}
    for policy, run_dir in discover_runs(args.results_root):
        meta = load_run_meta(run_dir)
        entry = {
            "policy": policy,
            "run_dir": run_dir.name,
            "concurrency": meta.get("concurrency"),
            "gamma": meta.get("boost_gamma") if policy == "boost" else None,
        }
        entry["ttft_input_fit"] = ttft_input_fit(run_dir)
        entry["admission_match_dist"] = admission_match_dist(run_dir)
        entry["self_hit_check"] = self_hit_check(run_dir, run_dir.parent)
        report["runs"].append(entry)

        fit = entry["ttft_input_fit"] or {}
        dist = entry["admission_match_dist"] or {}
        sh = entry["self_hit_check"] or {}
        print(f"\n=== {policy} c{entry['concurrency']} ({run_dir.name}) ===")
        if fit:
            print(
                f"  TTFT fit: n={fit['n']}  slope={fit['slope_s_per_token']:.2e} s/tok"
                f"  R2={fit['r_squared']:.3f}"
                f"  implied rate={fit['implied_prefill_rate_tps'] and format(fit['implied_prefill_rate_tps'], '.0f') or 'n/a'} tok/s"
            )
            print(
                f"  input>50k: n={fit['n_input_gt_50k']}"
                f"  median TTFT={fit['median_ttft_input_gt_50k_s'] and format(fit['median_ttft_input_gt_50k_s'], '.2f') or 'n/a'} s"
                f"  frac TTFT<2s={fit['frac_input_gt_50k_ttft_lt_2s'] and format(fit['frac_input_gt_50k_ttft_lt_2s'], '.2f') or 'n/a'}"
            )
        if dist:
            md = dist["matched_dist"]
            print(
                f"  admission matched: n={md['n']}  mean={md['mean']:.0f}"
                f"  median={md['median']:.0f}  p95={md['p95']:.0f}"
            )
            for cls in ("u_cold", "u_warm"):
                cs = dist[cls]
                if cs["n"]:
                    print(
                        f"  {cls}: n={cs['n']}  mean={cs['mean']:.0f}"
                        f"  median={cs['median']:.0f}  p95={cs['p95']:.0f}"
                    )
        if sh:
            print(
                f"  self-hit: observed={sh['observed_cached_tokens']:,}"
                f"  predicted={sh['predicted_self_hit_tokens']:,}"
                f"  ratio={sh['observed_over_predicted'] and format(sh['observed_over_predicted'], '.2f') or 'n/a'}"
                f"  ({sh['n_prefill_batches']} prefill batches, {sh['server_log']})"
            )

    out_path = args.results_root / "fcfs_reference_check.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
