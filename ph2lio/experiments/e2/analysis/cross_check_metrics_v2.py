"""Independent cross-check of e2_metrics_v2.json against the raw files.

Re-implements the metrics_v2 populations from the spec (docs/metrics_v2.md)
with separate code and reports deltas vs the stored e2_metrics_v2.json.
Any non-zero delta on the core metrics means the instrument or this check
diverged -- investigate before trusting either. Usage:

    python3 analysis/cross_check_metrics_v2.py <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _pctile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _time_s(record: dict, name: str) -> float | None:
    metric = (record.get("metrics") or {}).get(name) or {}
    value, unit = metric.get("value"), metric.get("unit")
    if value is None:
        return None
    return value / 1000.0 if unit == "ms" else (value if unit == "s" else None)


def _raw(record: dict, name: str) -> float | None:
    return ((record.get("metrics") or {}).get(name) or {}).get("value")


def wait_rows(run_dir: Path) -> list[dict]:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    offset = (meta.get("schedule_eval") or {}).get("offset_lines")
    offset = offset if isinstance(offset, int) and offset >= 0 else 0
    lines = [json.loads(l) for l in (run_dir / "schedule_eval.jsonl").open() if l.strip()]
    post = lines[offset:]
    nohealth = [r for r in post if not str(r.get("rid", "")).startswith("HEALTH_CHECK_")]
    cut = None
    export = run_dir / "server_metrics_export.jsonl"
    if export.exists():
        for line in export.open():
            record = json.loads(line)
            if record.get("benchmark_phase") == "profiling" and record.get("timestamp_ns"):
                cut = record["timestamp_ns"] / 1e9
                break
    inwin = [
        r
        for r in nohealth
        if cut is None
        or not (isinstance(r.get("arrival_wall_s"), (int, float)) and r["arrival_wall_s"] < cut)
    ]
    return [r for r in inwin if isinstance(r.get("queue_wait_s"), (int, float))]


def done_ttfts(run_dir: Path) -> list[float]:
    ttfts = []
    for line in (run_dir / "profile_export.jsonl").open():
        record = json.loads(line)
        if record.get("error"):
            continue
        if (record.get("metadata") or {}).get("benchmark_phase") != "profiling":
            continue
        ttft = _time_s(record, "time_to_first_token")
        lat = _time_s(record, "request_latency")
        isl = _raw(record, "input_sequence_length")
        osl = _raw(record, "output_sequence_length")
        if None in (ttft, lat, isl, osl):
            continue
        ttfts.append(ttft)
    return ttfts


def done_itl_gaps_ms(run_dir: Path) -> tuple[list[float], str]:
    """Pooled inter-chunk gaps (ms) over P_done, mirroring the spec's ITL
    row: per-chunk arrays when any record carries one, else TPOT."""
    gaps: list[float] = []
    tpot: list[float] = []
    n_gap_rows = 0
    for line in (run_dir / "profile_export.jsonl").open():
        record = json.loads(line)
        if record.get("error"):
            continue
        if (record.get("metadata") or {}).get("benchmark_phase") != "profiling":
            continue
        ttft = _time_s(record, "time_to_first_token")
        lat = _time_s(record, "request_latency")
        osl = _raw(record, "output_sequence_length")
        if None in (ttft, lat, osl):
            continue
        arr = ((record.get("metrics") or {}).get("inter_chunk_latency") or {}).get(
            "value"
        )
        if isinstance(arr, list):
            n_gap_rows += 1
            gaps.extend(arr)
        if osl >= 2:
            tpot.append((lat - ttft) * 1000.0 / (osl - 1))
    return (gaps, "per_chunk") if n_gap_rows else (tpot, "tpot_based")


def check(run_dir: Path) -> bool:
    v2 = json.loads((run_dir / "e2_metrics_v2.json").read_text())
    rows = wait_rows(run_dir)
    waits = [r["queue_wait_s"] for r in rows]
    hit = sum(r.get("matched_prefix_tokens_first_admission") or 0 for r in rows) / sum(
        r["input_len"] for r in rows
    )
    jain = (sum(waits) ** 2) / (len(waits) * sum(w * w for w in waits))
    srt = sorted(waits)
    k5 = -(-len(srt) * 5 // 100)
    worst5 = 100 * sum(srt[-k5:]) / sum(srt)
    ttfts = sorted(done_ttfts(run_dir))

    q, t = v2["queue_wait_s"], v2["ttft_ms"]
    gaps_ms, itl_source = done_itl_gaps_ms(run_dir)
    i = v2["itl_ms"]
    checks = [
        ("n_wait", len(rows), q["n"]),
        ("queue mean", sum(waits) / len(waits), q["mean"]),
        ("queue p50", _pctile(waits, 0.50), q["p50"]),
        ("queue p90", _pctile(waits, 0.90), q["p90"]),
        ("queue p95", _pctile(waits, 0.95), q["p95"]),
        ("queue p99", _pctile(waits, 0.99), q["p99"]),
        ("hit", hit, v2["cache_hit"]["rate"]),
        ("jain-qw", jain, v2["jain_queue_wait"]["value"]),
        ("worst5", worst5 / 100, v2["worst5_wait_share_pct"]["value"] / 100),
        ("n_done", len(ttfts), t["n"]),
        ("ttft mean", sum(ttfts) / len(ttfts), t["mean"] / 1000),
        ("ttft p50", _pctile(ttfts, 0.50), t["p50"] / 1000),
        ("ttft p99", _pctile(ttfts, 0.99), t["p99"] / 1000),
        ("itl p50", _pctile(gaps_ms, 0.50), i["p50"]),
        ("itl p95", _pctile(gaps_ms, 0.95), i["p95"]),
        ("itl p99", _pctile(gaps_ms, 0.99), i["p99"]),
    ]
    ok = True
    print(f"== {run_dir.name}")
    for name, mine, theirs in checks:
        delta = mine - theirs
        good = abs(delta) <= 1e-6 * max(1.0, abs(theirs))
        ok &= good
        print(f"  {name:<11} indep={mine:<12.4f} v2={theirs:<12.4f} delta={delta:+.6f} {'OK' if good else 'MISMATCH'}")
    starvers = sum(1 for w in waits if w > 1500)
    print(f"  starvers(>1500 s, indep): {starvers}")
    assert itl_source == i["source"]
    assert (
        _pctile(waits, 0.5)
        <= _pctile(waits, 0.9)
        <= _pctile(waits, 0.95)
        <= _pctile(waits, 0.99)
    )
    assert _pctile(ttfts, 0.5) <= _pctile(ttfts, 0.95) <= _pctile(ttfts, 0.99)
    assert _pctile(gaps_ms, 0.5) <= _pctile(gaps_ms, 0.95) <= _pctile(gaps_ms, 0.99)
    assert 0.0 <= jain <= 1.0 and 0.0 <= hit <= 1.0
    return ok


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    all_ok = True
    for arg in sys.argv[1:]:
        all_ok &= check(Path(arg))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
