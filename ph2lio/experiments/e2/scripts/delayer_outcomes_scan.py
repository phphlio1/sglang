#!/usr/bin/env python3
"""Quét server_metrics_export.jsonl lấy đủ 3 metric sglang:prefill_delayer_*
(outcomes counter + 2 histogram) KÈM ĐẦY ĐỦ label, tính delta trong benchmark
window của từng run (phase `profiling`, T0 = min request_start_ns, +1800 s).
Metric này đã có trong export nhưng extract_metrics_timeseries.py bỏ label
(keep-set chỉ giữ value per dp_rank) nên chưa phân tích được bao giờ.

Chạy trên node tính toán (file 4,4 GB/run, không chạy trên login node):
  srun -p hardware-hcm -w hgx-h200-02 --cpus-per-task=8 --mem=16G \
    --time=00:45:00 --immediate=180 python3 -I delayer_outcomes_scan.py <run_dir>...

Output: /shared/loipp/logs/delayer_outcomes_<stamp>.tsv (per name+label,
delta counter / delta bucket histogram) + tóm tắt ra stdout.
"""
import json
import os
import sys
from collections import defaultdict


def bench_window(run_dir):
    """T0 = min request_start_ns các record phase 'profiling' (như
    focus_report_lib.py:97-104); window = [T0, T0+1800]."""
    phases = defaultdict(int)
    starts = []
    with open(os.path.join(run_dir, "profile_export.jsonl")) as f:
        for line in f:
            try:
                md = json.loads(line).get("metadata", {})
            except Exception:
                continue
            ph = md.get("benchmark_phase")
            phases[ph] += 1
            if ph == "profiling" and md.get("request_start_ns"):
                starts.append(md["request_start_ns"] / 1e9)
    t0 = min(starts)
    return t0, t0 + 1800.0, dict(phases)


def scan(run_dir, t0, t1):
    """Stream file, chỉ json.loads các dòng có 'prefill_delayer' (bộ lọc
    substring trước khi parse). Giữ snapshot đầu/cuối trong window cho từng
    (name, label-signature)."""
    acc = {}  # (name, sig) -> [first_ts, first_snap, last_ts, last_snap]
    n = 0
    with open(os.path.join(run_dir, "server_metrics_export.jsonl")) as f:
        for line in f:
            if "prefill_delayer" not in line:
                continue
            n += 1
            rec = json.loads(line)
            if ":8081" not in rec["endpoint_url"].rsplit("/", 1)[0]:
                continue
            ts = rec["timestamp_ns"] / 1e9
            if not (t0 <= ts <= t1):
                continue
            for name, samples in rec.get("metrics", {}).items():
                if "prefill_delayer" not in name:
                    continue
                for s in samples:
                    lab = s.get("labels") or {}
                    sig = "|".join(f"{k}={lab[k]}" for k in sorted(lab))
                    if "buckets" in s:  # histogram: buckets + sum + count
                        snap = (dict(s["buckets"]), s.get("sum", 0.0), s.get("count", 0))
                    else:  # counter: value
                        snap = s.get("value", 0.0)
                    ent = acc.get((name, sig))
                    if ent is None:
                        acc[(name, sig)] = [ts, snap, ts, snap]
                    else:
                        ent[2], ent[3] = ts, snap
    return acc, n


def summarize(run_dir):
    t0, t1, phases = bench_window(run_dir)
    acc, n = scan(run_dir, t0, t1)
    stamp = run_dir.rstrip("/").rsplit("_", 1)[-1]
    out = f"/shared/loipp/logs/delayer_outcomes_{stamp}.tsv"
    with open(out, "w") as g:
        g.write("name\tlabels\tfirst_ts\tlast_ts\tdelta\n")
        for (name, sig), (ts0, s0, ts1, s1) in sorted(acc.items()):
            if isinstance(s0, tuple):
                b0, u0, c0 = s0
                b1, u1, c1 = s1
                keys = sorted(set(b0) | set(b1), key=float)
                db = {k: b1.get(k, 0) - b0.get(k, 0) for k in keys}
                delta = f"count={c1 - c0} sum={u1 - u0:.1f} buckets={db}"
            else:
                delta = f"{s1 - s0:.0f}"
            g.write(f"{name}\t{sig}\t{ts0:.0f}\t{ts1:.0f}\t{delta}\n")

    print(f"== {run_dir}")
    print(f"  window {t0:.0f}..{t1:.0f} phases={phases} matched_lines={n}")

    # Counter: gom delta theo (input_estimation, output_reason, allow, exec)
    tot = defaultdict(float)
    for (name, sig), (_, s0, _, s1) in acc.items():
        if name.endswith("outcomes") and isinstance(s0, float):
            lab = dict(x.split("=", 1) for x in sig.split("|"))
            k = (lab.get("input_estimation"), lab.get("output_reason"),
                 lab.get("output_allow"), lab.get("actual_execution"))
            tot[k] += s1 - s0
    for k in sorted(tot, key=lambda k: -tot[k]):
        print(f"  outcomes est={k[0]} reason={k[1]!r} allow={k[2]} exec={k[3]}: {tot[k]:.0f}")

    # Histogram: gom delta bucket/sum/count qua các rank
    for hn in ("sglang:prefill_delayer_wait_forward_passes",
               "sglang:prefill_delayer_wait_seconds"):
        db, du, dc = defaultdict(float), 0.0, 0
        found = False
        for (name, _), (_, s0, _, s1) in acc.items():
            if name != hn or not isinstance(s0, tuple):
                continue
            found = True
            b0, u0, c0 = s0
            b1, u1, c1 = s1
            for kk in set(b0) | set(b1):
                db[kk] += b1.get(kk, 0) - b0.get(kk, 0)
            du += u1 - u0
            dc += c1 - c0
        if found:
            bs = {k: db[k] for k in sorted(db, key=float)}
            print(f"  {hn}: delta_count={dc:.0f} delta_sum={du:.1f}")
            print(f"    buckets={bs}")
    sys.stdout.flush()


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        summarize(rd)
