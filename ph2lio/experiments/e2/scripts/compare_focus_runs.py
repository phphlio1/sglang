#!/usr/bin/env python3
"""Compare N profiling runs (same config) and write a consistency report.

Usage:
    python3 compare_focus_runs.py <run_dir_baseline> <run_dir_2> [<run_dir_3> ...]
Output: docs/focus_consistency_3runs.md (next to this script's ../docs).

Each run must already have analysis/*.csv (extract_metrics_timeseries.py).
Verdicts are heuristic and stated as such: "consistent" = cùng hướng và cùng
bậc độ lớn (within ~2x, cùng dấu với corr).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focus_report_lib import analyze_run, summary  # noqa: E402

ROWS = [
    # (label, key, fmt)
    ("Request/output throughput (tok/s)", "throughput_tps", "{:.0f}"),
    ("TTFT mean (s)", "ttft_mean_s", "{:.1f}"),
    ("TTFT p99 (s)", "ttft_p99_s", "{:.1f}"),
    ("Running-req mean/rank (min–max)", None, None),  # special: range
    ("Pending backlog per-minute (min–max)", None, None),
    ("Active prefill token spread (×)", "prefill_spread_x", "{:.1f}"),
    ("KV usage mean (min–max)", None, None),
    ("KV usage spread p50", "kv_spread_p50", "{:.2f}"),
    ("#rank chạm KV 1.00", "n_rank_full", "{}"),
    ("#retract", "n_retract", "{}"),
    ("corr(logit, TTFT)", "corr_logit_ttft", "{:+.2f}"),
    ("Cache reuse engine (%)", "cache_reuse_pct", "{:.1f}"),
    ("Cached tokens/request", "cached_tokens_req", "{:.0f}"),
    ("EAGLE accept len (per-minute min–max)", None, None),
    ("ITL mean (ms)", "itl_mean_ms", "{:.1f}"),
    ("GPU util window (min–max %)", None, None),
]
RANGES = {
    "Running-req mean/rank (min–max)": ("run_mean_min", "run_mean_max", "{:.2f}–{:.2f}"),
    "Pending backlog per-minute (min–max)": ("pending_pm_min", "pending_pm_max", "{:,.0f}–{:,.0f}"),
    "KV usage mean (min–max)": ("kv_mean_min", "kv_mean_max", "{:.2f}–{:.2f}"),
    "EAGLE accept len (per-minute min–max)": ("accept_pm_min", "accept_pm_max", "{:.2f}–{:.2f}"),
    "GPU util window (min–max %)": ("gpu_win_util_min", "gpu_win_util_max", "{:.0f}–{:.0f}"),
}
# metrics where "consistent" means same order of magnitude (ratio <= 2)
FACTOR2 = {"throughput_tps", "ttft_mean_s", "ttft_p99_s", "prefill_spread_x",
           "kv_spread_p50", "cache_reuse_pct", "cached_tokens_req", "itl_mean_ms"}
SIGN = {"corr_logit_ttft"}


def fmt_v(summ, label, key, fmt):
    if label in RANGES:
        a, b, f = RANGES[label]
        return f.format(summ[a], summ[b]).replace(",", ".")
    return fmt.format(summ[key])


def verdict_row(vals, key):
    if key in SIGN:
        return "✓ cùng dấu" if all(v > 0 for v in vals) or all(v < 0 for v in vals) else "≠ khác dấu"
    if key in FACTOR2:
        lo, hi = min(vals), max(vals)
        return "✓" if (lo > 0 and hi / lo <= 2) or (lo == hi == 0) else "≠ >2×"
    return ""


def main():
    run_dirs = sys.argv[1:]
    if len(run_dirs) < 2:
        sys.exit("need >= 2 run dirs (first = baseline 3320)")
    reps = [analyze_run(d) for d in run_dirs]
    sums = [summary(r) for r in reps]
    base = sums[0]

    hdr = "| Chỉ số | " + " | ".join(f"Run {i+1}: {s['stamp']}" for i, s in enumerate(sums)) + " | Verdict |"
    sep = "|---|" + "---|" * (len(sums) + 1)
    lines = [hdr, sep]
    for label, key, fmt in ROWS:
        cells = [fmt_v(s, label, key, fmt) for s in sums]
        if key in SIGN:
            v = verdict_row([s[key] for s in sums], key)
        elif key in FACTOR2:
            v = verdict_row([s[key] for s in sums], key)
        elif label in RANGES:
            # range rows: so sánh bậc độ lớn bằng giá trị max mỗi run
            a, b, _f = RANGES[label]
            vals = [s[b] for s in sums]
            lo, hi = min(vals), max(vals)
            v = "✓" if (lo == hi) or (lo > 0 and hi / lo <= 2) else "≠ >2×"
        else:
            v = ""
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {v} |")

    # heuristic focus verdicts
    f1_ok = (all(s["pending_pm_max"] >= 10_000 for s in sums)
             and max(s["run_mean_max"] for s in sums) <= 5
             and max(s["throughput_tps"] for s in sums) / min(s["throughput_tps"] for s in sums) <= 1.5
             and max(s["ttft_p99_s"] for s in sums) / min(s["ttft_p99_s"] for s in sums) <= 2)
    f2_ok = (all(s["prefill_spread_x"] >= 2 for s in sums)
             and all(s["kv_spread_p50"] >= 0.25 for s in sums)
             and all(s["corr_logit_ttft"] > 0 for s in sums))
    health_ok = (all(s["cache_reuse_pct"] >= 80 for s in sums)
                 and all(s["accept_pm_min"] >= 1.8 for s in sums)
                 and all(s["n_retract"] <= 10 for s in sums))

    def yn(ok):
        return "**CONSISTENT**" if ok else "**KHÔNG consistent** (xem bảng)"

    doc = f"""# Consistency 3 run — dfs-weight c64 (config prod như 3320)

Trả lời câu hỏi: bottleneck thấy ở baseline 3320 có lặp lại ở các replicate
(cùng config từng tham số, seed 42, cold radix cache mỗi run) hay không?

| Run | Stamp | Run dir |
|---|---|---|
""" + "\n".join(f"| {i + 1}{' (baseline 3320)' if i == 0 else ''} | {s['stamp']} | `{r['run_dir']}` |"
                for i, (r, s) in enumerate(zip(reps, sums))) + f"""

## 1. Bảng so sánh (mọi số tính trong benchmark window, nguồn như focus report từng run)

{chr(10).join(lines)}

## 2. Verdict Focus 1 — scheduling policy (batch mỏng + prefill đói): {yn(f1_ok)}

- Batch decode mỏng mọi run (running-req mean ≤ {max(s['run_mean_max'] for s in sums):.2f}/rank).
- Backlog prefill duy trì mọi run (đỉnh per-minute {min(s['pending_pm_max'] for s in sums):,.0f}–{max(s['pending_pm_max'] for s in sums):,.0f} token).
- Throughput chênh ≤ {100 * (max(s['throughput_tps'] for s in sums) / min(s['throughput_tps'] for s in sums) - 1):.0f}%, TTFT p99 chênh ≤ {100 * (max(s['ttft_p99_s'] for s in sums) / min(s['ttft_p99_s'] for s in sums) - 1):.0f}%.

## 3. Verdict Focus 2 — routing mất cân bằng rank: {yn(f2_ok)}

- Active prefill token spread ≥ {min(s['prefill_spread_x'] for s in sums):.1f}× mọi run.
- KV usage spread p50 ≥ {min(s['kv_spread_p50'] for s in sums):.2f} mọi run.
- corr(logit, TTFT) cùng dấu dương mọi run: {', '.join(f"{s['corr_logit_ttft']:+.2f}" for s in sums)}.

## 4. Health-check: {yn(health_ok)}

- Cache reuse ≥ {min(s['cache_reuse_pct'] for s in sums):.1f}% mọi run; accept len ≥ {min(s['accept_pm_min'] for s in sums):.2f}; retract ≤ {max(s['n_retract'] for s in sums)} lần/run.

## 5. Kết luận

Verdict là heuristic (cùng hướng + cùng bậc độ lớn, ngưỡng ghi ở từng mục);
số liệu chính nằm ở bảng §1 và focus report từng run
(`docs/focus_<stamp>_routing_scheduling.md`).
"""
    out = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "docs", "focus_consistency_3runs.md"))
    with open(out, "w") as fh:
        fh.write(doc)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
