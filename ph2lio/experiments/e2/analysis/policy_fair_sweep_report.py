#!/usr/bin/env python3
"""Fair-sweep aggregator: 7 arms x 3 seeds (policy_fair_sweep.sh).

Scans the RTX results root for campaign cells (campaign commits, c16,
pin=0, seed in the campaign set), maps each to an arm, pulls the shared
metrics from e2_metrics.json, computes the threshold-free tail metrics
(stretch p99, worst-5% wait share, cache hit) from schedule_eval.jsonl
with the same definitions as analysis/metrics_v2.py, and prints:

  1. per-arm mean +/- sd across seeds (works on partial campaigns)
  2. per-seed paired ranks by TTFT p99 (same seed = same request mix)

Usage: python3 policy_fair_sweep_report.py [--results-root DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

# Campaign identity (policy_fair_sweep.sh, 2026-10-19; trimmed 5 -> 3
# seeds on 2026-10-20 before seeds 15767/27781 ran).
SEEDS = [77188, 42302, 94815]
# run_meta records the repo HEAD at server start, and the campaign spans
# five commits (sweep launch, env fix, seed trim, root-cause docs, aggregator
# fix). All are scripts/docs only -- the policy code is identical since
# 0229372590. Later commits (b21cde8cce+) carry the leader-relative dedup
# fix and must NOT join this no-fix campaign; those cells are the J-grid.
COMMIT_PREFIXES = [
    "736733d495",
    "0623366f0e",
    "dfe209bcbe",
    "bec18cc62",
]
ARM_ORDER = ["fcfs", "lpm", "hrrn", "dfs-weight", "boost-g5.0", "boost-g15.0", "boost-ada-j60"]
STRETCH_UNCACHED_FLOOR_TOKENS = 256


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def arm_of(meta: dict) -> str | None:
    if meta.get("concurrency") != 16:
        return None
    if meta.get("seed") not in SEEDS:
        return None
    pin = meta.get("in_batch_prefix_caching_max_pin_s")
    if pin is None or float(pin) != 0.0:
        return None
    commit = meta.get("sglang_commit") or ""
    if not any(commit.startswith(p) for p in COMMIT_PREFIXES):
        return None
    policy = meta.get("policy")
    if policy != "boost":
        return policy if policy in ARM_ORDER else None
    if meta.get("boost_adaptive_gamma"):
        return f"boost-ada-j{int(meta.get('boost_adaptive_max_jump_s'))}"
    return f"boost-g{meta.get('boost_gamma')}"


def eval_records(path: str) -> list[dict]:
    """Real, warmup-excluded eval records (health checks and the first 60 s
    of arrivals dropped -- same rule the audit's offline TTFT recompute used,
    which matched e2_metrics within 0.95-0.99)."""
    real = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec["rid"].startswith("HEALTH_CHECK"):
                real.append(rec)
    if not real:
        return []
    t0 = min(r["arrival_wall_s"] for r in real)
    return [r for r in real if r["arrival_wall_s"] - t0 > 60.0]


def tail_metrics(records: list[dict]) -> dict:
    # Grace-period teardown leaves ttft_s null on requests that never
    # decoded a token; the tail metrics cover served requests only.
    records = [
        r
        for r in records
        if r["ttft_s"] is not None and r["queue_wait_s"] is not None
    ]
    waits = [r["queue_wait_s"] for r in records]
    ttfts = [r["ttft_s"] for r in records]
    uncached = [
        r["input_len"] - r["matched_prefix_tokens_first_admission"] for r in records
    ]
    stretch = []
    for wait, ttft, u in zip(waits, ttfts, uncached):
        if u < STRETCH_UNCACHED_FLOOR_TOKENS:
            continue
        denom = ttft - wait
        if denom > 0:
            stretch.append(wait / denom)
    n = len(waits)
    worst5 = None
    if n and sum(waits) > 0:
        k = math.ceil(0.05 * n)
        worst5 = 100.0 * sum(sorted(waits, reverse=True)[:k]) / sum(waits)
    in_tok = sum(r["input_len"] for r in records)
    matched = sum(r["matched_prefix_tokens_first_admission"] for r in records)
    return {
        "stretch_p99": percentile(stretch, 99) if stretch else None,
        "worst5_wait_pct": worst5,
        "cache_hit_pct": 100.0 * matched / in_tok if in_tok else None,
    }


def load_cell(run_dir: str) -> dict | None:
    with open(os.path.join(run_dir, "run_meta.json")) as fh:
        meta = json.load(fh)
    arm = arm_of(meta)
    if arm is None:
        return None
    cell = {"arm": arm, "seed": meta["seed"], "dir": os.path.basename(run_dir)}
    mf = os.path.join(run_dir, "e2_metrics.json")
    ef = os.path.join(run_dir, "schedule_eval.jsonl")
    if not os.path.exists(mf):
        return None  # cell not finished
    with open(mf) as fh:
        m = json.load(fh)
    ttft, queue = m["ttft"], m["queue"]
    cell.update(
        n_done=ttft["n"],
        ttft_mean_ms=ttft["mean_ms"],
        ttft_p95_ms=ttft["p95_ms"],
        ttft_p99_ms=ttft["p99_ms"],
        queue_mean_s=queue["queue_time_mean_s"],
        queue_p99_s=queue["queue_time_p99_s"],
        tok_s=m["throughput"]["output_token_throughput_tps"],
        jain=m["fairness"]["ttft_fairness_jain"],
    )
    if os.path.exists(ef):
        cell.update(tail_metrics(eval_records(ef)))
    return cell


def fmt_ms(v: float) -> str:
    return f"{v / 1000.0:.1f}s" if v is not None else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    default_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "results", "1x-rtx6000pro", "agentx",
    )
    ap.add_argument("--results-root", default=default_root)
    args = ap.parse_args()

    cells = []
    for pol in sorted(os.listdir(args.results_root)):
        pdir = os.path.join(args.results_root, pol)
        if not os.path.isdir(pdir):
            continue
        for d in sorted(os.listdir(pdir)):
            rd = os.path.join(pdir, d)
            if os.path.exists(os.path.join(rd, "run_meta.json")):
                cell = load_cell(rd)
                if cell:
                    cells.append(cell)

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_arm[c["arm"]].append(c)
    arms = [a for a in ARM_ORDER if a in by_arm]

    if not cells:
        print("no campaign cells found yet (seed/commit/pin filter)")
        return 0

    metrics = ["n_done", "ttft_p99_ms", "ttft_p95_ms", "ttft_mean_ms",
               "queue_p99_s", "queue_mean_s", "tok_s", "jain",
               "stretch_p99", "worst5_wait_pct", "cache_hit_pct"]

    print(f"cells found: {len(cells)}  arms: {len(arms)}  "
          f"seeds seen: {sorted({c['seed'] for c in cells})}\n")

    # Per-seed matrix by TTFT p99 (paired view).
    print("== TTFT p99 per seed (ms) -- paired: same seed = same request mix ==")
    hdr = f"{'arm':<15}" + "".join(f"{s:>9}" for s in SEEDS if s in {c['seed'] for c in cells})
    print(hdr)
    for arm in arms:
        row = by_arm[arm]
        per_seed = {c["seed"]: c["ttft_p99_ms"] for c in row}
        print(f"{arm:<15}" + "".join(
            f"{per_seed[s]:>9.0f}" if s in per_seed else f"{'-':>9}"
            for s in SEEDS if s in {c['seed'] for c in cells}))
    print()

    # Mean rank by TTFT p99 (1 = best), paired across seeds.
    seeds_seen = sorted({c["seed"] for c in cells})
    rank_sum = defaultdict(float)
    rank_n = defaultdict(int)
    for s in seeds_seen:
        at_seed = [(c["arm"], c["ttft_p99_ms"]) for c in cells if c["seed"] == s]
        at_seed.sort(key=lambda x: x[1])
        for rank, (arm, _) in enumerate(at_seed, start=1):
            rank_sum[arm] += rank
            rank_n[arm] += 1
    print("== mean rank by TTFT p99 (1 = best; paired across seeds) ==")
    for arm in sorted(arms, key=lambda a: rank_sum[a] / max(rank_n[a], 1)):
        print(f"  {arm:<15} mean_rank={rank_sum[arm] / rank_n[arm]:.2f}  (n={rank_n[arm]})")
    print()

    # mean +/- sd table.
    print("== mean +/- sd across seeds ==")
    print(f"{'arm':<15} {'seeds':>5} " + " ".join(f"{m[:11]:>13}" for m in metrics))
    for arm in arms:
        row = by_arm[arm]
        parts = [f"{arm:<15} {len(row):>5} "]
        for mname in metrics:
            vals = [c[mname] for c in row if c.get(mname) is not None]
            if not vals:
                parts.append(f"{'-':>13}")
                continue
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
            scale = 1000.0 if mname == "ttft_p99_ms" else 1.0
            parts.append(f"{mu / scale:>6.1f}±{sd / scale:<6.1f}")
        print(" ".join(parts))
    print("\n(ttft_p99 in s; stretch_p99 is x-multipier; others raw)")

    # Per-arm detail lines for traceability.
    print("== cells ==")
    for arm in arms:
        for c in sorted(by_arm[arm], key=lambda x: x["seed"]):
            print(f"  {arm:<15} seed={c['seed']:<6} {c['dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
