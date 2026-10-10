"""Print the metrics-v2 tables (threshold-independent metric set).

Per-full-run counting, matching print_results_table.py: cells are grouped
by their sweep's sglang_commit stamp, one block per full run. The two
canonical RTX blocks print by default (the c4/c8 full run 2026-10-07 and
the c16 sweep 2026-10-08); --all-blocks prints every landed block.

Columns (spec docs/metrics_v2.md section 3, in order): n, cache hit (%),
queue mean/p90/p99 (s), TTFT mean/p95/p99 (ms), ITL mean/p90/p99 (ms), tok/s,
Jain, stretch p95/p99, worst-5% wait share (%). n prints both populations
(P_done = completed profile records; P_wait = eval-log admissions) -- the
eval-log rid and the profile x_request_id do not join, so every metric
names its population in e2_metrics_v2.json.

fcfs-match prints as "fcfs" (decided 2026-10-08: it is the fcfs arm --
identical ordering plus the sort-time match the bookkeeping fix
bdc67a5720 made redundant to force); plain pre-fix fcfs cells are
excluded from this table.

A side table cross-checks queue p95/p99 (eval log, exact) against the
frozen v1 histogram percentiles per cell -- the evidence that moved the
v2 table off the histogram (audit docs/metrics_v2_audit.md section 3).

Self-contained: no import from print_results_table.py (the v1 view) -- the
v1 script may change shape without breaking this pipeline.

Usage:
    python3 analysis/print_metrics_v2_table.py [--all-blocks]
        [--results-root results/1x-rtx6000pro/agentx]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Self-contained on purpose: print_results_table.py (the v1 view) is free to
# change shape without breaking the v2 pipeline; only the results-root layout
# (results_root/<policy>/<run_dir>/{run_meta,e2_metrics_v2}.json) is shared.
FINAL_POLICY_ORDER = {"lpm": 0, "boost": 1, "hrrn": 2, "fcfs-match": 3, "fcfs": 4, "dfs-weight": 5}

# The harness records boost_gamma null when the flag was left to the clock-mode
# default (1e-4 tokens / 0.1 1/s wall); float(None) would crash.
CLOCK_DEFAULT_GAMMA = {"tokens": 1e-4, "wall": 0.1}


def _boost_gamma_resolved(meta: dict) -> float:
    gamma = meta["boost_gamma"]
    if gamma is not None:
        return float(gamma)
    return CLOCK_DEFAULT_GAMMA.get(meta.get("boost_clock", "tokens"), 1e-4)


def _cell_key(meta: dict) -> tuple:
    """(policy, conc, gamma, k, clock): the comparison cell a run belongs to.
    The clock rides on boost cells so a wall arm and a tokens arm with the
    same gamma/k never merge into one cell."""
    policy = meta["policy"]
    gamma, k, clock = (
        (_boost_gamma_resolved(meta), int(meta["boost_bin_size"]), meta.get("boost_clock", "tokens"))
        if policy == "boost"
        else (None, None, None)
    )
    return (policy, int(meta["concurrency"]), gamma, k, clock)


def _group_full_runs(runs: list[dict]) -> list[dict]:
    """One entry per full run -- the cells sharing a sweep's sglang_commit
    stamp -- latest first (the sweep a driver just landed prints first)."""
    by_commit: dict[str, dict[tuple, list[dict]]] = {}
    for run in runs:
        meta = run["meta"]
        by_commit.setdefault(meta["sglang_commit"], {}).setdefault(_cell_key(meta), []).append(run)

    groups = []
    for commit, cells in by_commit.items():
        ts = [r["meta"].get("timestamp") or "" for cell_runs in cells.values() for r in cell_runs]
        first, last = min(ts)[:10], max(ts)[:10]
        groups.append(
            {
                "commit": commit,
                # Heading label only; runs are grouped by commit, never by date.
                "date": first if first == last else f"{first}..{last}",
                "last_ts": max(ts),
                "n_cells": len(cells),
                "cells": cells,
            }
        )
    groups.sort(key=lambda g: g["last_ts"], reverse=True)
    return groups

# The two canonical RTX blocks: c4/c8 full run (2026-10-07) and the c16
# validation sweep (2026-10-08). Identified by commit stamp, never by date.
CANONICAL_COMMITS = ("c634f2c859", "0083303206", "101ea1dcf3")

HEADER = (
    "| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) "
    "| queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) "
    "| ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 "
    "| worst-5% wait (%) |"
)
SEP = "|" + "---|" * 17


def load_v2_runs(results_root: Path) -> list[dict]:
    """[{meta, metrics_v2}] for every run dir holding both files."""
    runs = []
    for v2_path in sorted(results_root.glob("*/*/e2_metrics_v2.json")):
        meta_path = v2_path.parent / "run_meta.json"
        if not meta_path.exists():
            continue
        runs.append(
            {
                "meta": json.loads(meta_path.read_text()),
                "metrics_v2": json.loads(v2_path.read_text()),
            }
        )
    return runs


def _fmt(value, spec):
    if value is None:
        return "—"
    return format(value, spec)


def _pct(fraction, spec=".2f"):
    if fraction is None:
        return "—"
    return format(100.0 * fraction, spec) + "%"


def table_row(cell: tuple, run: dict) -> str:
    meta, m = run["meta"], run["metrics_v2"]
    policy = "fcfs" if cell[0] == "fcfs-match" else cell[0]
    label = f"**{policy} (baseline)**" if policy == "lpm" else policy
    queue = m["queue_wait_s"]
    ttft = m["ttft_ms"]
    itl = m["itl_ms"]
    stretch = m["stretch"]
    return " | ".join(
        (
            label,
            str(m["n"]["p_done"]),
            str(m["n"]["p_wait"]),
            _pct(m["cache_hit"]["rate"]),
            _fmt(queue["mean"], ".3f"),
            _fmt(queue["p90"], ".3f"),
            _fmt(queue["p99"], ".3f"),
            _fmt(ttft["mean"], ".2f"),
            _fmt(ttft["p95"], ".2f"),
            _fmt(ttft["p99"], ".2f"),
            _fmt(itl["mean"], ".2f"),
            _fmt(itl["p90"], ".2f"),
            _fmt(itl["p99"], ".2f"),
            _fmt(m["tok_per_s"]["value"], ".1f"),
            _fmt(m["jain_queue_wait"]["value"], ".3f"),
            _fmt(stretch["p95"], ".2f"),
            _fmt(stretch["p99"], ".2f"),
            # Already a percentage (0-100), unlike cache hit's fraction.
            _fmt(m["worst5_wait_share_pct"]["value"], ".2f") + "%",
        )
    )


def print_group(group: dict):
    print(f"\n## Full run {group['date']} — commit {group['commit'][:10]} ({group['n_cells']} cells)\n")
    conc_prev = None
    for cell in sorted(group["cells"], key=lambda c: (c[1], FINAL_POLICY_ORDER.get(c[0], 99), c[2] or 0)):
        if cell[1] != conc_prev:
            conc_prev = cell[1]
            print(f"### c{cell[1]} — concurrency {cell[1]}\n")
            print(HEADER)
            print(SEP)
        # Latest run per cell wins (a same-cell re-run supersedes).
        print(table_row(cell, group["cells"][cell][-1]))
    print()


def _rel_delta(new, old):
    if new is None or old in (None, 0):
        return "—"
    return f"{100.0 * (new - old) / old:+.1f}%"


def print_cross_check(groups: list[dict]):
    """Side table: eval-log queue p99 (exact) vs frozen v1 histogram, per cell."""
    print("\n## Cross-check: queue p99, eval log (v2) vs frozen histogram (v1)\n")
    print("| Block | Cell | n wait | p99 eval (s) | p99 hist (s) | Δp99 |")
    print("|" + "---|" * 6)
    for group in groups:
        for cell in sorted(group["cells"], key=lambda c: (c[1], FINAL_POLICY_ORDER.get(c[0], 99))):
            if cell[0] == "fcfs":
                continue
            m = group["cells"][cell][-1]["metrics_v2"]
            queue = m["queue_wait_s"]
            frozen = m["cross_check"]["queue_histogram_frozen_v1"]
            policy = "fcfs" if cell[0] == "fcfs-match" else cell[0]
            conc = cell[1]
            print(
                " | ".join(
                    (
                        group["commit"][:10],
                        f"{policy}/c{conc}",
                        str(queue["n"]),
                        _fmt(queue["p99"], ".3f"),
                        _fmt(frozen["p99_s"], ".3f"),
                        _rel_delta(queue["p99"], frozen["p99_s"]),
                    )
                )
            )
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-blocks", action="store_true", help="print every landed block, not just the two canonical ones")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "1x-rtx6000pro" / "agentx",
    )
    args = parser.parse_args()

    groups = _group_full_runs(load_v2_runs(args.results_root))
    if not groups:
        sys.exit(f"no e2_metrics_v2.json under {args.results_root}; run: python3 analysis/metrics_v2.py --all")
    # Plain fcfs cells are superseded by fcfs-match (which prints as "fcfs");
    # they stay in e2_metrics_v2.json as the fcfs-match cross-check.
    for group in groups:
        group["cells"] = {c: runs for c, runs in group["cells"].items() if c[0] != "fcfs"}
        group["n_cells"] = len(group["cells"])

    selected = groups if args.all_blocks else [g for g in groups if g["commit"][:10] in CANONICAL_COMMITS]
    missing = [c for c in CANONICAL_COMMITS if c not in {g["commit"][:10] for g in selected}]
    if missing:
        sys.exit(f"canonical block(s) {missing} have no v2 metrics")

    for group in selected:
        print_group(group)
    print_cross_check(selected)


if __name__ == "__main__":
    main()
