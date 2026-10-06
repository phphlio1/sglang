#!/usr/bin/env python3
"""Task-4 entry selector for the targeted starvation regime at c8.

The agentx pool is a two-level structure in scan order: 393 long multi-turn
root chains first, then 9209 short subagent child chains (1-3 turns, ~0 s
think-times). AIPerf draws root trajectories sequentially (wrap at pool end;
each wrap is cache-cold because the scenario force-enables first-turn-prefix
cache-bust) and spawns a trajectory's child chains when the replay clock
reaches their recorded branch offsets.

`--num-dataset-entries N` caps the pool at the first N trajectories (root
chain + that trace's children). The scenario forbids every delay-compression
flag, so the only arrival-density lever left is the trajectory resume window
(`--trajectory-start-max-ratio`): early turns carry seconds-scale
think-times while late turns carry minutes-scale agent-work gaps, so
resuming in the first quarter makes c8 lanes fire densely.

This script reports the pool composition per candidate N (roots, children,
reachable turns under the resume window) and recommends the smallest N that
still yields a dense, child-rich pool. Reachable-turn counts are upper
bounds: they walk recorded delays only and ignore serving time, warmup
requests, and child-spawn traffic (child turns arrive on top); the smoke run
calibrates the real density.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_MANIFEST = Path(
    "~/.cache/aiperf/dataset_mmap/6a0aaf72b4733892e3ba3531440832b2/manifest.json"
)


def load_conversations(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    raw = manifest.get("dataset_metadata_json")
    meta = json.loads(raw) if isinstance(raw, str) else raw
    convs = meta.get("conversations")
    if not convs:
        raise SystemExit(f"no conversations in {manifest_path}")
    return convs


def split_pool(convs: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    roots, by_parent = [], {}
    for c in convs:
        parent = c.get("parent_conversation_id")
        if parent:
            by_parent.setdefault(parent, []).append(c)
        else:
            roots.append(c)
    return roots, by_parent


def delay_s(turn: dict) -> float:
    return float(turn.get("delay_ms") or 0) / 1000.0


def window_start_index(turns: list[dict], max_ratio: float) -> int:
    # Resume at turn k (0-based) -> profiling continues from index k.
    return max(1, math.ceil(max_ratio * len(turns)))


def window_pace_s(turns: list[dict], max_ratio: float) -> float:
    k = window_start_index(turns, max_ratio)
    ds = [delay_s(t) for t in turns[:k]]
    return sum(ds) / len(ds) if ds else 0.0


def reachable_turns(turns: list[dict], start_index: int, budget_s: float) -> int:
    cum, count = 0.0, 0
    for t in turns[start_index:]:
        cum += delay_s(t)
        if cum > budget_s:
            break
        count += 1
    return count


def earliest_branch_offset_s(root: dict, kids: list[dict]) -> float | None:
    if not kids:
        return None
    base = root["turns"][0].get("timestamp_ms") or 0
    return min(((k["turns"][0].get("timestamp_ms") or 0) - base) / 1000.0 for k in kids)


def trajectory_row(
    pos: int, root: dict, kids: list[dict], max_ratio: float, budget_s: float
) -> dict:
    turns = root["turns"]
    k = window_start_index(turns, max_ratio)
    return {
        "pos": pos,
        "turns": len(turns),
        "win_turns": k,
        "win_pace_s": window_pace_s(turns, max_ratio),
        "reach_from_win": reachable_turns(turns, k, budget_s),
        "reach_from_0": reachable_turns(turns, 0, budget_s),
        "kids": len(kids),
        "kid_turns": sum(len(k["turns"]) for k in kids),
        "branch_off_s": earliest_branch_offset_s(root, kids),
    }


def print_trajectories(rows: list[dict]) -> None:
    print(
        "Per-trajectory composition (scan order; window = first "
        "ceil(max_ratio * turns) turns):"
    )
    hdr = (
        f"{'pos':>3} {'turns':>5} {'win':>4} {'pace_s':>8} "
        f"{'reach@win':>9} {'reach@0':>7} {'kids':>4} {'kidT':>5} {'branch_s':>9}"
    )
    print(hdr)
    for r in rows:
        off = "-" if r["branch_off_s"] is None else f"{r['branch_off_s']:.0f}"
        print(
            f"{r['pos']:>3} {r['turns']:>5} {r['win_turns']:>4} "
            f"{r['win_pace_s']:>8.1f} {r['reach_from_win']:>9} "
            f"{r['reach_from_0']:>7} {r['kids']:>4} {r['kid_turns']:>5} {off:>9}"
        )


def pool_summary(n: int, rows: list[dict]) -> dict:
    seg = rows[:n]
    return {
        "n": n,
        "root_turns": sum(r["turns"] for r in seg),
        "mean_root_turns": sum(r["turns"] for r in seg) / len(seg),
        "reach_from_win": sum(r["reach_from_win"] for r in seg),
        "reach_from_0": sum(r["reach_from_0"] for r in seg),
        "kids": sum(r["kids"] for r in seg),
        "kid_turns": sum(r["kid_turns"] for r in seg),
    }


def print_pools(pools: list[dict]) -> None:
    print("\nCandidate pools (reachable = upper bound, recorded delays only):")
    print(
        f"{'N':>3} {'rootT':>6} {'meanT':>6} {'reach@win':>9} {'reach@0':>8} "
        f"{'kids':>5} {'kidT':>6} {'poolReq':>7}"
    )
    for p in pools:
        print(
            f"{p['n']:>3} {p['root_turns']:>6} {p['mean_root_turns']:>6.1f} "
            f"{p['reach_from_win']:>9} {p['reach_from_0']:>8} "
            f"{p['kids']:>5} {p['kid_turns']:>6} "
            f"{p['root_turns'] + p['kid_turns']:>7}"
        )


def recommend(pools: list[dict]) -> dict | None:
    # Smallest pool that is still dense (>= MIN_REACH root-continuation turns
    # reachable under the resume window) and child-rich (>= MIN_KIDS short
    # chains to interleave cold new-session starts).
    min_reach, min_kids = 200, 50
    for p in pools:
        if p["reach_from_win"] >= min_reach and p["kids"] >= min_kids:
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--start-max-ratio", type=float, default=0.25)
    parser.add_argument(
        "--top", type=int, default=16, help="per-trajectory rows to print"
    )
    parser.add_argument("--candidates", type=int, nargs="+", default=[8, 16, 32])
    args = parser.parse_args()

    convs = load_conversations(args.manifest.expanduser())
    roots, by_parent = split_pool(convs)
    top = max(args.top, max(args.candidates))
    rows = [
        trajectory_row(
            pos,
            root,
            by_parent.get(root["conversation_id"], []),
            args.start_max_ratio,
            args.duration,
        )
        for pos, root in enumerate(roots[:top], 1)
    ]
    print_trajectories(rows[: args.top])

    pools = [pool_summary(n, rows) for n in args.candidates]
    print_pools(pools)

    pick = recommend(pools)
    if pick is None:
        print(
            "\nNo candidate satisfies the density/child rule; "
            "widen the candidates or the resume window."
        )
        return
    print(
        f"\nN* = {pick['n']} (smallest candidate with "
        f">= 200 reachable root turns and >= 50 child chains)"
    )
    print(
        f"regime: {pick['n']} long multi-turn roots (mean "
        f"{pick['mean_root_turns']:.0f} turns) + {pick['kids']} short child "
        f"chains; wraps re-draw exhausted traces cache-cold"
    )
    print(f"\nrun_e2.sh env for the targeted regime at c{args.concurrency}:")
    print(
        f"  OUTPUT_DIR=results/agentx_targeted "
        f"NUM_DATASET_ENTRIES={pick['n']} "
        f"TRAJECTORY_START_MIN_RATIO=0.0 "
        f"TRAJECTORY_START_MAX_RATIO={args.start_max_ratio} "
        f"CONCURRENCY={args.concurrency} SEED=42"
    )


if __name__ == "__main__":
    main()
