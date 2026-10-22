"""Build the final sweep tables (one per seed) from e2_metrics_v2.json.

Usage:

    python3 analysis/final_tables.py <seed> [--allow-partial]

Reads analysis/final_tables_manifest.json: {"<seed>": {"<arm>": [run_dir, ...]}}
(run_dir relative to the experiment root). Every listed dir must already carry
a verified e2_metrics_v2.json (metrics_v2 + cross_check 16/16) and a run_meta
seed matching the manifest key; anything else aborts before printing unless
--allow-partial (interim views; the final file must run without it). Cells are
the mean over the listed runs -- no hand-copied numbers.

One wide table per seed: policy rows, metric columns. Sources
(docs/metrics_v2.md): n/queue/hit/Jain/worst-5 come from P_wait, TTFT/ITL/
tok-per-s from P_done; the populations are built independently and can drift
by a request or two (done-record outside the profiling-phase cut), so the n
column shows P_wait n and any drift is warned on stderr (large drift aborts).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).parent / "final_tables_manifest.json"
ARM_ORDER = [
    "fcfs",
    "lpm",
    "dfs-weight",
    "hrrn",
    "boost g0.1",
    "boost g0.25",
    "boost g0.5",
    "boost g1.0",
    "boost g5.0",
    "boost ada j60",
]
SEED_LABEL = {"77188": "overload", "42302": "light"}


def load_arm(seed: str, arm: str, dirs: list[str]) -> dict:
    """Load the verified per-run metrics for one arm; abort on any mismatch."""
    runs = []
    for d in dirs:
        run_dir = EXP_ROOT / d
        meta = json.loads((run_dir / "run_meta.json").read_text())
        if str(meta["seed"]) != seed:
            sys.exit(f"ABORT {d}: run_meta seed {meta['seed']} != manifest {seed}")
        v2_path = run_dir / "e2_metrics_v2.json"
        if not v2_path.exists():
            sys.exit(f"ABORT {d}: no e2_metrics_v2.json (run metrics_v2 first)")
        v2 = json.loads(v2_path.read_text())
        n_wait, n_done = v2["queue_wait_s"]["n"], v2["ttft_ms"]["n"]
        if abs(n_wait - n_done) > max(2, 0.05 * n_wait):
            sys.exit(f"ABORT {d}: P_wait n {n_wait} vs P_done n {n_done}")
        if n_wait != n_done:
            print(f"WARN {d}: P_wait n {n_wait} != P_done n {n_done}; "
                  "n column shows P_wait", file=sys.stderr)
        runs.append(
            {
                "q": v2["queue_wait_s"],
                "t": v2["ttft_ms"],
                "i": v2["itl_ms"],
                "hit": v2["cache_hit"]["rate"] * 100,
                "toks": v2["tok_per_s"]["value"],
                "jain": v2["jain_queue_wait"]["value"],
                "w5": v2["worst5_wait_share_pct"]["value"],
            }
        )
    return {"arm": arm, "runs": runs}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def fmt_scaled(values: list[float], unit_scale: float = 1.0) -> str:
    """Adaptive decimals on the mean: 2 under 1, 1 under 10, else rounded."""
    m = mean([v * unit_scale for v in values])
    decimals = 2 if m < 1 else (1 if m < 10 else 0)
    return f"{m:.{decimals}f}"


def fmt_n(values: list[float]) -> str:
    m = mean(values)
    return f"{m:.0f}" if m == int(m) else f"{m:.1f}"


HEADER = [
    "n",
    "q p50 (s)",
    "q p95 (s)",
    "q p99 (s)",
    "TTFT p50 (s)",
    "TTFT p95 (s)",
    "TTFT p99 (s)",
    "ITL p50 (ms)",
    "ITL p95 (ms)",
    "ITL p99 (ms)",
    "hit %",
    "tok/s",
    "Jain",
    "worst-5 %",
]


def main() -> None:
    partial = "--allow-partial" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "--allow-partial"]
    if len(args) != 1:
        sys.exit(__doc__)
    seed = args[0]
    manifest = json.loads(MANIFEST.read_text())
    if seed not in manifest:
        sys.exit(f"ABORT: seed {seed} not in manifest (have {list(manifest)})")
    arms = {a: load_arm(seed, a, dirs) for a, dirs in manifest[seed].items()}
    extra = [a for a in arms if a not in ARM_ORDER]
    if extra:
        sys.exit(f"ABORT: seed {seed} has arms outside ARM_ORDER: {extra}")
    missing = [a for a in ARM_ORDER if a not in arms]
    if missing and not partial:
        sys.exit(f"ABORT: seed {seed} missing arms {missing}")
    order = [a for a in ARM_ORDER if a in arms]
    label = SEED_LABEL.get(seed, seed)
    print(f"## Seed {seed} ({label}, c16) -- runs/arm: "
          + ", ".join(f"{a}={len(arms[a]['runs'])}" for a in order)
          + (f"  PARTIAL missing {missing}" if missing else ""), file=sys.stderr)

    def v(a: str, key: str, field: str) -> list[float]:
        return [r[key][field] for r in arms[a]["runs"]]

    def raw(a: str, key: str) -> list[float]:
        return [r[key] for r in arms[a]["runs"]]

    print("| policy | " + " | ".join(HEADER) + " |")
    print("|---" * (len(HEADER) + 1) + "|")
    for a in order:
        cells = [
            fmt_n(v(a, "q", "n")),
            fmt_scaled(v(a, "q", "p50")),
            fmt_scaled(v(a, "q", "p95")),
            fmt_scaled(v(a, "q", "p99")),
            fmt_scaled(v(a, "t", "p50"), 0.001),
            fmt_scaled(v(a, "t", "p95"), 0.001),
            fmt_scaled(v(a, "t", "p99"), 0.001),
            fmt_scaled(v(a, "i", "p50")),
            fmt_scaled(v(a, "i", "p95")),
            fmt_scaled(v(a, "i", "p99")),
            f"{mean(raw(a, 'hit')):.1f}",
            f"{mean(raw(a, 'toks')):.1f}",
            f"{mean(raw(a, 'jain')):.3f}",
            f"{mean(raw(a, 'w5')):.1f}",
        ]
        print("| " + " | ".join([a] + cells) + " |")


if __name__ == "__main__":
    main()
