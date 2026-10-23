"""Build the final sweep table for one seed from e2_metrics_v2.json.

Usage:

    python3 analysis/final_tables.py <seed> [--allow-partial]

Prints one markdown table: dfs-weight is the baseline row (plain values);
every other row shows "value (pct%)" per cell, pct being the difference vs the
baseline COMPUTED FROM THE DISPLAYED VALUES (calculator-checkable), signed so
+ always means better: n/hit/tok-per-s/Jain are higher-is-better, queue/TTFT/
ITL/worst-5 are lower-is-better. Means replace the p50 columns and n shows
the plain value with no pct (manager format, 2026-10-23); n is the mean over
runs displayed as a plain integer (half-up).

Reads analysis/final_tables_manifest.json: {"<seed>": {"<arm>": [run_dir, ...]}}
(run_dir relative to the experiment root). Every listed dir must already carry
a verified e2_metrics_v2.json (metrics_v2 + cross_check 17/17) and a run_meta
seed matching the manifest key; anything else aborts before printing unless
--allow-partial (interim views; the final file must run without it). Cells are
the mean over the listed runs -- no hand-copied numbers.

Sources (docs/metrics_v2.md): n/queue/hit/Jain/worst-5 come from P_wait, TTFT/
ITL/tok-per-s from P_done; the populations are built independently and can
drift by a request or two (done-record outside the profiling-phase cut), so
the n column shows P_wait n and any drift is warned on stderr (large drift
aborts).
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
BASELINE = "dfs-weight"
SEED_LABEL = {"77188": "overload", "42302": "light"}

# fmt mode: None -> plain integer (half-up); "scaled" -> 2 decimals under 1,
# 1 under 100, else rounded; int -> that many decimals.
FMT_SCALED = "scaled"
COLUMNS: list[tuple[str, float, str | int | None, bool]] = [
    ("n", 1.0, None, True),
    ("q mean (s)", 1.0, FMT_SCALED, False),
    ("q p95 (s)", 1.0, FMT_SCALED, False),
    ("q p99 (s)", 1.0, FMT_SCALED, False),
    ("TTFT mean (s)", 0.001, FMT_SCALED, False),
    ("TTFT p95 (s)", 0.001, FMT_SCALED, False),
    ("TTFT p99 (s)", 0.001, FMT_SCALED, False),
    ("ITL mean (ms)", 1.0, FMT_SCALED, False),
    ("ITL p95 (ms)", 1.0, FMT_SCALED, False),
    ("ITL p99 (ms)", 1.0, FMT_SCALED, False),
    ("hit %", 1.0, 1, True),
    ("tok/s", 1.0, 1, True),
    ("Jain", 1.0, 3, True),
    ("worst-5 %", 1.0, 1, False),
]


def run_metrics(v2: dict) -> dict:
    """Flat per-run dict keyed by the COLUMNS headers."""
    q, t, i = v2["queue_wait_s"], v2["ttft_ms"], v2["itl_ms"]
    return {
        "n": q["n"],
        "q mean (s)": q["mean"],
        "q p95 (s)": q["p95"],
        "q p99 (s)": q["p99"],
        "TTFT mean (s)": t["mean"],
        "TTFT p95 (s)": t["p95"],
        "TTFT p99 (s)": t["p99"],
        "ITL mean (ms)": i["mean"],
        "ITL p95 (ms)": i["p95"],
        "ITL p99 (ms)": i["p99"],
        "hit %": v2["cache_hit"]["rate"] * 100,
        "tok/s": v2["tok_per_s"]["value"],
        "Jain": v2["jain_queue_wait"]["value"],
        "worst-5 %": v2["worst5_wait_share_pct"]["value"],
    }


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
        runs.append(run_metrics(v2))
    return {"arm": arm, "runs": runs}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def format_cell(values: list[float], scale: float, mode: str | int | None) -> str:
    m = mean([v * scale for v in values])
    if mode is None:
        return str(int(m + 0.5))
    decimals = (
        (2 if m < 1 else (1 if m < 100 else 0)) if mode == FMT_SCALED else mode
    )
    return f"{m:.{decimals}f}"


def pct_cell(value_disp: str, base_disp: str, higher_is_better: bool) -> str:
    v, b = float(value_disp), float(base_disp)
    diff = v - b if higher_is_better else b - v
    return f"{100 * diff / b:+.1f}%"


# n shows the plain value only -- no pct vs the baseline (manager format).
NO_PCT = {"n"}


def table_row(label: str, runs: list[dict], base_runs: list[dict]) -> str:
    cells = []
    for header, scale, mode, higher in COLUMNS:
        disp = format_cell([r[header] for r in runs], scale, mode)
        if base_runs is runs or header in NO_PCT:
            cells.append(disp)
        else:
            base = format_cell([r[header] for r in base_runs], scale, mode)
            cells.append(f"{disp} ({pct_cell(disp, base, higher)})")
    return "| " + " | ".join([label] + cells) + " |"


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
    if BASELINE not in arms:
        sys.exit(f"ABORT: seed {seed} missing baseline {BASELINE}")
    order = [BASELINE] + [a for a in ARM_ORDER if a != BASELINE and a in arms]
    label = SEED_LABEL.get(seed, seed)
    print(f"## Seed {seed} ({label}, c16) -- runs/arm: "
          + ", ".join(f"{a}={len(arms[a]['runs'])}" for a in order)
          + (f"  PARTIAL missing {missing}" if missing else ""), file=sys.stderr)

    print(f"## Seed {seed} ({label}, c16)")
    print()
    print("| policy | " + " | ".join(h for h, *_ in COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    base_runs = arms[BASELINE]["runs"]
    for a in order:
        label_row = a if a != BASELINE else f"{a} (baseline)"
        print(table_row(label_row, arms[a]["runs"], base_runs))


if __name__ == "__main__":
    main()
