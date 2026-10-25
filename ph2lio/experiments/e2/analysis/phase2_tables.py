"""H200 phase-2 manager-format table (2 seeds x 6 arms, c64, n=1).

Same format as the RTX final tables (final_tables.py): dfs-weight baseline
row with plain values, every other cell "value (pct%)" computed from the
displayed values, signed so + is universally better. Reuses the
final_tables helpers; only the orchestration differs (manifest, arm
order, c64 heading).

    python3 analysis/phase2_tables.py <seed> [--allow-partial]

Reads analysis/phase2_tables_manifest.json: {"<seed>": {"<arm>":
[run_dir, ...]}} (run_dir relative to the experiment root). Interim n=1
view of the phase-2 cells -- final H200 conclusions need >= 3 seeds or
n >= 2 per cell (N_RUNS).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import final_tables as ft

MANIFEST = Path(__file__).parent / "phase2_tables_manifest.json"
ARM_ORDER = [
    "fcfs",
    "lpm",
    "dfs-weight",
    "hrrn",
    "boost g0.25",
    "boost ada j50",
]
# Boost-cache fusion arms (2026-10-25): optional until their cells land -- a
# manifest without them still tables (classic arms stay required), so the
# pre-bc interim tables run without --allow-partial.
BC_ARM_ORDER = [
    "bc l0.5 m2048",
    "bc l0.5 m8192",
    "bc l0.5 m32768",
]
BASELINE = "dfs-weight"
SEED_LABEL = {"77188": "overload", "42302": "light"}


def main() -> None:
    partial = "--allow-partial" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "--allow-partial"]
    if len(args) != 1:
        sys.exit(__doc__)
    seed = args[0]
    manifest = json.loads(MANIFEST.read_text())
    if seed not in manifest:
        sys.exit(f"ABORT: seed {seed} not in manifest (have {list(manifest)})")
    arms = {a: ft.load_arm(seed, a, dirs) for a, dirs in manifest[seed].items()}
    extra = [a for a in arms if a not in ARM_ORDER + BC_ARM_ORDER]
    if extra:
        sys.exit(f"ABORT: seed {seed} has arms outside ARM_ORDER: {extra}")
    missing = [a for a in ARM_ORDER if a not in arms]
    if missing and not partial:
        sys.exit(f"ABORT: seed {seed} missing arms {missing}")
    if BASELINE not in arms:
        sys.exit(f"ABORT: seed {seed} missing baseline {BASELINE}")
    order = (
        [BASELINE]
        + [a for a in ARM_ORDER if a != BASELINE and a in arms]
        + [a for a in BC_ARM_ORDER if a in arms]
    )
    label = SEED_LABEL.get(seed, seed)
    print(
        f"## Seed {seed} ({label}, c64) -- runs/arm: "
        + ", ".join(f"{a}={len(arms[a]['runs'])}" for a in order)
        + (f"  PARTIAL missing {missing}" if missing else ""),
        file=sys.stderr,
    )
    print(f"## Seed {seed} ({label}, c64)")
    print()
    print("| policy | " + " | ".join(h for h, *_ in ft.COLUMNS) + " |")
    print("|---" * (len(ft.COLUMNS) + 1) + "|")
    base_runs = arms[BASELINE]["runs"]
    for a in order:
        row_label = a if a != BASELINE else f"{a} (baseline)"
        print(ft.table_row(row_label, arms[a]["runs"], base_runs))


if __name__ == "__main__":
    main()
