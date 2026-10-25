"""e2A lambda-sweep manager-format tables (9 arms x 2 seeds, c16, n=1).

The RTX campaign format (final_tables.py): dfs-weight baseline row with
plain values, every other cell "value (pct%)" computed from the displayed
values, + universally better. Arm set is the e2A grid from
docs/e2a_design.md: boost g0.25 is the lambda=0 reference, the bc arms are
the fusion (lambda dose-response, cap sensitivity, ada-J shared budget),
lpm + dfs-weight are the reference lines.

    python3 analysis/e2a_tables.py <seed> [--allow-partial]

Reads analysis/e2a_tables_manifest.json: {"<seed>": {"<arm>": [run_dir,
...]}} (run_dir relative to the experiment root). Interim n=1 view;
final e2A conclusions need n >= 2 per cell (same-seed repeats vary).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import final_tables as ft

MANIFEST = Path(__file__).parent / "e2a_tables_manifest.json"
ARM_ORDER = [
    "dfs-weight",
    "lpm",
    "boost g0.25",
    "bc l0.25 m8192",
    "bc l0.5 m8192",
    "bc l1.0 m8192",
    "bc l0.5 m2048",
    "bc l0.5 m32768",
    "bc ada j50",
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
    print(
        f"## Seed {seed} ({label}, c16) -- runs/arm: "
        + ", ".join(f"{a}={len(arms[a]['runs'])}" for a in order)
        + (f"  PARTIAL missing {missing}" if missing else ""),
        file=sys.stderr,
    )
    print(f"## Seed {seed} ({label}, c16)")
    print()
    print("| policy | " + " | ".join(h for h, *_ in ft.COLUMNS) + " |")
    print("|---" * (len(ft.COLUMNS) + 1) + "|")
    base_runs = arms[BASELINE]["runs"]
    for a in order:
        row_label = a if a != BASELINE else f"{a} (baseline)"
        print(ft.table_row(row_label, arms[a]["runs"], base_runs))


if __name__ == "__main__":
    main()
