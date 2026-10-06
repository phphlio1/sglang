#!/usr/bin/env python3
"""Task-8.1 offline gamma-hat estimator from existing schedule-eval logs.

gamma-hat = ln(0.05/0.01) / (w99 - w95) = ln 5 / Delta, with the admission
wait tail measured in CLOCK TOKENS: w = queue_wait_s * R, where R is the
per-concurrency total-token throughput of the fcfs arm (no admission-time
token stamp exists in these logs; the fcfs arm is the policy-neutral rate --
this round's fcfs-match arm refreshes it once it lands).

Two reference tails per concurrency:
  - lpm's tail: the gamma a locality-driven policy's wait distribution
    implies (what gamma-Ada would estimate while running lpm-like ordering);
  - boost's own tail (gamma=1e-4): the fixed point boost sits at today.
gamma-Ada should converge between the two (it measures the tail under its
own current policy).

Clip range [3e-7, 1e-3]: the runtime clip. The offline table reports both
the raw estimate and the clipped value.

Writes gamma_hat_offline.json under the results root and prints the table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from agentx_metrics import load_run_meta, load_schedule_eval_records
from starvation import _percentile

LN5 = math.log(5.0)
GAMMA_MIN, GAMMA_MAX = 3e-7, 1e-3


def gamma_hat(delta_tokens: float) -> float:
    """ln5 / (w99 - w95) in clock tokens; the tail-width-to-gamma map."""
    if delta_tokens <= 0:
        return float("inf")
    return LN5 / delta_tokens


def clip(g: float) -> float:
    return min(max(g, GAMMA_MIN), GAMMA_MAX)


def run_tail_tokens(run_dir: Path, r_tps: float) -> dict | None:
    """Wait-tail stats for one run, scaled to clock tokens by R."""
    records, _ = load_schedule_eval_records(run_dir)
    waits = [
        r["queue_wait_s"]
        for r in records
        if isinstance(r.get("queue_wait_s"), (int, float))
    ]
    if len(waits) < 20:
        return None
    ordered = sorted(waits)
    p95, p99 = _percentile(ordered, 95), _percentile(ordered, 99)
    return {
        "n": len(waits),
        "p95_s": p95,
        "p99_s": p99,
        "delta_s": p99 - p95,
        "p95_tokens": p95 * r_tps,
        "p99_tokens": p99 * r_tps,
        "delta_tokens": (p99 - p95) * r_tps,
    }


def load_r_tps(run_dir: Path) -> float | None:
    path = run_dir / "e2_metrics.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())
    tps = metrics.get("throughput", {}).get("total_token_throughput_tps")
    return tps if isinstance(tps, (int, float)) else None


def find_run(
    policy_dir: Path, concurrency: int, gamma: float | None = None
) -> Path | None:
    """Latest run dir of a policy at a concurrency (optionally a boost gamma)."""
    best = None
    for run_dir in policy_dir.glob("c*_*"):
        meta = load_run_meta(run_dir)
        if meta.get("concurrency") != concurrency:
            continue
        if gamma is not None and policy_dir.name == "boost":
            if abs(float(meta.get("boost_gamma") or 0) - gamma) > 1e-12:
                continue
        best = run_dir  # glob order is unordered; keep the lexicographically last
        if best is None or run_dir.name > best.name:
            best = run_dir
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_root",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "results" / "agentx",
    )
    args = parser.parse_args()
    root = args.results_root

    table = []
    for conc in (4, 8, 16):
        fcfs_dir = find_run(root / "fcfs", conc)
        if fcfs_dir is None:
            continue
        r_tps = load_r_tps(fcfs_dir)
        row: dict = {"concurrency": conc, "r_tps_fcfs": r_tps, "tails": {}}
        for label, policy, gamma in (
            ("lpm", "lpm", None),
            ("boost_g1e-4", "boost", 1e-4),
        ):
            run_dir = find_run(root / policy, conc, gamma)
            if run_dir is None:
                continue
            tail = run_tail_tokens(run_dir, r_tps or 0.0)
            if tail is None:
                continue
            raw = gamma_hat(tail["delta_tokens"])
            tail["gamma_hat_raw"] = raw
            tail["gamma_hat_clipped"] = clip(raw)
            row["tails"][label] = tail
        table.append(row)

        tails = row["tails"]
        parts = [f"c{conc}  R={r_tps and format(r_tps, '.0f') or 'n/a'} tok/s"]
        for label, tail in tails.items():
            parts.append(
                f"{label}: p99-p95={tail['delta_s']:.3f}s = {tail['delta_tokens']:,.0f} tok"
                f" -> gamma_hat={tail['gamma_hat_raw']:.2e} (clip {tail['gamma_hat_clipped']:.1e})"
            )
        print("\n  ".join(parts))

    out = {"clip_range": [GAMMA_MIN, GAMMA_MAX], "table": table}
    out_path = root / "gamma_hat_offline.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
