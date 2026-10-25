"""B-stage thrash audit: eviction and retraction-victim evidence, no reruns.

Parses the v2-sweep manifest arms' existing artifacts:
  server_metrics_export.jsonl  sglang:evicted_tokens (cumulative counter),
                               sglang:kv_evictable_tokens (gauge)
  schedule_eval.jsonl          per-request retract victims + prefix loss
  server_info.json             max_total_num_tokens (pool size)
  e2_metrics_v2.json           hit% / tok/s context

Thrash hypothesis under test: boost-family arms evict far more than
dfs/lpm, scrape kv_evictable toward zero under load, and the retract
victims are big-input requests whose cached prefix is partly lost on
re-admission (matched_first > matched_last).

    python3 analysis/thrash_audit.py <seed>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).parent.parent
MANIFEST = Path(__file__).parent / "e2a_v2_tables_manifest.json"
ARM_ORDER = [
    "dfs-weight",
    "lpm",
    "boost g0.25",
    "ada j50",
    "bc ada j50 a0.25",
    "bc ada j50 a0.5",
    "bc ada j50 a1.0",
]
EVICTED = "sglang:evicted_tokens"
EVICTABLE = "sglang:kv_evictable_tokens"
# Boot + benchmark ramp; evictable starts near 0 on an empty tree, so the
# floor must be read after the tree fills, not from the whole timeline.
WARMUP_S = 300.0
DECAY_BINS = [(0, 30), (30, 100), (100, 300), (300, 10**9)]


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(int(q * (len(s) - 1)), len(s) - 1)]


def eviction_stats(run_dir: Path) -> dict:
    evicted_total = 0.0
    evictable: list[float] = []
    first_ns = None
    with (run_dir / "server_metrics_export.jsonl").open() as f:
        for line in f:
            if EVICTED not in line and EVICTABLE not in line:
                continue
            d = json.loads(line)
            ns = d["timestamp_ns"]
            if first_ns is None:
                first_ns = ns
            samples = d["metrics"].get(EVICTED)
            if samples:
                # Cumulative counter: last snapshot wins (sum over label sets).
                evicted_total = sum(s["value"] for s in samples)
            if ns - first_ns >= WARMUP_S * 1e9:
                samples = d["metrics"].get(EVICTABLE)
                if samples:
                    evictable.append(sum(s["value"] for s in samples))
    return {
        "evicted_tokens": evicted_total,
        "evictable_p10": pct(evictable, 0.10),
        "evictable_med": pct(evictable, 0.50),
        "evictable_final": evictable[-1] if evictable else float("nan"),
    }


def victim_stats(run_dir: Path) -> dict:
    inputs, victim_inputs = [], []
    loss_tokens, loss_victims, events = 0, 0, 0
    with (run_dir / "schedule_eval.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            inputs.append(r["input_len"])
            n = r["num_retractions"] or 0
            if n:
                events += n
                victim_inputs.append(r["input_len"])
                loss = (r["matched_prefix_tokens_first_admission"] or 0) - (
                    r["matched_prefix_tokens_last_admission"] or 0
                )
                if loss > 0:
                    loss_victims += 1
                    loss_tokens += loss
    return {
        "n_reqs": len(inputs),
        "n_victims": len(victim_inputs),
        "retract_events": events,
        "victim_input_p50": pct(victim_inputs, 0.50),
        "pop_input_p50": pct(inputs, 0.50),
        "victims_above_p75": sum(1 for v in victim_inputs if v > pct(inputs, 0.75)),
        "prefix_loss_tokens": loss_tokens,
        "victims_with_loss": loss_victims,
    }


def decay_table(run_dir: Path) -> list[str]:
    """Mean first-admission match ratio per queue-wait bin.

    Separates prefix decay during queue wait from decay during the agent's
    inter-turn idle gap: a low m_hat in the 0-30s bin means the prefix was
    already gone at arrival, before any wait.
    """
    rows = [json.loads(line) for line in (run_dir / "schedule_eval.jsonl").open()]
    cells = []
    for lo, hi in DECAY_BINS:
        sel = [
            r
            for r in rows
            if lo <= r["queue_wait_s"] < hi and r["input_len"] > 0
        ]
        if sel:
            mhat = (
                sum(
                    (r["matched_prefix_tokens_first_admission"] or 0)
                    / r["input_len"]
                    for r in sel
                )
                / len(sel)
            )
            cells.append(f"{mhat:.2f}/{len(sel)}")
        else:
            cells.append("-/0")
    return cells


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    seed = sys.argv[1]
    manifest = json.loads(MANIFEST.read_text())
    if seed not in manifest:
        sys.exit(f"ABORT: seed {seed} not in manifest (have {list(manifest)})")

    print(f"## Thrash audit, seed {seed} (pool size from server_info.json)")
    print()
    header = (
        "| policy | evicted tok | evict p10 | evict med | evict final | "
        "pool %p10 | retract | n vict | vict in p50 | pop in p50 | "
        "vict>p75 | prefix lost tok | lost vict | hit % | tok/s |"
    )
    print(header)
    print("|---" * (header.count("|") - 1) + "|")
    for arm in ARM_ORDER:
        dirs = manifest[seed].get(arm)
        if not dirs:
            continue
        run_dir = EXPERIMENT_ROOT / dirs[0]
        ev = eviction_stats(run_dir)
        vi = victim_stats(run_dir)
        pool = json.loads((run_dir / "server_info.json").read_text())[
            "max_total_num_tokens"
        ]
        m = json.loads((run_dir / "e2_metrics_v2.json").read_text())
        hit = 100 * m["cache_hit"]["rate"]
        tps = m["tok_per_s"]["value"]
        print(
            f"| {arm} | {ev['evicted_tokens']:.0f} | {ev['evictable_p10']:.0f} | "
            f"{ev['evictable_med']:.0f} | {ev['evictable_final']:.0f} | "
            f"{100 * ev['evictable_p10'] / pool:.1f} | "
            f"{vi['retract_events']} | {vi['n_victims']} | "
            f"{vi['victim_input_p50']:.0f} | {vi['pop_input_p50']:.0f} | "
            f"{vi['victims_above_p75']} | "
            f"{vi['prefix_loss_tokens']} | {vi['victims_with_loss']} | "
            f"{hit:.1f} | {tps:.1f} |"
        )

    print()
    print("## Match ratio (m_hat, first admission) by queue-wait bin")
    print()
    header = "| policy | " + " | ".join(
        f"{lo}-{'' if hi > 10**8 else hi}s" for lo, hi in DECAY_BINS
    ) + " |"
    print(header)
    print("|---" * (header.count("|") - 1) + "|")
    for arm in ARM_ORDER:
        dirs = manifest[seed].get(arm)
        if not dirs:
            continue
        print(
            f"| {arm} | "
            + " | ".join(decay_table(EXPERIMENT_ROOT / dirs[0]))
            + " |"
        )


if __name__ == "__main__":
    main()
