"""Raw-data accuracy checks for the v2 metric sources (docs/metrics_v2_raw_data_check.md).

Answers "is the raw data itself accurate" with three read-only checks over
the canonical cells (the c4/c8 full run 2026-10-07 and the c16 sweep
2026-10-08, identified by commit stamp):

  1. Little's law: mean queue_wait * n / profiling window (client-side eval
     log) vs the server's own sglang:num_queue_reqs gauge mean over
     profiling-phase scrapes. Two fully independent measurement paths;
     a corrupted queue_wait_s would break the identity.
  2. Physical constraint ttft_s >= queue_wait_s on every P_wait record
     carrying both fields: service cannot start before admission.
  3. Raw sorted wait tails: the c4/c8 bimodal cliff and the c16 saturation
     blockers, read straight from the eval records.

server_metrics_export.jsonl.gz (the gzip policy on c4/c8 cells) is read
through gzip, never extracted. Nothing is written.

Usage:
    python3 analysis/raw_data_check.py [--results-root DIR] [--json]
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_v2 import load_done_rows, load_wait_rows, warmup_cut_s
from print_metrics_v2_table import CANONICAL_COMMITS

# Show the raw tail of these cells (the doc's bimodal/saturation exhibits).
TAIL_CELLS = ("lpm/c4_20261007_022137", "lpm/c16_20261008_011852", "boost/c16_gada_wall_k256_20261008_045917")
TAIL_K = 12


def gauge_mean_queue_reqs(run_dir: Path, cut_s: float | None) -> tuple[float | None, int]:
    """Mean sglang:num_queue_reqs over post-cut profiling scrapes, .gz-aware."""
    plain = run_dir / "server_metrics_export.jsonl"
    path = plain if plain.exists() else run_dir / "server_metrics_export.jsonl.gz"
    opener = open if plain.exists() else gzip.open
    values = []
    with opener(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("benchmark_phase") != "profiling":
                continue
            if cut_s is not None and record["timestamp_ns"] / 1e9 < cut_s:
                continue
            for family in record["metrics"].get("sglang:num_queue_reqs", []):
                values.append(family["value"])
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def check_cell(run_dir: Path) -> dict:
    """One cell's checks; the values feed the doc's section 2/3 tables."""
    done_rows, _ = load_done_rows(run_dir)
    wait_rows, _ = load_wait_rows(run_dir, done_rows=done_rows)
    cut_s, _ = warmup_cut_s(run_dir, done_rows=done_rows)

    stamped = [(r["start_ns"], r["end_ns"]) for r in done_rows if r["start_ns"] and r["end_ns"]]
    window_s = (max(e for _, e in stamped) - min(s for s, _ in stamped)) / 1e9
    waits = [r["queue_wait_s"] for r in wait_rows]
    gauge, n_scrapes = gauge_mean_queue_reqs(run_dir, cut_s)
    return {
        "policy": run_dir.parent.name,
        "run_dir_name": run_dir.name,
        "concurrency": json.loads((run_dir / "run_meta.json").read_text())["concurrency"],
        "n_wait": len(wait_rows),
        "mean_wait_s": sum(waits) / len(waits) if waits else None,
        "window_s": window_s,
        "littles_l_n_over_t": sum(waits) / window_s if waits and window_s > 0 else None,
        "gauge_queue_reqs_mean": gauge,
        "gauge_scrapes": n_scrapes,
        "n_ttft_lt_wait": sum(
            1 for r in wait_rows if r["ttft_s"] is not None and r["ttft_s"] < r["queue_wait_s"]
        ),
        "waits_sorted_desc": sorted(waits, reverse=True),
        "wait_rows": wait_rows,
    }


def canonical_run_dirs(results_root: Path) -> list[Path]:
    dirs = []
    for meta_path in sorted(results_root.glob("*/*/run_meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("sglang_commit", "")[:10] in CANONICAL_COMMITS:
            dirs.append(meta_path.parent)
    return dirs


def _tail_lines(checks: list[dict]) -> list[str]:
    lines = []
    for name in TAIL_CELLS:
        policy, dirname = name.split("/", 1)
        match = next((c for c in checks if c["policy"] == policy and c["run_dir_name"] == dirname), None)
        if match is None:
            continue
        waits = match["waits_sorted_desc"]
        lines.append(
            f"{name}: n={len(waits)} top{TAIL_K}="
            + ", ".join(f"{w:.3f}" for w in waits[:TAIL_K])
            + f" #>1s={sum(1 for w in waits if w > 1.0)}"
        )
        for row in sorted(match["wait_rows"], key=lambda r: -r["queue_wait_s"])[:3]:
            lines.append(
                f"  wait={row['queue_wait_s']:.1f}s input={row['input_len']}"
                f" matched={row['matched_first']}"
                f" uncached={row['input_len'] - row['matched_first']}"
            )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "1x-rtx6000pro" / "agentx",
    )
    parser.add_argument("--json", action="store_true", help="print the per-cell checks as JSON")
    args = parser.parse_args()

    run_dirs = canonical_run_dirs(args.results_root)
    if not run_dirs:
        sys.exit(f"no canonical run dirs under {args.results_root}")

    checks = [check_cell(rd) for rd in run_dirs]
    checks.sort(key=lambda c: (c["concurrency"], c["policy"]))

    if args.json:
        slim = [{k: v for k, v in c.items() if k not in ("wait_rows", "waits_sorted_desc")} for c in checks]
        print(json.dumps(slim, indent=2))
        return

    print(f"{'policy':<12} c   n_wait mean_wait  window   L=W*n/T  gauge  scrapes ttft<wait")
    for c in checks:
        print(
            f"{c['policy']:<12} c{c['concurrency']:<3} {c['n_wait']:<6}"
            f" {c['mean_wait_s']:<9.3f} {c['window_s']:<7.1f} {c['littles_l_n_over_t']:<8.3f}"
            f" {c['gauge_queue_reqs_mean']:<6.3f} {c['gauge_scrapes']:<7} {c['n_ttft_lt_wait']}"
        )
    print()
    for line in _tail_lines(checks):
        print(line)


if __name__ == "__main__":
    main()
