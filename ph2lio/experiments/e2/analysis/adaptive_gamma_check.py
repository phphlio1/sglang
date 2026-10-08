#!/usr/bin/env python3
"""Task 8.3 gamma-Ada validation: trajectory, bracket, and behavior guards.

Reads the adaptive arms (results/1x-rtx6000pro/agentx_adaptive/boost/c*_gada_k256_*), pulls
the per-record gamma trajectory from each run's schedule_eval.jsonl (the
scheduler stamps the current adapted gamma on every record), and checks:

  1. Convergence: mean gamma over the last quarter of arrivals (the EMA
     settles once the sliding window fills; MIN_SAMPLES=64 first admissions).
  2. Bracket: the converged gamma must land inside the round-2 sweep's best
     bracket (--bracket LOW HIGH, the neighbors of gamma*).
  3. Behavior: hit rate and cold-class starvation no worse than the static
     gamma* arm at the same concurrency (hit >= -2 pts; cold starvation
     <= +2 pts absolute).

Static references: c8 gamma* is the fresh round-6 run under the adaptive
root; c16 gamma* is the fresh round-2 cell under the official root. Writes
adaptive_gamma_check.json under the adaptive root and prints a doc-ready
table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

E2 = Path(__file__).resolve().parent.parent


def latest_run(dirs: list[Path]) -> Path | None:
    with_meta = [d for d in dirs if (d / "run_meta.json").exists()]
    return max(with_meta, key=lambda d: d.name) if with_meta else None


def find_run(root: Path, policy: str, prefix: str) -> Path | None:
    return latest_run(sorted((root / policy).glob(f"{prefix}_*")))


def find_static_run(root: Path, conc: int, gamma_star: float) -> Path | None:
    # Run dirs carry BOOST_GAMMA as typed on the command line (3e-5, 1e-4,
    # ...), so match on run_meta's float instead of the label text.
    matches = []
    for d in sorted((root / "boost").glob(f"c{conc}_g*_k256_*")):
        meta_path = d / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("boost_adaptive_gamma"):
            continue
        g = meta.get("boost_gamma")
        if g is not None and abs(float(g) - gamma_star) <= 1e-12 * max(
            1.0, abs(gamma_star)
        ):
            matches.append(d)
    return max(matches, key=lambda d: d.name) if matches else None


def gamma_trajectory(run_dir: Path) -> list[dict]:
    out = []
    path = run_dir / "schedule_eval.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if isinstance(r.get("gamma"), (int, float)) and isinstance(
            r.get("arrival_wall_s"), (int, float)
        ):
            out.append({"arrival_wall_s": r["arrival_wall_s"], "gamma": r["gamma"]})
    out.sort(key=lambda r: r["arrival_wall_s"])
    return out


def converged_gamma(traj: list[dict]) -> float | None:
    if not traj:
        return None
    cut = traj[int(len(traj) * 0.75)]["arrival_wall_s"]
    tail = [r["gamma"] for r in traj if r["arrival_wall_s"] >= cut]
    return sum(tail) / len(tail) if tail else None


def dig(metrics: dict, dotted: str) -> float | None:
    node = metrics
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, (int, float)) else None


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "e2_metrics.json").read_text())


def arm_row(name: str, run_dir: Path | None) -> dict:
    if run_dir is None:
        return {"arm": name, "run": None}
    m = load_metrics(run_dir)
    return {
        "arm": name,
        "run": str(run_dir.relative_to(E2)),
        "n": dig(m, "queue_wait_per_request.n_records"),
        "hit": dig(m, "queue_wait_per_request.first_attempt_cache_hit_rate"),
        "cold_n": dig(m, "queue_wait_per_request.cold_class.n"),
        "cold_p99_s": dig(m, "queue_wait_per_request.cold_class.p99_s"),
        "cold_starv": dig(m, "queue_wait_per_request.cold_class.starvation_rate"),
        "otpm": dig(m, "throughput.output_tokens_per_min"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adaptive-root", type=Path, default=E2 / "results" / "1x-rtx6000pro" / "agentx_adaptive"
    )
    parser.add_argument("--official-root", type=Path, default=E2 / "results" / "1x-rtx6000pro" / "agentx")
    parser.add_argument("--gamma-star", type=float, required=True)
    parser.add_argument(
        "--bracket",
        type=float,
        nargs=2,
        required=True,
        metavar=("LOW", "HIGH"),
        help="round-2 sweep neighbors of gamma*",
    )
    args = parser.parse_args()

    arms = {
        "ada_c8": find_run(args.adaptive_root, "boost", "c8_gada_k256"),
        "ada_c16": find_run(args.adaptive_root, "boost", "c16_gada_k256"),
        "static_c8": find_static_run(args.adaptive_root, 8, args.gamma_star),
        "static_c16": find_static_run(args.official_root, 16, args.gamma_star),
    }

    out = {"gamma_star": args.gamma_star, "bracket": args.bracket, "arms": {}}
    for conc in (8, 16):
        ada, static = arms[f"ada_c{conc}"], arms[f"static_c{conc}"]
        row = {
            "adaptive": arm_row(f"boost (gamma-Ada) c{conc}", ada),
            "static": arm_row(f"boost (gamma*={args.gamma_star:g}) c{conc}", static),
        }
        if ada is not None:
            traj = gamma_trajectory(ada)
            row["trajectory"] = {
                "n_records": len(traj),
                "gamma_first": traj[0]["gamma"] if traj else None,
                "gamma_min": min((r["gamma"] for r in traj), default=None),
                "gamma_max": max((r["gamma"] for r in traj), default=None),
                "gamma_converged": converged_gamma(traj),
            }
            conv = row["trajectory"]["gamma_converged"]
            row["pass_bracket"] = (
                conv is not None and args.bracket[0] <= conv <= args.bracket[1]
            )
        if ada is not None and static is not None:
            a, s = row["adaptive"], row["static"]
            row["pass_hit_within_2pts"] = (
                a["hit"] is not None
                and s["hit"] is not None
                and a["hit"] - s["hit"] >= -0.02
            )
            row["pass_starv_within_2pts"] = (
                a["cold_starv"] is not None
                and s["cold_starv"] is not None
                and a["cold_starv"] - s["cold_starv"] <= 0.02
            )
        out["arms"][f"c{conc}"] = row

    dest = args.adaptive_root / "adaptive_gamma_check.json"
    args.adaptive_root.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")

    for conc in (8, 16):
        row = out["arms"][f"c{conc}"]
        print(f"\n=== c{conc} ===")
        for key in ("adaptive", "static"):
            a = row.get(key) or {}
            print(f"  {a.get('arm', key):<38} run={a.get('run')}")
            print(
                f"    n={a.get('n')} hit={a.get('hit')} "
                f"cold_n={a.get('cold_n')} cold_p99={a.get('cold_p99_s')} "
                f"cold_starv={a.get('cold_starv')}"
            )
        t = row.get("trajectory")
        if t:
            print(
                f"  trajectory: n={t['n_records']} first={t['gamma_first']} "
                f"min={t['gamma_min']} max={t['gamma_max']} "
                f"converged={t['gamma_converged']}"
            )
            print(
                f"  bracket [{args.bracket[0]:g}, {args.bracket[1]:g}]: "
                f"{'PASS' if row['pass_bracket'] else 'FAIL'}"
            )
        for k in ("pass_hit_within_2pts", "pass_starv_within_2pts"):
            if k in row:
                print(f"  {k}: {'PASS' if row[k] else 'FAIL'}")
    print(f"\nwritten -> {dest.relative_to(E2)}")


if __name__ == "__main__":
    main()
