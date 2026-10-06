#!/usr/bin/env python3
"""Task-2 frontier plot: locality vs protection at c16.

One point per boost gamma (fresh-boot round-2 cells, connected in gamma
ascending order), with the c16 reference arms (lpm, fcfs-match, fcfs, hrrn)
as distinct markers. Rows: hit rate and oTPM (the locality axes). Columns:
cold-class queue-wait p99 and cold-class starvation rate (the protection
axes). A flat frontier over gamma is itself the finding (memory/eviction
pressure binds, not ordering).

Round-1 cells lack run_meta.server_fresh_boot (the field landed with round
2) and ran cache-warm; they plot as hollow markers with a protocol note. In
a cell with both, the latest-per-seed rule keeps the fresh run.

Writes frontier_c16.png and frontier_c16.json (point coordinates + the
per-point guards: hit within 2 pts of lpm, oTPM within 5%, TPOT p95 within
+10%) under the results root.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from print_results_table import (
    averaged_cell,
    dig,
    fmt_gamma,
    latest_per_seed,
    load_cells,
)

# Okabe-Ito categorical palette; boost owns the connected frontier line.
STYLE = {
    "boost": dict(color="#0072B2", marker="o", label="boost (per gamma)"),
    "lpm": dict(color="#E69F00", marker="s", label="lpm"),
    "fcfs-match": dict(color="#009E73", marker="D", label="fcfs-match"),
    "fcfs": dict(color="#555555", marker="X", label="fcfs"),
    "hrrn": dict(color="#CC79A7", marker="^", label="hrrn"),
}
CONC = 16


def c16_points(cells: dict) -> list[dict]:
    """One point per c16 cell: averaged metrics + protocol (fresh/warm)."""
    points = []
    for (policy, conc, gamma, _k), runs in cells.items():
        if conc != CONC:
            continue
        fresh = any(r["meta"].get("server_fresh_boot") for r in latest_per_seed(runs))
        m = averaged_cell(runs)
        point = {
            "policy": policy,
            "gamma": gamma,
            "gamma_s": fmt_gamma(gamma),
            "fresh_boot": fresh,
            "hit": dig(m, "queue_wait_per_request.first_attempt_cache_hit_rate"),
            "otpm": dig(m, "throughput.output_tokens_per_min"),
            "tpot_p95_ms": dig(m, "tpot.p95_ms"),
            "cold_q_p99_s": dig(m, "queue_wait_per_request.cold_class.p99_s"),
            "cold_starv": dig(m, "queue_wait_per_request.cold_class.starvation_rate"),
        }
        points.append(point)
    return points


def guards(points: list[dict]) -> list[dict]:
    """Task-2 per-point guards vs the lpm c16 baseline."""
    lpm = next((p for p in points if p["policy"] == "lpm"), None)
    if lpm is None:
        return []
    out = []
    for p in points:
        if p["policy"] != "boost":
            continue
        row = {
            "policy": p["policy"],
            "gamma_s": p["gamma_s"],
            "hit": p["hit"],
            "hit_gap_vs_lpm": (
                p["hit"] - lpm["hit"]
                if p["hit"] is not None and lpm["hit"] is not None
                else None
            ),
            "otpm_delta_pct_vs_lpm": (
                (p["otpm"] - lpm["otpm"]) / lpm["otpm"] * 100.0
                if p["otpm"] is not None and lpm["otpm"]
                else None
            ),
            "tpot_p95_delta_pct_vs_lpm": (
                (p["tpot_p95_ms"] - lpm["tpot_p95_ms"]) / lpm["tpot_p95_ms"] * 100.0
                if p["tpot_p95_ms"] is not None and lpm["tpot_p95_ms"]
                else None
            ),
        }
        row["pass_hit_within_2pts"] = (
            row["hit_gap_vs_lpm"] is not None and row["hit_gap_vs_lpm"] >= -0.02
        )
        row["pass_otpm_within_5pct"] = (
            row["otpm_delta_pct_vs_lpm"] is not None
            and row["otpm_delta_pct_vs_lpm"] >= -5.0
        )
        row["pass_tpot_within_10pct"] = (
            row["tpot_p95_delta_pct_vs_lpm"] is not None
            and row["tpot_p95_delta_pct_vs_lpm"] <= 10.0
        )
        out.append(row)
    return out


def _finite(points: list[dict], xk: str, yk: str) -> list[dict]:
    return [
        p
        for p in points
        if isinstance(p.get(xk), (int, float))
        and not math.isnan(p[xk])
        and isinstance(p.get(yk), (int, float))
        and not math.isnan(p[yk])
    ]


def plot(points: list[dict], out_png: Path, gamma_star: float | None) -> None:
    refs = [p for p in points if p["policy"] != "boost"]
    boost = sorted(
        (p for p in points if p["policy"] == "boost" and p["gamma"] is not None),
        key=lambda p: p["gamma"],
    )
    panels = [
        (
            "cold_q_p99_s",
            "hit",
            "cold-class queue-wait p99 (s)",
            "first-attempt hit rate",
        ),
        (
            "cold_q_p99_s",
            "otpm",
            "cold-class queue-wait p99 (s)",
            "output tokens / min",
        ),
        ("cold_starv", "hit", "cold-class starvation rate", "first-attempt hit rate"),
        ("cold_starv", "otpm", "cold-class starvation rate", "output tokens / min"),
    ]
    warm_refs = [p for p in refs if not p["fresh_boot"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    for ax, (xk, yk, xl, yl) in zip(axes.flat, panels):
        for p in refs:
            st = STYLE.get(p["policy"])
            if st is None:
                continue
            ax.scatter(
                p[xk],
                p[yk],
                s=90,
                marker=st["marker"],
                color=st["color"],
                facecolors="none" if not p["fresh_boot"] else st["color"],
                linewidths=1.8,
                label=st["label"],
                zorder=3,
            )
        bx = _finite(boost, xk, yk)
        if bx:
            st = STYLE["boost"]
            ax.plot(
                [p[xk] for p in bx],
                [p[yk] for p in bx],
                color=st["color"],
                linewidth=1.2,
                zorder=2,
            )
            ax.scatter(
                [p[xk] for p in bx],
                [p[yk] for p in bx],
                s=70,
                marker=st["marker"],
                color=st["color"],
                label=st["label"],
                zorder=4,
            )
            for p in bx:
                if gamma_star is not None and p["gamma"] == gamma_star:
                    ax.scatter(
                        [p[xk]],
                        [p[yk]],
                        s=260,
                        facecolors="none",
                        edgecolors=st["color"],
                        linewidths=2.0,
                        zorder=5,
                    )
                ax.annotate(
                    p["gamma_s"],
                    (p[xk], p[yk]),
                    textcoords="offset points",
                    xytext=(6, 5),
                    fontsize=8,
                    color=st["color"],
                )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    axes.flat[0].legend(seen.values(), seen.keys(), loc="best", fontsize=8)
    warm_note = (
        f"; hollow = round-1 warm-start ({', '.join(sorted(p['policy'] for p in warm_refs))})"
        if warm_refs
        else ""
    )
    fig.suptitle(
        f"c16 locality vs protection frontier -- one point per boost gamma"
        f" (seed 42, 1800 s, fresh boot per cell{warm_note})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    print(f"plot -> {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_root",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "results" / "agentx",
    )
    parser.add_argument(
        "--gamma-star",
        type=float,
        default=None,
        help="ring this gamma's point (the chosen gamma*)",
    )
    args = parser.parse_args()

    cells = load_cells(args.results_root)
    points = c16_points(cells)
    if not any(p["policy"] == "boost" for p in points):
        raise SystemExit(f"no c16 boost cells under {args.results_root}")

    out_png = args.results_root / "frontier_c16.png"
    plot(points, out_png, args.gamma_star)
    out_json = args.results_root / "frontier_c16.json"
    out_json.write_text(
        json.dumps(
            {
                "concurrency": CONC,
                "gamma_star": args.gamma_star,
                "points": points,
                "guards_vs_lpm": guards(points),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"data -> {out_json}")
    for g in guards(points):
        print(
            f"  gamma={g['gamma_s']}: hit {g['hit_gap_vs_lpm']:+.3f} vs lpm"
            f"  oTPM {g['otpm_delta_pct_vs_lpm']:+.1f}%"
            f"  TPOT p95 {g['tpot_p95_delta_pct_vs_lpm']:+.1f}%"
            f"  pass: hit={g['pass_hit_within_2pts']} oTPM={g['pass_otpm_within_5pct']}"
            f" TPOT={g['pass_tpot_within_10pct']}"
        )


if __name__ == "__main__":
    main()
