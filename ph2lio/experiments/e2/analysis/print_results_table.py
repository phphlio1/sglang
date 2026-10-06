#!/usr/bin/env python3
"""Print the E2 policy-comparison tables from per-run e2_metrics.json files.

Default: the canonical FINAL TABLE (markdown) -- rows grouped by concurrency,
each group led by its lpm baseline (bold), boost rows sorted by gamma
ascending, then hrrn / fcfs-match / fcfs, then any other arms; two computed
delta columns against the lpm baseline of the same concurrency; footnotes
(a)-(d). Within a cell the latest run per seed is kept (a fresh-boot re-run
supersedes an older same-seed run of that cell); seeds then average into one
n-weighted row by default (--per-seed emits one row per seed).

--flat prints the round-1 flat table (one row per cell, latest run wins),
--calibration the R / b_gamma(k) calibration table.

Pure reporting: reads JSON, prints text. No server contact.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

# Display order matches the matrix driver (run_boost_matrix.sh).
POLICY_ORDER = {"fcfs": 0, "lpm": 1, "boost": 2, "hrrn": 3, "dfs-weight": 4, "lof": 5}
SWEEP_GAMMAS = ("3e-6", "5e-6", "1e-5", "2e-5", "3e-5", "5e-5", "1e-4", "2e-4", "5e-4")
# Within a concurrency group: lpm baseline first, boost by gamma, then the
# reference arms, then anything else.
FINAL_POLICY_ORDER = {"lpm": 0, "boost": 1, "hrrn": 2, "fcfs-match": 3, "fcfs": 4}

EM = "—"  # em dash: no value / not applicable
MINUS = "−"  # minus sign, matches the doc tables


def load_cells(results_root: Path) -> dict[tuple, list[dict]]:
    """(policy, conc, gamma, k) -> [(meta, metrics), ...] per run dir.

    Every run dir with run_meta.json is a comparison cell member (smoke runs
    are not); the final table averages replicas, the flat table keeps the
    latest.
    """
    cells: dict[tuple, list[dict]] = {}
    for metrics_path in sorted(results_root.glob("*/*/e2_metrics.json")):
        run_dir = metrics_path.parent
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        policy = meta["policy"]
        conc = int(meta["concurrency"])
        gamma, k = (
            (float(meta["boost_gamma"]), int(meta["boost_bin_size"]))
            if policy == "boost"
            else (None, None)
        )
        cells.setdefault((policy, conc, gamma, k), []).append(
            {"meta": meta, "metrics": json.loads(metrics_path.read_text())}
        )
    return cells


def cell_sort_key(key: tuple) -> tuple:
    policy, conc, gamma, k = key
    gamma_s = f"{gamma:.10g}" if gamma is not None else ""
    return (POLICY_ORDER.get(policy, 99), conc, gamma_s, k or 0)


def latest(cells: dict[tuple, list[dict]]) -> dict[tuple, dict]:
    """One (meta, metrics) per cell: the latest run dir wins."""
    out = {}
    for key, runs in cells.items():
        out[key] = max(runs, key=lambda r: r["meta"].get("timestamp") or "")
    return out


def fmt(v, spec=".3f") -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    return format(v, spec)


def fmt_gamma(gamma: float | None) -> str:
    if gamma is None:
        return EM
    s = format(gamma, ".0e")  # 1e-04 -> 1e-4
    return re.sub(r"e([+-])0(\d)", r"e\1\2", s)


def signed_pct(x: float | None, spec=".1f") -> str:
    if x is None:
        return EM
    sign = "+" if x >= 0 else MINUS
    return f"{sign}{format(abs(x), spec)}%"


def signed_fmt(v: float | None, spec=".3f") -> str:
    if v is None:
        return EM
    sign = "+" if v >= 0 else MINUS
    return f"{sign}{format(abs(v), spec)}"


# ---------------------------------------------------------------------------
# Final table
# ---------------------------------------------------------------------------


def n_weighted_mean(pairs: list[tuple[float | None, int]]) -> float | None:
    """n-weighted mean over replicas; a replica with a missing value is
    dropped from both sides (its requests cannot contribute a number it
    never produced)."""
    vals = [
        (v, n)
        for v, n in pairs
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))
    ]
    if not vals:
        return None
    total_n = sum(n for _, n in vals)
    return sum(v * n for v, n in vals) / total_n if total_n else None


def latest_per_seed(runs: list[dict]) -> list[dict]:
    """Latest run per seed within a cell: a re-run of the same cell and seed
    (e.g. a fresh-boot round-2 run over a round-1 warm-start run) supersedes
    the older one; different seeds are replicas and both stay."""
    by_seed: dict = {}
    for r in runs:
        seed = r["meta"].get("seed")
        prev = by_seed.get(seed)
        if prev is None or (r["meta"].get("timestamp") or "") >= (
            prev["meta"].get("timestamp") or ""
        ):
            by_seed[seed] = r
    return list(by_seed.values())


def averaged_cell(runs: list[dict]) -> dict:
    """One n-weighted row from a cell's seed replicas (single run: itself)."""
    runs = latest_per_seed(runs)
    if len(runs) == 1:
        return runs[0]["metrics"]
    total_n = sum(r["metrics"].get("n_complete_records") or 0 for r in runs)
    out: dict = {"n_complete_records": total_n}

    def avg(path: str) -> float | None:
        return n_weighted_mean(
            [
                (dig(r["metrics"], path), r["metrics"].get("n_complete_records") or 0)
                for r in runs
            ]
        )

    def dig(d: dict, path: str):
        for part in path.split("."):
            if not isinstance(d, dict):
                return None
            d = d.get(part)
        return d

    for path in (
        "queue_wait_per_request.first_attempt_cache_hit_rate",
        "queue.queue_time_p95_s",
        "queue.queue_time_p99_s",
        "queue.starvation_rate",
        "queue_wait_per_request.cold_class.n",
        "queue_wait_per_request.cold_class.starvation_rate",
        "queue_wait_per_request.warm_class.n",
        "queue_wait_per_request.warm_class.starvation_rate",
        "queue_wait_per_request.cold_warm_queue_wait_gap.p99_s",
        "throughput.request_per_min",
        "throughput.output_tokens_per_min",
        "ttft.mean_ms",
        "tpot.p95_ms",
    ):
        node = out
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = avg(path)
    retr = sum(
        (
            r["metrics"]
            .get("queue_wait_per_request", {})
            .get("retractions", {})
            .get("value")
            or 0
        )
        for r in runs
    )
    out.setdefault("queue_wait_per_request", {})["retractions"] = {"value": retr}
    return out


def dig(d: dict, path: str):
    for part in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def q_p99_delta(cfg_q_p99: float | None, lpm_q_p99: float | None) -> str:
    if cfg_q_p99 is None or lpm_q_p99 in (None, 0):
        return EM
    # From the displayed .3f values so a reader recomputing the delta from the
    # printed columns gets the printed answer.
    cfg_r, lpm_r = round(cfg_q_p99, 3), round(lpm_q_p99, 3)
    return signed_pct((cfg_r - lpm_r) / lpm_r * 100.0)


def anti_starvation_delta(cfg_starv: float | None, lpm_starv: float | None) -> str:
    if cfg_starv is None or lpm_starv is None:
        return EM
    # Same displayed-value rule as q_p99_delta; the zero-baseline check reads
    # the displayed value too (0.0004 displays as 0.000 and is degenerate).
    cfg_r, lpm_r = round(cfg_starv, 3), round(lpm_starv, 3)
    if lpm_r == 0:
        # Relative change is undefined against a zero baseline.
        if cfg_r == 0:
            return "0% ¹"
        return f"{MINUS}{cfg_r * 100:.1f} pp ¹"
    return signed_pct((lpm_r - cfg_r) / lpm_r * 100.0)


def final_rows(cells: dict[tuple, list[dict]], per_seed: bool) -> list[dict]:
    """Ordered row dicts: one per (cell [, seed]) with lpm baselines attached."""
    by_conc: dict[int, list[dict]] = {}
    for (policy, conc, gamma, k), runs in cells.items():
        seed_latest = latest_per_seed(runs)
        replicas = (
            [
                {
                    "policy": policy,
                    "conc": conc,
                    "gamma": gamma,
                    "seed": r["meta"].get("seed"),
                    "m": r["metrics"],
                }
                for r in seed_latest
            ]
            if per_seed
            else [
                {
                    "policy": policy,
                    "conc": conc,
                    "gamma": gamma,
                    "seed": None,
                    "m": averaged_cell(runs),
                }
            ]
        )
        by_conc.setdefault(conc, []).extend(replicas)

    rows = []
    for conc in sorted(by_conc):
        group = by_conc[conc]
        lpm_rows = [g for g in group if g["policy"] == "lpm"]
        lpm_m = lpm_rows[0]["m"] if lpm_rows else None
        for g in group:
            g["lpm_m"] = lpm_m
        group.sort(
            key=lambda g: (
                FINAL_POLICY_ORDER.get(g["policy"], 9),
                g["gamma"] if g["gamma"] is not None else math.inf,
                g["seed"] if g["seed"] is not None else math.inf,
            )
        )
        rows.extend(group)
    return rows


def final_table(cells: dict[tuple, list[dict]], per_seed: bool = False) -> str:
    header = (
        "| Policy | c | γ | n | hit | q_p95 (s) | q_p99 (s) | starv | cold n |"
        " cold starv | warm n | warm starv | gap99 (s) | RPM | oTPM | TTFT (ms) |"
        " TPOT p95 (ms) | retr | q_p99 Δ vs LPM | Anti-starv. vs LPM |"
    )
    sep = "|---|" + "---|" * 19
    lines = [header, sep]

    for row in final_rows(cells, per_seed):
        m = row["m"]
        lpm_m = row["lpm_m"]
        policy = row["policy"]
        label = policy
        if policy == "lpm":
            label = "**lpm (baseline)**"
        if policy == "fcfs":
            label = "fcfs ‡"
        if row["seed"] is not None:
            label += f" (seed {row['seed']})"

        cold_n = dig(m, "queue_wait_per_request.cold_class.n") or 0
        warm_n = dig(m, "queue_wait_per_request.warm_class.n") or 0
        warm_starv = dig(m, "queue_wait_per_request.warm_class.starvation_rate")
        gap99 = dig(m, "queue_wait_per_request.cold_warm_queue_wait_gap.p99_s")

        q_p99 = dig(m, "queue.queue_time_p99_s")
        starv = dig(m, "queue.starvation_rate")
        lpm_q_p99 = dig(lpm_m, "queue.queue_time_p99_s") if lpm_m else None
        lpm_starv = dig(lpm_m, "queue.starvation_rate") if lpm_m else None
        is_baseline = policy == "lpm"

        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row["conc"]),
                    fmt_gamma(row["gamma"]),
                    str(dig(m, "n_complete_records") or 0),
                    fmt(dig(m, "queue_wait_per_request.first_attempt_cache_hit_rate")),
                    fmt(dig(m, "queue.queue_time_p95_s")),
                    fmt(q_p99),
                    fmt(starv),
                    str(cold_n),
                    fmt(dig(m, "queue_wait_per_request.cold_class.starvation_rate")),
                    str(warm_n),
                    fmt(warm_starv) if warm_n else EM,
                    signed_fmt(gap99) if (warm_n and cold_n) else EM,
                    fmt(dig(m, "throughput.request_per_min"), ".1f"),
                    fmt(dig(m, "throughput.output_tokens_per_min"), ".0f"),
                    fmt(dig(m, "ttft.mean_ms"), ".0f"),
                    fmt(dig(m, "tpot.p95_ms"), ".1f"),
                    fmt(dig(m, "queue_wait_per_request.retractions.value"), ".0f"),
                    EM if is_baseline else q_p99_delta(q_p99, lpm_q_p99),
                    EM if is_baseline else anti_starvation_delta(starv, lpm_starv),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def best_c16_boost(cells: dict[tuple, list[dict]]) -> tuple | None:
    """The c16 boost cell with the lowest cold-class starvation (footnote c's
    boost cell; not the frontier pick gamma*, which rests on hit / TPOT p95 /
    the offline gamma-hat region, not on cold starvation alone)."""
    c16 = [
        (key, averaged_cell(runs))
        for key, runs in cells.items()
        if key[0] == "boost" and key[1] == 16
    ]
    if not c16:
        return None
    return min(
        c16,
        key=lambda km: (
            dig(km[1], "queue_wait_per_request.cold_class.starvation_rate")
            if dig(km[1], "queue_wait_per_request.cold_class.starvation_rate")
            is not None
            else math.inf,
            km[0][2] or 0.0,
        ),
    )[0]


def footnotes(cells: dict[tuple, list[dict]]) -> list[str]:
    lpm16 = latest(cells).get(("lpm", 16, None, None))
    fcfs16 = latest(cells).get(("fcfs", 16, None, None))
    best = best_c16_boost(cells)
    best16 = latest(cells).get(best) if best else None

    def rel(a: float | None, b: float | None) -> str:
        if a is None or b in (None, 0):
            return "n/a"
        return f"{(a - b) / b * 100:+.1f}%"

    lpm_starv16 = dig(lpm16["metrics"], "queue.starvation_rate") if lpm16 else None
    fcfs_starv16 = dig(fcfs16["metrics"], "queue.starvation_rate") if fcfs16 else None
    boost_starv16 = dig(best16["metrics"], "queue.starvation_rate") if best16 else None
    gamma_s = fmt_gamma(best[2]) if best else "n/a"
    cold16 = (
        dig(best16["metrics"], "queue_wait_per_request.cold_class.starvation_rate")
        if best16
        else None
    )
    lpm_cold16 = (
        dig(lpm16["metrics"], "queue_wait_per_request.cold_class.starvation_rate")
        if lpm16
        else None
    )
    fcfs_cold16 = (
        dig(fcfs16["metrics"], "queue_wait_per_request.cold_class.starvation_rate")
        if fcfs16
        else None
    )

    return [
        f"**Footnotes**\n\n(a) ¹ Anti-starvation is relative to the LPM baseline of the same concurrency; a relative change is undefined when the baseline is 0, so the cell prints the absolute point change instead (`−X.X pp`). The starvation rate counts queue waits > 10 s and is blind below that threshold, so at c4/c8 (waits ≤ 4 s) the rate column stays 0.000 for every arm -- the c4 evidence is the cold-vs-warm queue-wait gap (LPM +6.39 s → boost ~0.00 s, eliminated) and q_p99 −99.7% at hit parity.",
        f"(b) ‡ fcfs confound: fcfs leaves the sort-time prefix-match bookkeeping at zero for this deployment's cache (every current implementation returns False from `supports_fast_match_prefix()`), so its hit column reads 0.000 even though admission-time matching gives it real cache reuse (Task 1's verdict); fcfs-match is the reference arm with the bookkeeping forced on. fcfs still measures FCFS *ordering*, which is what boost degenerates toward at large γ.",
        f"(c) Cold-class starvation at c16: boost (γ={gamma_s}) {fmt(cold16)} vs LPM {fmt(lpm_cold16)} ({rel(cold16, lpm_cold16)} relative); fcfs {fmt(fcfs_cold16)} ({rel(fcfs_cold16, lpm_cold16)}). Overall starvation at c16: LPM {fmt(lpm_starv16)}, boost {fmt(boost_starv16)}, fcfs {fmt(fcfs_starv16)}.",
        "(d) Metric definitions: starv = share of requests with queue wait > 10 s; gap99 = cold-class p99 − warm-class p99 queue wait (schedule-eval log, first admission stamp); hit = token-weighted first-attempt prefix hit rate; q_p95/p99 = queue-wait percentiles (server histogram); TTFT = client-observed mean; retr = retraction count. Seed replicas average into one n-weighted row.",
    ]


# ---------------------------------------------------------------------------
# Flat table (round 1) and calibration
# ---------------------------------------------------------------------------


def flat_table(cells: dict[tuple, list[dict]]) -> None:
    singles = latest(cells)
    superseded = sum(len(runs) - 1 for runs in cells.values())
    if superseded:
        print(f"# {superseded} superseded same-cell run dir(s) ignored (latest wins)\n")
    hdr = (
        f"{'policy':10s} {'c':>2s} {'gamma':>6s} | {'n':>4s} {'hit':>5s} "
        f"| {'q_p95':>6s} {'q_p99':>6s} {'starv':>5s} | {'cold_n':>6s} {'c_strv':>6s} "
        f"{'warm_n':>6s} {'w_strv':>6s} {'gap99':>6s} | {'rpm':>5s} {'oTPM':>6s} "
        f"{'ttft':>6s} {'tpot95':>6s} {'retr':>4s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(singles, key=cell_sort_key):
        policy, conc, gamma, k = key
        m = singles[key]["metrics"]
        q = m.get("queue_wait_per_request", {})
        queue = m.get("queue", {})
        thr = m.get("throughput", {})
        retr = q.get("retractions", {}).get("value")
        print(
            f"{policy:10s} {conc:2d} {fmt(gamma, '.0e') if gamma else '-':>6s} | "
            f"{m.get('n_complete_records', 0):4d} {fmt(q.get('first_attempt_cache_hit_rate'), '.3f'):>5s} | "
            f"{fmt(queue.get('queue_time_p95_s')):>6s} {fmt(queue.get('queue_time_p99_s')):>6s} "
            f"{fmt(queue.get('starvation_rate'), '.3f'):>5s} | "
            f"{q.get('cold_class', {}).get('n', 0):6d} {fmt(q.get('cold_class', {}).get('starvation_rate'), '.3f'):>6s} "
            f"{q.get('warm_class', {}).get('n', 0):6d} {fmt(q.get('warm_class', {}).get('starvation_rate'), '.3f'):>6s} "
            f"{fmt(q.get('cold_warm_queue_wait_gap', {}).get('p99_s')):>6s} | "
            f"{fmt(thr.get('request_per_min'), '.1f'):>5s} {fmt(thr.get('output_tokens_per_min'), '.0f'):>6s} "
            f"{fmt(m.get('ttft', {}).get('mean_ms'), '.0f'):>6s} "
            f"{fmt(m.get('tpot', {}).get('p95_ms'), '.1f'):>6s} {fmt(retr, '.0f'):>4s}"
        )
    print(
        "\nlegend: hit=first-attempt token-weighted prefix hit rate; q_p95/p99/starv ="
        " queue-wait percentiles + >10s share (server histogram); c/w_strv = cold/warm"
        " per-class starvation (schedule-eval log); gap99 = cold p99 - warm p99 queue"
        " wait; retr = retraction count (gauge sum). Units: s except rpm/oTPM."
    )


def b_gamma(gamma: float, w: float) -> float:
    """Boost bound b_gamma(w) = -ln(1 - exp(-gamma*w))/gamma (expm1-stable)."""
    x = gamma * w
    if x <= 0:  # w <= 0: no bound
        return math.inf
    v = -math.expm1(-x)  # 1 - exp(-x), in (0, 1]
    if v >= 1.0:
        return 0.0
    return -math.log(v) / gamma


def calibration_table(cells: dict[tuple, list[dict]], k: int = 256) -> None:
    print("\ncalibration (R = fcfs total-token throughput; bound = b_gamma(k)/R):")
    r_cells = latest(cells)
    r_runs = [(key, r["metrics"]) for key, r in r_cells.items() if key[0] == "fcfs"]
    if not r_runs:
        print("  no fcfs runs")
        return
    for key, m in sorted(r_runs, key=lambda km: cell_sort_key(km[0])):
        r = m.get("throughput", {}).get("total_token_throughput_tps")
        print(f"  fcfs c{key[1]}: R = {fmt(r, '.0f')} tok/s")
    r = max(
        (
            m.get("throughput", {}).get("total_token_throughput_tps", 0.0)
            for _, m in r_runs
        ),
        default=0.0,
    )
    print(f"  R_max = {fmt(r, '.0f')} tok/s; k = {k}")
    for g in SWEEP_GAMMAS:
        gamma = float(g)
        bound_tokens = b_gamma(gamma, k)
        print(
            f"  gamma={g}: b_gamma(k) = {bound_tokens:.0f} tokens -> ~{bound_tokens / r:.1f} s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_root",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "results" / "agentx",
    )
    parser.add_argument("--flat", action="store_true", help="round-1 flat table")
    parser.add_argument(
        "--calibration", action="store_true", help="append the calibration table"
    )
    parser.add_argument(
        "--per-seed",
        action="store_true",
        help="final table: one row per seed replica instead of the n-weighted average",
    )
    args = parser.parse_args()

    cells = load_cells(args.results_root)
    if not cells:
        sys.exit(f"no e2_metrics.json under {args.results_root}")

    if args.flat:
        flat_table(cells)
        if args.calibration:
            calibration_table(cells)
        return

    print(final_table(cells, per_seed=args.per_seed))
    print()
    print("\n\n".join(footnotes(cells)))
    if args.calibration:
        print()
        calibration_table(cells)


if __name__ == "__main__":
    main()
