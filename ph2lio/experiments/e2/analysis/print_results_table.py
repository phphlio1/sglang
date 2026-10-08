#!/usr/bin/env python3
"""Print the E2 policy-comparison tables from per-run e2_metrics.json files.

Results are counted PER FULL RUN: the cells of one sweep-driver invocation,
identified by the sglang_commit stamp every cell of a sweep shares (the docs
label sweeps the same way -- "6/6 cells under commit 0083303206"). One block
per full run, latest first, so the sweep a driver just landed prints first
(the full-run drivers call this script at sweep end). The block heading's
date is a label read off the cells' timestamps, not a filter. Each block:
one ### cN section per concurrency, in the docs/full_run.md table format
(percent displays, TTFT/TPOT avg/p95/p99, Jain, TPM in M/min) plus a gamma
column (the clock/scale disambiguator for boost rows) and two computed delta
columns against the lpm baseline of the same table. Footnotes (a)-(e) print
at the end; footnote (c) prints under each full run that holds c16
lpm/fcfs/boost cells. Within a cell the latest run per seed is kept (a
fresh-boot re-run supersedes an older same-seed run of that cell); seeds
then average into one n-weighted row by default (--per-seed emits one row
per seed).

--flat prints the round-1 flat table (one row per cell across all full
runs, latest run wins), --calibration the R / b_gamma(k) calibration table.

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
# Within a concurrency group (full_run.md arm order): lpm baseline first,
# hrrn, boost by gamma, fcfs-match, fcfs, then anything else.
FINAL_POLICY_ORDER = {"lpm": 0, "hrrn": 1, "boost": 2, "fcfs-match": 3, "fcfs": 4, "dfs-weight": 5, "lof": 6}

EM = "—"  # em dash: no value / not applicable
MINUS = "−"  # minus sign, matches the doc tables


# The harness records boost_gamma null when the flag was left to the
# clock-mode default (1e-4 tokens / 0.1 1/s wall); float(None) would crash.
CLOCK_DEFAULT_GAMMA = {"tokens": 1e-4, "wall": 0.1}


def boost_gamma_resolved(meta: dict) -> float:
    gamma = meta["boost_gamma"]
    if gamma is not None:
        return float(gamma)
    return CLOCK_DEFAULT_GAMMA.get(meta.get("boost_clock", "tokens"), 1e-4)


# ---------------------------------------------------------------------------
# Loading and full-run grouping
# ---------------------------------------------------------------------------


def load_runs(results_root: Path) -> list[dict]:
    """[{meta, metrics}] for every run dir holding run_meta.json (smoke runs
    record neither file and never appear)."""
    runs = []
    for metrics_path in sorted(results_root.glob("*/*/e2_metrics.json")):
        meta_path = metrics_path.parent / "run_meta.json"
        if not meta_path.exists():
            continue
        runs.append(
            {
                "meta": json.loads(meta_path.read_text()),
                "metrics": json.loads(metrics_path.read_text()),
            }
        )
    return runs


def cell_key(meta: dict) -> tuple:
    """(policy, conc, gamma, k): the comparison cell a run belongs to."""
    policy = meta["policy"]
    gamma, k = (
        (boost_gamma_resolved(meta), int(meta["boost_bin_size"]))
        if policy == "boost"
        else (None, None)
    )
    return (policy, int(meta["concurrency"]), gamma, k)


def group_full_runs(runs: list[dict]) -> list[dict]:
    """One entry per full run -- the cells sharing a sweep's sglang_commit
    stamp -- latest first (the sweep a driver just landed prints first)."""
    by_commit: dict[str, dict[tuple, list[dict]]] = {}
    for run in runs:
        meta = run["meta"]
        by_commit.setdefault(meta["sglang_commit"], {}).setdefault(
            cell_key(meta), []
        ).append(run)

    groups = []
    for commit, cells in by_commit.items():
        ts = [
            r["meta"].get("timestamp") or ""
            for cell_runs in cells.values()
            for r in cell_runs
        ]
        first, last = min(ts)[:10], max(ts)[:10]
        groups.append(
            {
                "commit": commit,
                # Heading label only; runs are grouped by commit, never by date.
                "date": first if first == last else f"{first}..{last}",
                "last_ts": max(ts),
                "n_cells": len(cells),
                "cells": cells,
            }
        )
    groups.sort(key=lambda g: g["last_ts"], reverse=True)
    return groups


def merge_cells(groups: list[dict]) -> dict[tuple, list[dict]]:
    """All full runs' cells in one {cell key -> [runs]} map (flat table)."""
    cells: dict[tuple, list[dict]] = {}
    for g in groups:
        for key, cell_runs in g["cells"].items():
            cells.setdefault(key, []).extend(cell_runs)
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


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def fmt(v, spec=".3f") -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    return format(v, spec)


def pct(v, spec=".2f") -> str:
    """Fraction -> percent display (full_run.md's hit/starv convention)."""
    if v is None:
        return EM
    if isinstance(v, float) and math.isnan(v):
        return EM
    return format(v * 100.0, spec) + "%"


def fmt_s(ms: float | None) -> str:
    """Milliseconds -> seconds, full_run.md's TTFT display."""
    if ms is None:
        return EM
    return fmt(ms / 1000.0, ".2f")


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


def dig(d: dict, path: str):
    for part in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


# ---------------------------------------------------------------------------
# Final table (per full run)
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
        "throughput.total_token_throughput_tps",
        "ttft.mean_ms",
        "ttft.p95_ms",
        "ttft.p99_ms",
        "tpot.mean_ms",
        "tpot.p95_ms",
        "tpot.p99_ms",
        "fairness.ttft_fairness_jain",
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


def q_p99_delta(cfg_q_p99: float | None, lpm_q_p99: float | None) -> str:
    if cfg_q_p99 is None or lpm_q_p99 in (None, 0):
        return EM
    # From the displayed .3f values so a reader recomputing the delta from the
    # printed columns gets the printed answer.
    cfg_r, lpm_r = round(cfg_q_p99, 3), round(lpm_q_p99, 3)
    return signed_pct((cfg_r - lpm_r) / lpm_r * 100.0, ".2f")


def starv_vs_lpm_delta(cfg_starv: float | None, lpm_starv: float | None) -> str:
    if cfg_starv is None or lpm_starv is None:
        return EM
    # Same displayed-value rule as q_p99_delta, at the printed precision
    # (starv prints as a 2-dp percent = 4-dp fraction); the zero-baseline
    # check reads the displayed value too (a 0.01% baseline is degenerate).
    cfg_r, lpm_r = round(cfg_starv, 4), round(lpm_starv, 4)
    if lpm_r == 0:
        # Relative change is undefined against a zero baseline.
        if cfg_r == 0:
            return "0% ¹"
        pp = (cfg_r - lpm_r) * 100.0
        sign = "+" if pp >= 0 else MINUS
        return f"{sign}{abs(pp):.1f} pp ¹"
    return signed_pct((cfg_r - lpm_r) / lpm_r * 100.0, ".2f")


def final_rows(cells: dict[tuple, list[dict]], per_seed: bool) -> list[dict]:
    """Ordered row dicts: one per (cell [, seed]) with lpm baselines attached."""
    by_conc: dict[int, list[dict]] = {}
    for (policy, conc, gamma, k), runs in cells.items():
        # Every run of a cell shares the harness config; only boost rows
        # render a clock, so the clock rides on boost rows alone.
        clock = runs[0]["meta"].get("boost_clock", "tokens") if policy == "boost" else None
        seed_latest = latest_per_seed(runs)
        replicas = (
            [
                {
                    "policy": policy,
                    "conc": conc,
                    "gamma": gamma,
                    "clock": clock,
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
                    "clock": clock,
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
    # full_run.md's columns and displays, plus the gamma column (the clock/
    # scale disambiguator for boost rows); one ### section per concurrency.
    header = (
        "| Policy | γ | n | retr | cache hit (%) | q_p95 (s) | q_p99 (s) | starv (%) |"
        " cold n | warm n | cold starv (%) | warm starv (%) | gap99 (s) | RPM | TPM (M) |"
        " TTFT avg (s) | TTFT p95 (s) | TTFT p99 (s) | TPOT avg (ms) | TPOT p95 (ms) |"
        " TPOT p99 (ms) | Jain | q_p99 Δ vs LPM | Starv vs LPM |"
    )
    sep = "|---|" + "---|" * 23

    out: list[str] = []
    conc_prev: int | None = None
    for row in final_rows(cells, per_seed):
        if row["conc"] != conc_prev:
            out.append(f"### c{row['conc']} — concurrency {row['conc']}")
            out.append("")
            out.append(header)
            out.append(sep)
            conc_prev = row["conc"]
        m = row["m"]
        lpm_m = row["lpm_m"]
        policy = row["policy"]
        hit = dig(m, "queue_wait_per_request.first_attempt_cache_hit_rate")
        label = policy
        if policy == "lpm":
            label = "**lpm (baseline)**"
        # ‡ marks the pre-fix bookkeeping confound (hit stuck at ~0 with real
        # reuse), not fcfs itself; post-fix cells read a real hit.
        if policy == "fcfs" and (hit or 0.0) < 0.01:
            label = "fcfs ‡"
        if row["seed"] is not None:
            label += f" (seed {row['seed']})"
        # † = wall-clock gamma in 1/s; unmarked boost gammas are per-token.
        gamma_cell = fmt_gamma(row["gamma"])
        if policy == "boost" and row["clock"] == "wall":
            gamma_cell += " †"

        cold_n = dig(m, "queue_wait_per_request.cold_class.n") or 0
        warm_n = dig(m, "queue_wait_per_request.warm_class.n") or 0
        gap99 = dig(m, "queue_wait_per_request.cold_warm_queue_wait_gap.p99_s")
        tps = dig(m, "throughput.total_token_throughput_tps")

        q_p99 = dig(m, "queue.queue_time_p99_s")
        starv = dig(m, "queue.starvation_rate")
        lpm_q_p99 = dig(lpm_m, "queue.queue_time_p99_s") if lpm_m else None
        lpm_starv = dig(lpm_m, "queue.starvation_rate") if lpm_m else None
        is_baseline = policy == "lpm"

        out.append(
            "| "
            + " | ".join(
                [
                    label,
                    gamma_cell,
                    str(dig(m, "n_complete_records") or 0),
                    fmt(dig(m, "queue_wait_per_request.retractions.value"), ".0f"),
                    pct(hit),
                    fmt(dig(m, "queue.queue_time_p95_s")),
                    fmt(q_p99),
                    pct(starv),
                    str(cold_n),
                    str(warm_n),
                    pct(dig(m, "queue_wait_per_request.cold_class.starvation_rate"))
                    if cold_n
                    else EM,
                    pct(dig(m, "queue_wait_per_request.warm_class.starvation_rate"))
                    if warm_n
                    else EM,
                    signed_fmt(gap99, ".2f") if (warm_n and cold_n) else EM,
                    fmt(dig(m, "throughput.request_per_min"), ".1f"),
                    fmt(tps * 60.0 / 1e6, ".3f") if tps is not None else EM,
                    fmt_s(dig(m, "ttft.mean_ms")),
                    fmt_s(dig(m, "ttft.p95_ms")),
                    fmt_s(dig(m, "ttft.p99_ms")),
                    fmt(dig(m, "tpot.mean_ms"), ".2f"),
                    fmt(dig(m, "tpot.p95_ms"), ".2f"),
                    fmt(dig(m, "tpot.p99_ms"), ".2f"),
                    fmt(dig(m, "fairness.ttft_fairness_jain"), ".3f"),
                    EM if is_baseline else q_p99_delta(q_p99, lpm_q_p99),
                    EM if is_baseline else starv_vs_lpm_delta(starv, lpm_starv),
                ]
            )
            + " |"
        )
    return "\n".join(out)


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


def c16_note(cells: dict[tuple, list[dict]]) -> str | None:
    """Footnote (c) for one full run: the c16 cold-vs-baseline starvation
    comparison. Only a run holding c16 lpm, fcfs and boost cells carries it."""
    singles = latest(cells)
    lpm16 = singles.get(("lpm", 16, None, None))
    fcfs16 = singles.get(("fcfs", 16, None, None))
    best = best_c16_boost(cells)
    if not (lpm16 and fcfs16 and best):
        return None
    best16 = singles.get(best)

    def rel(a: float | None, b: float | None) -> str:
        if a is None or b in (None, 0):
            return "n/a"
        return f"{(a - b) / b * 100:+.1f}%"

    def starv(run: dict | None) -> float | None:
        return dig(run["metrics"], "queue.starvation_rate") if run else None

    def cold(run: dict | None) -> float | None:
        return (
            dig(run["metrics"], "queue_wait_per_request.cold_class.starvation_rate")
            if run
            else None
        )

    cold16, lpm_cold16, fcfs_cold16 = cold(best16), cold(lpm16), cold(fcfs16)
    return (
        f"(c) Cold-class starvation at c16 (this full run): boost (γ={fmt_gamma(best[2])})"
        f" {pct(cold16)} vs LPM {pct(lpm_cold16)} ({rel(cold16, lpm_cold16)} relative);"
        f" fcfs {pct(fcfs_cold16)} ({rel(fcfs_cold16, lpm_cold16)}). Overall starvation"
        f" at c16: LPM {pct(starv(lpm16))}, boost {pct(starv(best16))},"
        f" fcfs {pct(starv(fcfs16))}."
    )


def footnotes(cells: dict[tuple, list[dict]]) -> list[str]:
    thresholds = sorted(
        {
            fmt(dig(r["metrics"], "queue.starvation_threshold_s"), ".4g")
            for runs in cells.values()
            for r in runs
            if dig(r["metrics"], "queue.starvation_threshold_s") is not None
        }
    )
    thresholds_str = " / ".join(f"{t} s" for t in thresholds) if thresholds else "per-cell"

    notes = [
        f"**Footnotes**\n\n(a) q_p99 Δ vs LPM and Starv vs LPM are relative to the same-table lpm baseline -- negative = better than baseline (lower queue tail / fewer stalled requests). ¹ the baseline's displayed starvation is 0.00%, so the relative change is undefined and the cell prints the absolute point change instead (±X.X pp). The starvation rate is blind below its per-cell threshold (footnote d).",
        "(b) ‡ marks fcfs rows whose hit column reads ~0.00%: the pre-fix confound, where fcfs recorded no admission-time prefix-match bookkeeping, so real cache reuse showed as zero (admission-time matching does give fcfs reuse; fcfs-match stays the reference arm with sort-time matching forced on). Post-fix cells (commit bdc67a5720 and later) read a real hit -- the c16 validation sweep reads 78.5% -- and stay unmarked. fcfs still measures FCFS *ordering*, which is what boost degenerates toward at large γ.",
        f"(d) Metric definitions: cache hit = token-weighted first-attempt prefix hit rate; starv = share of requests with queue wait > the cell's recorded starvation threshold ({thresholds_str} across these cells; the rate is blind below it); q_p95/p99 = queue-wait percentiles (server histogram); gap99 = cold-class p99 − warm-class p99 queue wait (schedule-eval log, first admission stamp); RPM = requests/min; TPM (M) = total (input + output) tokens/min in millions; TTFT/TPOT = client-observed avg/p95/p99 (TTFT in s, TPOT in ms); Jain = TTFT fairness index (higher = fairer); retr = retraction count. Seed replicas average into one n-weighted row.",
    ]
    if any(
        key[0] == "boost" and runs[0]["meta"].get("boost_clock", "tokens") == "wall"
        for key, runs in cells.items()
    ):
        notes.append(
            "(e) † wall-clock gamma, in 1/s: the row runs `--boost-clock wall`, so gamma "
            "scales seconds of service (arrival stamped in wall seconds, work scaled by "
            "the measured token rate); unmarked boost gammas are tokens-clock (per token). "
            "The two scales are not comparable -- the cross-mode identity is "
            "γ_wall = γ_tokens * R (R = measured token rate)."
        )
    return notes


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
        # † = wall-clock gamma in 1/s; unmarked boost gammas are per-token.
        gamma_s = fmt(gamma, ".0e") if gamma else "-"
        if policy == "boost" and singles[key]["meta"].get("boost_clock", "tokens") == "wall":
            gamma_s += "†"
        q = m.get("queue_wait_per_request", {})
        queue = m.get("queue", {})
        thr = m.get("throughput", {})
        retr = q.get("retractions", {}).get("value")
        print(
            f"{policy:10s} {conc:2d} {gamma_s:>6s} | "
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
        " queue-wait percentiles + above-threshold share (per-cell"
        " starvation_threshold_s; server histogram); c/w_strv = cold/warm"
        " per-class starvation (schedule-eval log); gap99 = cold p99 - warm p99 queue"
        " wait; retr = retraction count (gauge sum). Units: s except rpm/oTPM;"
        " gamma† = wall-clock (1/s), unmarked gamma = per-token."
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
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "1x-rtx6000pro"
        / "agentx",
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

    runs = load_runs(args.results_root)
    if not runs:
        sys.exit(f"no e2_metrics.json under {args.results_root}")

    groups = group_full_runs(runs)
    all_cells = merge_cells(groups)

    if args.flat:
        flat_table(all_cells)
        if args.calibration:
            calibration_table(all_cells)
        return

    for g in groups:
        print(f"## Full run {g['date']} — commit {g['commit'][:10]} ({g['n_cells']} cells)")
        print()
        print(final_table(g["cells"], per_seed=args.per_seed))
        note = c16_note(g["cells"])
        if note:
            print()
            print(note)
        print()
    print("\n\n".join(footnotes(all_cells)))
    if args.calibration:
        print()
        calibration_table(all_cells)


if __name__ == "__main__":
    main()
