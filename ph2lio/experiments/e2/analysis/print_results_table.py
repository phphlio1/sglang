"""Print the E2 policy-comparison table from per-run e2_metrics.json files.

One row per (policy, concurrency, gamma, k) cell -- the latest run dir wins
(earlier same-cell dirs are counted as superseded). Two tables:

  results   -- headline queue/starvation metrics + spec guards
  calibration -- R per fcfs run and the boost starvation bound b_gamma(k)/R

Pure reporting: reads JSON, prints text. No server contact.
"""

import json
import math
import sys
from pathlib import Path

# Display order matches the matrix driver (run_boost_matrix.sh).
POLICY_ORDER = {"fcfs": 0, "lpm": 1, "boost": 2, "hrrn": 3, "dfs-weight": 4, "lof": 5}
SWEEP_GAMMAS = ("5e-5", "1e-4", "2e-4", "5e-4")


def load_cells(results_root: Path) -> dict[tuple, dict]:
    """(policy, conc, gamma, k) -> metrics of the latest run dir for that cell."""
    cells: dict[tuple, dict] = {}
    superseded = 0
    for metrics_path in sorted(results_root.glob("*/*/e2_metrics.json")):
        run_dir = metrics_path.parent
        # Smoke runs have no run_meta.json and are never comparison cells.
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        policy = meta["policy"]
        conc = int(meta["concurrency"])
        gamma, k = (
            (meta["boost_gamma"], meta["boost_bin_size"])
            if policy == "boost"
            else (None, None)
        )
        key = (policy, conc, gamma, k)
        if key in cells:
            superseded += 1
        cells[key] = json.loads(metrics_path.read_text())
    if superseded:
        print(f"# {superseded} superseded same-cell run dir(s) ignored (latest wins)\n")
    return cells


def cell_sort_key(key: tuple) -> tuple:
    policy, conc, gamma, k = key
    gamma_s = f"{gamma:.10g}" if gamma is not None else ""
    return (POLICY_ORDER.get(policy, 99), conc, gamma_s, k or 0)


def fmt(v, spec=".3f") -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    return format(v, spec)


def results_table(cells: dict) -> None:
    hdr = (
        f"{'policy':10s} {'c':>2s} {'gamma':>6s} | {'n':>4s} {'hit':>5s} "
        f"| {'q_p95':>6s} {'q_p99':>6s} {'starv':>5s} | {'cold_n':>6s} {'c_strv':>6s} "
        f"{'warm_n':>6s} {'w_strv':>6s} {'gap99':>6s} | {'rpm':>5s} {'oTPM':>6s} "
        f"{'ttft':>6s} {'tpot95':>6s} {'retr':>4s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(cells, key=cell_sort_key):
        policy, conc, gamma, k = key
        m = cells[key]
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


def calibration_table(cells: dict, k: int = 256) -> None:
    print("\ncalibration (R = fcfs total-token throughput; bound = b_gamma(k)/R):")
    r_runs = [(key, m) for key, m in cells.items() if key[0] == "fcfs"]
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
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/agentx")
    cells = load_cells(root)
    if not cells:
        sys.exit(f"no e2_metrics.json under {root}")
    results_table(cells)
    calibration_table(cells)


if __name__ == "__main__":
    main()
