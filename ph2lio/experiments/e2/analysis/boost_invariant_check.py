#!/usr/bin/env python3
"""Task-3 post-hoc implementation invariants over schedule-eval logs.

1. Overtaking invariant (boost runs only): for any two requests where x
   entered the queue earlier in the token clock (A(x) < A(y)) but y was
   admitted first, the later arrival's head start is bounded by the max
   boost: A(y) - A(x) <= b_gamma(k). Any HARD violation is an implementation
   bug (sort key or stamp-once arrival). Pairs are classified:
     - same-batch: identical admission stamps (within-batch placement, not
       overtaking);
     - dedup-follower: the overtaker was an in-batch-dedup follower at its
       first admission (new logs; followers sort last by design, so they can
       be admitted after anyone -- exempt);
     - deep-match (legacy logs): the overtaken request had a deep admission
       match; suspected dedup deprioritization / host-hit load-back gating,
       not distinguishable without the follower field;
     - hard: none of the above -> bug.
   Admission order is arrival_wall_s + queue_wait_s (both stamp-once), so
   retraction cycles cannot fake order.
2. u bimodality: uncached work u = input + output - matched at first
   admission, cold vs warm (warm = matched > 0). The boost design assumes
   warm deltas cluster well below cold work.
3. c8 p99 pinning (3.600 s identically across gamma): top-10 queue waits per
   run with their context, to attribute the tail to chunked-prefill blocking
   (a decode-vs-prefill interleave effect) rather than policy.

Usage: boost_invariant_check.py [results_root] [--run-dir DIR ...]
Writes boost_invariant_check.json under the results root.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from agentx_metrics import load_run_meta, load_schedule_eval_records
from starvation import _percentile

# Admission stamps of batch mates come from one scheduler iteration; anything
# closer than this in wall time is within-batch placement, not overtaking.
SAME_BATCH_EPS_S = 0.05


def gamma_boost(work: float, gamma: float) -> float:
    """b(w) = -ln(1 - exp(-gamma*w))/gamma (matches schedule_policy.gamma_boost)."""
    x = gamma * work
    if x <= 0:
        return float("inf")
    v = -math.expm1(-x)
    if v >= 1.0:
        return 0.0
    return -math.log(v) / gamma


def _num(v) -> bool:
    return isinstance(v, (int, float))


def check_overtaking(records: list[dict], gamma: float, bin_size: int) -> dict:
    bound = gamma_boost(bin_size, gamma)
    usable = [
        r
        for r in records
        if _num(r.get("boost_arrival_tokens"))
        and _num(r.get("arrival_wall_s"))
        and _num(r.get("queue_wait_s"))
    ]
    for r in usable:
        r["_admission_s"] = r["arrival_wall_s"] + r["queue_wait_s"]
    ordered = sorted(usable, key=lambda r: r["_admission_s"])

    pairs = 0
    overtakes = 0
    same_batch = 0
    exempt_dedup = 0
    legacy_deep_match = 0
    hard: list[dict] = []
    for i, y in enumerate(ordered):
        # y admitted at position i; every x admitted after y with an earlier
        # token-clock arrival was overtaken by y.
        for x in ordered[i + 1 :]:
            if x["boost_arrival_tokens"] >= y["boost_arrival_tokens"]:
                continue
            pairs += 1
            gap = y["boost_arrival_tokens"] - x["boost_arrival_tokens"]
            if gap <= bound:
                overtakes += 1
                continue
            if abs(x["_admission_s"] - y["_admission_s"]) < SAME_BATCH_EPS_S:
                same_batch += 1
                continue
            if y.get("in_batch_dedup_follower_first_admission"):
                exempt_dedup += 1
                continue
            if (x.get("matched_prefix_tokens_first_admission") or 0) > 0:
                legacy_deep_match += 1
                continue
            hard.append(
                {
                    "overtaker_rid": y.get("rid"),
                    "overtaker_arrival_tokens": y["boost_arrival_tokens"],
                    "overtaker_admission_s": y["_admission_s"],
                    "overtaker_matched": y.get("matched_prefix_tokens_first_admission"),
                    "overtaken_rid": x.get("rid"),
                    "overtaken_arrival_tokens": x["boost_arrival_tokens"],
                    "overtaken_admission_s": x["_admission_s"],
                    "overtaken_matched": x.get("matched_prefix_tokens_first_admission"),
                    "overtaken_retractions": x.get("num_retractions"),
                    "gap_tokens": gap,
                    "bound_tokens": bound,
                }
            )
    return {
        "bound_tokens": bound,
        "n_usable_records": len(usable),
        "n_pairs_earlier_arrival": pairs,
        "n_overtakes_within_bound": overtakes,
        "n_beyond_bound": same_batch + exempt_dedup + legacy_deep_match + len(hard),
        "beyond_bound_same_batch": same_batch,
        "beyond_bound_exempt_dedup_follower": exempt_dedup,
        "beyond_bound_legacy_deep_match": legacy_deep_match,
        "hard_violations": hard,
        "pass": not hard,
    }


def check_u_bimodality(records: list[dict]) -> dict:
    rows = [r for r in records if isinstance(r.get("input_len"), int)]
    if not rows:
        return {"n_records": 0}

    def u(r: dict) -> int:
        matched = r.get("matched_prefix_tokens_first_admission") or 0
        return r.get("input_len", 0) + r.get("output_len", 0) - matched

    warm = [
        u(r) for r in rows if (r.get("matched_prefix_tokens_first_admission") or 0) > 0
    ]
    cold = [
        u(r) for r in rows if not (r.get("matched_prefix_tokens_first_admission") or 0)
    ]
    stats = lambda v: (
        {
            "n": len(v),
            "mean": sum(v) / len(v) if v else None,
            "median": _percentile(sorted(v), 50) if v else None,
            "p10": _percentile(sorted(v), 10) if v else None,
            "p95": _percentile(sorted(v), 95) if v else None,
        }
        if v
        else {"n": 0}
    )
    warm_s, cold_s = stats(warm), stats(cold)
    cold_p10 = cold_s.get("p10")
    return {
        "n_records": len(rows),
        "u_warm": warm_s,
        "u_cold": cold_s,
        # The design assumption: warm deltas sit well below cold work. The
        # fraction of warm u below the cold p10 quantifies the separation;
        # warm_median_over_cold_p10 << 1 is the clean statement.
        "frac_warm_below_cold_p10": (
            sum(1 for v in warm if cold_p10 is not None and v < cold_p10) / len(warm)
            if warm and cold_p10 is not None
            else None
        ),
        "warm_median_over_cold_p10": (
            warm_s["median"] / cold_p10
            if warm_s.get("median") is not None and cold_p10
            else None
        ),
    }


def check_tail_waits(records: list[dict], top: int = 10) -> dict:
    usable = [r for r in records if _num(r.get("queue_wait_s"))]
    if not usable:
        return {"n_records": 0}
    top_waits = sorted(usable, key=lambda r: -r["queue_wait_s"])[:top]
    waits = sorted(r["queue_wait_s"] for r in usable)
    return {
        "n_records": len(usable),
        "p99_s": _percentile(waits, 99),
        "max_s": waits[-1],
        "n_at_p99_within_1pct": sum(
            1 for w in waits if w >= _percentile(waits, 99) * 0.99
        ),
        "top": [
            {
                "rid": r.get("rid"),
                "queue_wait_s": r["queue_wait_s"],
                "input_len": r.get("input_len"),
                "matched_first_admission": r.get(
                    "matched_prefix_tokens_first_admission"
                ),
                "num_retractions": r.get("num_retractions"),
                "arrival_wall_s": r.get("arrival_wall_s"),
            }
            for r in top_waits
        ],
    }


def discover_boost_runs(results_root: Path) -> list[Path]:
    return sorted(
        (
            d
            for d in (results_root / "boost").glob("c*_*")
            if (d / "run_meta.json").exists()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_root",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "results" / "agentx",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="check only these run dirs (default: every boost run dir)",
    )
    parser.add_argument(
        "--all-policies",
        action="store_true",
        help="also run the u-bimodality / tail checks on non-boost runs",
    )
    args = parser.parse_args()

    run_dirs = args.run_dir or discover_boost_runs(args.results_root)
    report: dict = {"results_root": str(args.results_root), "runs": []}
    overall_pass = True
    for run_dir in run_dirs:
        meta = load_run_meta(run_dir)
        records, _ = load_schedule_eval_records(run_dir)
        if records is None:
            continue
        policy = meta.get("policy") or run_dir.parent.name
        entry = {
            "run_dir": str(run_dir),
            "policy": policy,
            "concurrency": meta.get("concurrency"),
            "gamma": meta.get("boost_gamma"),
            "bin_size": meta.get("boost_bin_size"),
        }
        if policy == "boost":
            inv = check_overtaking(
                records,
                float(meta.get("boost_gamma") or 1e-4),
                int(meta.get("boost_bin_size") or 256),
            )
            entry["overtaking_invariant"] = inv
            overall_pass &= inv["pass"]
            verdict = (
                "PASS" if inv["pass"] else f"FAIL ({len(inv['hard_violations'])} hard)"
            )
            print(
                f"{run_dir.name}: overtaking {verdict}"
                f"  pairs={inv['n_pairs_earlier_arrival']}"
                f"  within-bound={inv['n_overtakes_within_bound']}"
                f"  beyond: same-batch={inv['beyond_bound_same_batch']}"
                f" dedup={inv['beyond_bound_exempt_dedup_follower']}"
                f" legacy-deep={inv['beyond_bound_legacy_deep_match']}"
                f" hard={len(inv['hard_violations'])}"
            )
        elif not args.all_policies:
            continue
        entry["u_bimodality"] = check_u_bimodality(records)
        entry["tail_waits"] = check_tail_waits(records)
        report["runs"].append(entry)

    report["overtaking_invariant_all_pass"] = overall_pass
    out_path = args.results_root / "boost_invariant_check.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\noverall overtaking invariant: {'PASS' if overall_pass else 'FAIL'}")
    print(f"report -> {out_path}")


if __name__ == "__main__":
    main()
