"""E2 bottleneck report: one run dir -> where the serving limit sits.

Reads the same per-run artifacts as agentx_metrics.py / metrics_v2.py
(run_meta.json, profile_export.jsonl, schedule_eval.jsonl,
server_metrics_export.jsonl) plus the Layer-1 gpu_samples.csv (optional;
corroborator only) and the boot's server log (fresh boot per cell, so every
Decode-batch line belongs to this run). Writes bottleneck_report.json into
the run dir. Never touches e2_metrics.json / e2_metrics_v2.json.

Stage-1 has no external router: requests go client -> TokenizerManager ->
DataParallelController (single-threaded, load-blind round_robin) ->
schedulers. "Router bottleneck" here therefore means DPC dispatch
imbalance, measured per dp_rank label (Q4).

Decision tree (symptom -> hypothesis -> proof):
  Q1 GPU busy fraction = sum(delta scheduler_stage_seconds{category=
  run_batch}) /
     (window_s * n_ranks); < BUSY_NOT_GPU_BOUND -> not GPU-bound, sub-split
     by waiting-queue depth and scheduler idle/CPU rates; >= BUSY_GPU_BOUND
     -> GPU-bound, continue at Q2/Q6.
  Q2 client ITL p50 vs server TPOT p50 (schedule_eval e2e/ttft): ratio >
     ITL_TPOT_GAP -> detokenizer/streaming path (the gap IS the evidence;
     the detokenizer exports no metrics of its own).
  Q3 TTFT decomposition: client TTFT vs tokenize-stage p50 + server ttft
     (queue wait + prefill); residual share > TTFT_RESIDUAL_SHARE ->
     HTTP/TokenizerManager/DPC dispatch overhead.
  Q4 per-dp_rank num_running_reqs CV across ranks > RANK_CV_IMBALANCE ->
     round-robin load-blindness (in-tree stage-2 fix: --load-balance-method
     total_tokens; report only).
  Q5 token_usage max + retraction share + retraction counter -> KV-bound.
  Q6 mean spec accept len from the server log < ACCEPT_LEN_LOW ->
     EAGLE overhead dominates decode.
  Q7 hicache host-hit share of prefill-effective tokens + cache_hit_rate
     trend.

Usage:
    python3 analysis/bottleneck_report.py <run_dir> [--json]
    python3 analysis/bottleneck_report.py --all [--results-root DIR]
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentx_metrics import load_schedule_eval_records, prometheus_quantile
from metrics_v2 import itl_chunk_gaps, load_done_rows, stats, tpot_values_ms, tok_per_s

SCHEMA = "bottleneck_report/1"

# This branch's /metrics export stores counter families WITHOUT the _total
# suffix and has no forward_execution family at all; GPU busy is the
# scheduler_stage_seconds {category=run_batch} label (per dp_rank).
FAMILY_IDLE = "sglang:scheduler_idle_seconds"
FAMILY_CPU = "sglang:scheduler_process_cpu_seconds"
FAMILY_STAGE = "sglang:scheduler_stage_seconds"
STAGE_BUSY_CATEGORY = "run_batch"
FAMILY_RUNNING = "sglang:num_running_reqs"
FAMILY_QUEUE = "sglang:num_queue_reqs"
FAMILY_TOKEN_USAGE = "sglang:token_usage"
FAMILY_RETRACT_TOTAL = "sglang:num_retracted_reqs"
FAMILY_STAGE_LATENCY = "sglang:per_stage_req_latency_seconds"
FAMILY_PREFILL_EFFECTIVE = "sglang:prefill_effective_tokens"
FAMILY_CACHE_HIT_RATE = "sglang:cache_hit_rate"

# All thresholds below are analysis heuristics, not measured constants;
# they steer the verdict text, and every block also carries the raw numbers
# the threshold was applied to.
BUSY_GPU_BOUND = 0.90        # arbitrary: GPU-busy share above this = GPU-bound
BUSY_NOT_GPU_BOUND = 0.70    # arbitrary: below this, look off-GPU first
ITL_TPOT_GAP = 1.15          # arbitrary: 15% client-vs-server token gap
TTFT_RESIDUAL_SHARE = 0.10   # arbitrary: 10% unaccounted TTFT = front-path
RANK_CV_IMBALANCE = 0.15     # arbitrary: rank CV above this = imbalance
TOKEN_USAGE_KV_BOUND = 0.90  # arbitrary: KV pool this full invites retraction
RETRACTION_SHARE = 0.01      # arbitrary: >1% of requests retracted
ACCEPT_LEN_LOW = 2.0         # arbitrary: draft 4, accept < 2 = spec overhead
QUEUE_EMPTY_MEAN = 1.0       # arbitrary: mean waiting depth under this = empty


def _clean(value):
    # NaN -> None so output stays valid JSON (json.dumps writes bare NaN).
    if isinstance(value, float) and value != value:
        return None
    return value


def _clean_rec(obj):
    if isinstance(obj, dict):
        return {k: _clean_rec(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_rec(v) for v in obj]
    return _clean(obj)


def _label_key(labels: dict | None) -> tuple:
    return tuple(sorted((labels or {}).items()))


def load_scrapes(run_dir: Path) -> list[tuple[float, dict]]:
    """PROFILING-phase (timestamp_s, metrics) pairs from the 3 Hz export."""
    path = run_dir / "server_metrics_export.jsonl"
    if not path.exists():
        return []
    scrapes = []
    for line in path.open():
        rec = json.loads(line)
        if rec.get("benchmark_phase") != "profiling" or not rec.get("timestamp_ns"):
            continue
        scrapes.append((rec["timestamp_ns"] / 1e9, rec.get("metrics") or {}))
    return scrapes


def _samples(metrics: dict, family: str) -> list[dict]:
    # The aiperf export stores each family as a flat list of sample dicts;
    # the older fixture shape wrapped them in {"samples": [...]}.
    fam = metrics.get(family)
    if isinstance(fam, list):
        return fam
    return fam.get("samples", []) if isinstance(fam, dict) else []


def counter_deltas(
    scrapes: list[tuple[float, dict]], family: str
) -> tuple[dict[tuple, float], float, int]:
    """Per-label-set counter delta over the scrape window.

    Returns (deltas, window_s, n_started_mid_window). A label set first seen
    after the window's first scrape (e.g. a forward category that only ever
    runs mid-benchmark) has its delta taken from its own first sighting,
    which undercounts it; those sets are counted, not silently kept.
    """
    if not scrapes:
        return {}, 0.0, 0
    window_s = scrapes[-1][0] - scrapes[0][0]
    first_ts = scrapes[0][0]
    seen_ts: dict[tuple, float] = {}
    first_val: dict[tuple, float | None] = {}
    last: dict[tuple, float] = {}
    for ts, metrics in scrapes:
        for sample in _samples(metrics, family):
            key = _label_key(sample.get("labels"))
            if key not in seen_ts:
                seen_ts[key] = ts
                first_val[key] = sample.get("value")
            if sample.get("value") is not None:
                last[key] = sample["value"]
    started_mid = sum(1 for ts in seen_ts.values() if ts > first_ts)
    deltas = {
        key: last[key] - v
        for key, v in first_val.items()
        if v is not None and key in last and last[key] - v >= 0
    }
    return deltas, window_s, started_mid


def gauge_series(
    scrapes: list[tuple[float, dict]], family: str
) -> dict[tuple, list[float]]:
    """Per-label-set gauge value series across the scrapes."""
    series: dict[tuple, list[float]] = {}
    for _, metrics in scrapes:
        for sample in _samples(metrics, family):
            if sample.get("value") is None:
                continue
            series.setdefault(_label_key(sample.get("labels")), []).append(sample["value"])
    return series


def gauge_mean_per_scrape(scrapes: list[tuple[float, dict]], family: str) -> list[float]:
    """Mean across label sets per scrape (e.g. token_usage across ranks)."""
    out = []
    for _, metrics in scrapes:
        values = [s["value"] for s in _samples(metrics, family) if s.get("value") is not None]
        if values:
            out.append(sum(values) / len(values))
    return out


def histogram_delta(
    scrapes: list[tuple[float, dict]], family: str, label_contains: dict[str, str]
) -> dict | None:
    """First->last delta of one labeled histogram (label_contains matches a
    subset of the sample's labels), Prometheus bucket dict + sum + count."""
    def match(sample: dict) -> bool:
        labels = sample.get("labels") or {}
        return all(labels.get(k) == v for k, v in label_contains.items())

    first = None
    last = None
    for _, metrics in scrapes:
        for sample in _samples(metrics, family):
            if not match(sample) or sample.get("count") is None:
                continue
            snap = {
                "buckets": dict(sample.get("buckets") or {}),
                "sum": sample.get("sum"),
                "count": sample["count"],
            }
            if first is None:
                first = snap
            last = snap
    if first is None or last is None:
        return None
    delta_count = last["count"] - first["count"]
    if delta_count <= 0:
        return None
    return {
        "buckets": {
            le: last["buckets"].get(le, 0.0) - first["buckets"].get(le, 0.0)
            for le in last["buckets"]
        },
        "sum": (last["sum"] or 0.0) - (first["sum"] or 0.0),
        "count": delta_count,
    }


def load_gpu_samples(run_dir: Path) -> list[dict]:
    """Layer-1 nvidia-smi CSV rows (header names normalized, units stripped).
    The last line can be a partial write (sampler killed mid-query); csv
    reader drops short rows via restkey accumulation, so filter by width."""
    path = run_dir / "gpu_samples.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return []
        names = [h.split("[")[0].strip().lower().replace(".", "_") for h in header]
        for raw in reader:
            if len(raw) != len(names):
                continue
            rows.append(dict(zip(names, raw)))
    return rows


def _gpu_util_mean(rows: list[dict], gpu_index: str | None = None) -> float | None:
    values = []
    for row in rows:
        if gpu_index is not None and row.get("index") != gpu_index:
            continue
        try:
            values.append(float(row.get("utilization_gpu", "")))
        except ValueError:
            continue
    return sum(values) / len(values) if values else None


def find_server_log(run_dir: Path) -> Path | None:
    """This run's own boot log. Fresh boot per cell, so the policy dir
    accumulates one server_*.log per cell (plus killed retries); the run dir
    name c<conc>_<ts> and the boot log server_<ts> are stamped seconds apart
    (run dir first, boot right after), so match on that timestamp instead of
    taking the newest. profile_cell symlinks the boot log into the run dir,
    which wins outright."""
    linked = sorted(run_dir.glob("server_*.log"))
    if linked:
        return linked[-1]
    m = re.search(r"_(\d{8}_\d{6})$", run_dir.name)
    if not m:
        logs = sorted(run_dir.parent.glob("server_*.log"))
        return logs[-1] if logs else None
    run_ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    best = None
    best_delta = None
    for log in run_dir.parent.glob("server_*.log"):
        lm = re.search(r"server_(\d{8}_\d{6})\.log$", log.name)
        if not lm:
            continue
        delta = abs(
            (datetime.strptime(lm.group(1), "%Y%m%d_%H%M%S") - run_ts).total_seconds()
        )
        # Boot log lands within seconds of the run dir; anything further is a
        # different cell's boot.
        if delta <= 120 and (best_delta is None or delta < best_delta):
            best, best_delta = log, delta
    return best


ACCEPT_LEN_RE = re.compile(r"accept len: ([0-9.]+)")
GEN_TPS_RE = re.compile(r"gen throughput \(token/s\): ([0-9.]+)")


def parse_spec_log(path: Path | None) -> dict:
    """Mean accept len / gen throughput over the server log's Decode-batch
    interval lines. Fresh boot per cell (FORCE_RESTART=1), so the whole log
    is this run; the values are per-interval means, averaged unweighted."""
    out = {
        "server_log": str(path) if path else None,
        "accept_len_mean": None,
        "accept_len_intervals": 0,
        "gen_throughput_mean": None,
    }
    if path is None or not path.exists():
        return out
    accept = []
    gen_tps = []
    for line in path.open(errors="replace"):
        if "Decode batch" not in line:
            continue
        m = ACCEPT_LEN_RE.search(line)
        if m:
            accept.append(float(m.group(1)))
        m = GEN_TPS_RE.search(line)
        if m:
            gen_tps.append(float(m.group(1)))
    out["accept_len_mean"] = sum(accept) / len(accept) if accept else None
    out["accept_len_intervals"] = len(accept)
    out["gen_throughput_mean"] = sum(gen_tps) / len(gen_tps) if gen_tps else None
    return out


def _num(value) -> float | None:
    return value if isinstance(value, (int, float)) else None


def _series_rank_key(key: tuple) -> str:
    for name, value in key:
        if name == "dp_rank":
            return value
    return "0"


def q1_gpu_busy(
    scrapes: list[tuple[float, dict]], gpu_rows: list[dict]
) -> dict:
    """GPU busy fraction + the not-GPU-bound sub-split (queue depth, idle,
    CPU rate, ranked scheduler-stage seconds)."""
    idle_deltas, _, _ = counter_deltas(scrapes, FAMILY_IDLE)
    cpu_deltas, _, _ = counter_deltas(scrapes, FAMILY_CPU)
    stage_deltas, window_s, n_mid = counter_deltas(scrapes, FAMILY_STAGE)
    running = gauge_series(scrapes, FAMILY_RUNNING)
    queue = gauge_series(scrapes, FAMILY_QUEUE)

    busy_deltas = {
        key: delta
        for key, delta in stage_deltas.items()
        if dict(key).get("category") == STAGE_BUSY_CATEGORY
    }
    # Distinct dp_rank labels, not label sets: the stage family carries a
    # category label, so label-set counting would multiply every rank.
    ranks = set()
    for family_keys in (running.keys(), busy_deltas.keys()):
        for key in family_keys:
            ranks.add(_series_rank_key(key))
    n_ranks = len(ranks) or 1
    busy_s = sum(busy_deltas.values())
    busy_fraction = busy_s / (window_s * n_ranks) if window_s > 0 else None
    idle_rate = (
        sum(idle_deltas.values()) / (window_s * n_ranks) if window_s > 0 else None
    )
    cpu_rate = sum(cpu_deltas.values()) / (window_s * n_ranks) if window_s > 0 else None
    queue_means = [sum(v) / len(v) for v in queue.values()]
    queue_mean = sum(queue_means) / len(queue_means) if queue_means else None

    # Rank stage seconds by category (summed over forward_overlap labels).
    stage_by_category: dict[str, float] = {}
    for key, delta in stage_deltas.items():
        category = dict(key).get("category", "?")
        stage_by_category[category] = stage_by_category.get(category, 0.0) + delta
    total_stage_s = sum(stage_by_category.values())
    stage_ranked = []
    if total_stage_s > 0:
        stage_ranked = sorted(
            (
                {"category": c, "seconds": s, "share_of_stage_s": s / total_stage_s}
                for c, s in stage_by_category.items()
            ),
            key=lambda e: -e["seconds"],
        )

    if busy_fraction is None:
        verdict = "INSUFFICIENT_DATA"
    elif busy_fraction >= BUSY_GPU_BOUND:
        verdict = "GPU_BOUND"
    elif busy_fraction <= BUSY_NOT_GPU_BOUND:
        if queue_mean is not None and queue_mean < QUEUE_EMPTY_MEAN and (idle_rate or 0) > 0.1:
            verdict = "NOT_GPU_BOUND__CLIENT_STARVED"
        elif (queue_mean or 0) >= QUEUE_EMPTY_MEAN:
            verdict = "NOT_GPU_BOUND__SCHEDULER_STALL"
        else:
            verdict = "NOT_GPU_BOUND__UNEXPLAINED_ESCALATE_PROFILE_CELL"
    else:
        verdict = "MIXED_GPU_BUSY"

    return {
        "verdict": verdict,
        "window_s": window_s,
        "n_ranks": n_ranks,
        "busy_fraction": busy_fraction,
        "run_batch_seconds_sum": busy_s,
        "n_forward_labelsets_started_mid_window": n_mid,
        "gpu_util_mean_pct": _gpu_util_mean(gpu_rows),
        "gpu_samples_available": bool(gpu_rows),
        "scheduler_idle_rate_per_rank": idle_rate,
        "scheduler_cpu_rate_per_rank": cpu_rate,
        "queue_reqs_mean": queue_mean,
        "stage_seconds_ranked": stage_ranked[:6],
        "thresholds": {
            "gpu_bound": BUSY_GPU_BOUND,
            "not_gpu_bound": BUSY_NOT_GPU_BOUND,
        },
    }


def q2_stream_gap(done_rows: list[dict], eval_records: list[dict]) -> dict:
    """Client ITL p50 vs server TPOT p50: the detokenizer/streaming path has
    no metrics, so a persistent gap is itself the evidence."""
    gaps_ms = itl_chunk_gaps(done_rows)
    if gaps_ms is not None:
        client = {"source": "per_chunk", **stats(gaps_ms, (50, 99))}
    else:
        client = {"source": "tpot_based", **stats(tpot_values_ms(done_rows), (50, 99))}
    server_tpots_ms = [
        (r["e2e_s"] - r["ttft_s"]) * 1000.0 / (r["output_len"] - 1)
        for r in eval_records
        if _num(r.get("e2e_s")) is not None
        and _num(r.get("ttft_s")) is not None
        and isinstance(r.get("output_len"), int)
        and r["output_len"] >= 2
    ]
    server = stats(server_tpots_ms, (50, 99)) if server_tpots_ms else {"n": 0}
    ratio = (
        client["p50"] / server["p50"]
        if client.get("p50") is not None
        and server.get("p50")
        else None
    )
    verdict = (
        "DETOKENIZER_OR_STREAMING_SUSPECT"
        if ratio is not None and ratio > ITL_TPOT_GAP
        else "NO_CLIENT_SERVER_GAP"
    )
    if client["n"] == 0 or server["n"] == 0:
        verdict = "INSUFFICIENT_DATA"
    return {
        "verdict": verdict,
        "client_itl_ms": client,
        "server_tpot_ms": server,
        "p50_ratio": ratio,
        "threshold_ratio": ITL_TPOT_GAP,
    }


def q3_ttft_decomposition(
    done_rows: list[dict], eval_records: list[dict], scrapes: list[tuple[float, dict]]
) -> dict:
    """Client TTFT split into tokenize stage + server ttft (queue wait +
    prefill); the residual is the HTTP/TokenizerManager/DPC front path. This
    branch has no TOKENIZE stage in per_stage_req_latency (only
    chunked_prefill/prefill_forward/request_process), so tokenize_p50 is
    None there and the residual then also covers tokenization."""
    client_ttft_ms = [row["ttft_s"] * 1000.0 for row in done_rows]
    client = stats(client_ttft_ms, (50, 95)) if client_ttft_ms else {"n": 0}
    waits_s = [
        r["queue_wait_s"] for r in eval_records if _num(r.get("queue_wait_s")) is not None
    ]
    server_ttfts_s = [
        r["ttft_s"] for r in eval_records if _num(r.get("ttft_s")) is not None
    ]
    queue_wait = stats([w * 1000.0 for w in waits_s], (50, 95)) if waits_s else {"n": 0}
    server_ttft = stats([t * 1000.0 for t in server_ttfts_s], (50, 95)) if server_ttfts_s else {"n": 0}

    tokenize_hist = histogram_delta(scrapes, FAMILY_STAGE_LATENCY, {"stage": "TOKENIZE"})
    tokenize_p50_s = (
        prometheus_quantile(tokenize_hist["buckets"], 0.50)
        if tokenize_hist
        else None
    )
    tokenize_p50_ms = tokenize_p50_s * 1000.0 if tokenize_p50_s is not None else None

    # schedule_eval ttft = scheduler_recv -> prefill_finished (queue wait +
    # prefill), and tokenize happens before scheduler_recv, so the two
    # addends do not double count. tokenize_p50 is None on this branch (no
    # TOKENIZE stage), and the residual then also covers tokenization.
    residual_ms = None
    residual_share = None
    if client.get("p50") is not None and server_ttft.get("p50") is not None:
        residual_ms = (
            client["p50"]
            - server_ttft["p50"]
            - (tokenize_p50_ms if tokenize_p50_ms is not None else 0.0)
        )
        residual_share = residual_ms / client["p50"] if client["p50"] > 0 else None
    queue_share = (
        queue_wait["p50"] / server_ttft["p50"]
        if queue_wait.get("p50") is not None and server_ttft.get("p50")
        else None
    )
    verdict = (
        "FRONT_PATH_OVERHEAD"
        if residual_share is not None and residual_share > TTFT_RESIDUAL_SHARE
        else "TTFT_ACCOUNTED_FOR"
    )
    if client["n"] == 0 or server_ttft["n"] == 0:
        verdict = "INSUFFICIENT_DATA"
    return {
        "verdict": verdict,
        "client_ttft_ms": client,
        "tokenize_stage_p50_ms": tokenize_p50_ms,
        "server_ttft_ms": server_ttft,
        "queue_wait_ms": queue_wait,
        "queue_share_of_server_ttft": queue_share,
        "front_path_residual_ms": residual_ms,
        "front_path_residual_share": residual_share,
        "threshold_share": TTFT_RESIDUAL_SHARE,
    }


def q4_rank_balance(scrapes: list[tuple[float, dict]]) -> dict:
    """Per-dp_rank running/queue balance: the DPC's round-robin is
    load-blind, and agentx ISLs vary widely, so skew is likely."""
    running = gauge_series(scrapes, FAMILY_RUNNING)
    queue = gauge_series(scrapes, FAMILY_QUEUE)
    per_rank = []
    for key, series in sorted(running.items(), key=lambda kv: _series_rank_key(kv[0])):
        rank = _series_rank_key(key)
        q_series = queue.get(key, [])
        per_rank.append(
            {
                "dp_rank": rank,
                "running_mean": sum(series) / len(series) if series else None,
                "queue_mean": sum(q_series) / len(q_series) if q_series else None,
                "n_scrapes": len(series),
            }
        )
    means = [r["running_mean"] for r in per_rank if r["running_mean"] is not None]
    cv = None
    if len(means) >= 2 and sum(means) > 0:
        mu = sum(means) / len(means)
        var = sum((m - mu) ** 2 for m in means) / len(means)
        cv = var**0.5 / mu
    max_queue = max(
        (r["queue_mean"] for r in per_rank if r["queue_mean"] is not None), default=None
    )
    if not per_rank:
        verdict = "INSUFFICIENT_DATA"
    elif cv is not None and cv > RANK_CV_IMBALANCE:
        verdict = "ROUND_ROBIN_IMBALANCE"
    else:
        verdict = "BALANCED"
    return {
        "verdict": verdict,
        "n_ranks": len(per_rank),
        "per_rank": per_rank,
        "running_mean_cv": cv,
        "max_queue_mean": max_queue,
        "threshold_cv": RANK_CV_IMBALANCE,
        "stage2_fix": "--load-balance-method total_tokens (in-tree, report only)",
    }


def q5_kv(eval_records: list[dict], scrapes: list[tuple[float, dict]]) -> dict:
    """KV pressure: pool usage + share of requests retracted + the
    monotonic retraction counter over the window."""
    usage = gauge_mean_per_scrape(scrapes, FAMILY_TOKEN_USAGE)
    retracted = [
        r for r in eval_records if isinstance(r.get("num_retractions"), int) and r["num_retractions"] > 0
    ]
    retract_deltas, _, _ = counter_deltas(scrapes, FAMILY_RETRACT_TOTAL)
    usage_max = max(usage) if usage else None
    share = len(retracted) / len(eval_records) if eval_records else None
    delta = sum(retract_deltas.values())
    verdict = (
        "KV_BOUND"
        if (usage_max or 0) > TOKEN_USAGE_KV_BOUND
        and (share or 0) > RETRACTION_SHARE
        and delta > 0
        else "KV_NOT_LIMITING"
    )
    return {
        "verdict": verdict,
        "token_usage_mean": sum(usage) / len(usage) if usage else None,
        "token_usage_max": usage_max,
        "retracted_request_share": share,
        "n_retracted_records": len(retracted),
        "retraction_counter_delta": delta,
        "thresholds": {
            "token_usage": TOKEN_USAGE_KV_BOUND,
            "retraction_share": RETRACTION_SHARE,
        },
    }


def q6_spec(server_log: Path | None, speculative: bool) -> dict:
    """Spec-decode efficiency from the server log's per-interval accept len
    (both arms draft 4 tokens). Low accept len turns EAGLE into overhead."""
    parsed = parse_spec_log(server_log)
    accept = parsed["accept_len_mean"]
    if not speculative:
        parsed["verdict"] = "NOT_APPLICABLE"
    elif accept is None:
        parsed["verdict"] = "INSUFFICIENT_DATA"
    else:
        parsed["verdict"] = (
            "SPEC_ACCEPT_LOW" if accept < ACCEPT_LEN_LOW else "SPEC_HEALTHY"
        )
    parsed["threshold_accept_len"] = ACCEPT_LEN_LOW
    return parsed


def q7_hicache(eval_records: list[dict], scrapes: list[tuple[float, dict]]) -> dict:
    """Hierarchical-cache behavior: host-hit share of prefill-effective
    tokens (write_back ramps host hits up over a run) and the hit-rate
    gauge trend (first third vs last third)."""
    deltas, _, _ = counter_deltas(scrapes, FAMILY_PREFILL_EFFECTIVE)
    mode_totals: dict[str, float] = {}
    for key, delta in deltas.items():
        mode = dict(key).get("mode", "?")
        mode_totals[mode] = mode_totals.get(mode, 0.0) + delta
    total = sum(mode_totals.values())
    host_hit_share = (
        mode_totals.get("host_hit", 0.0) / total if total > 0 else None
    )
    device_hit_share = (
        mode_totals.get("device_hit", 0.0) / total if total > 0 else None
    )
    hit_rate = gauge_mean_per_scrape(scrapes, FAMILY_CACHE_HIT_RATE)
    if len(hit_rate) >= 6:
        third = len(hit_rate) // 3
        trend = hit_rate[-third:][-1] - hit_rate[:third][0]
        trend_note = "last-third minus first-third mean"
    else:
        trend = None
        trend_note = "too few scrapes"
    host_records = [
        r
        for r in eval_records
        if _num(r.get("host_hit_prefix_tokens_first_admission")) not in (None, 0)
    ]
    return {
        "verdict": "INFO_ONLY",
        "prefill_effective_tokens": mode_totals,
        "host_hit_share": host_hit_share,
        "device_hit_share": device_hit_share,
        "cache_hit_rate_first": hit_rate[0] if hit_rate else None,
        "cache_hit_rate_last": hit_rate[-1] if hit_rate else None,
        "cache_hit_rate_trend": trend,
        "cache_hit_rate_trend_note": trend_note,
        "host_hit_request_share": (
            len(host_records) / len(eval_records) if eval_records else None
        ),
    }


def compute_report(run_dir: Path) -> dict:
    """Assemble the Q1-Q7 blocks + a cross-question verdict list."""
    meta = json.loads((run_dir / "run_meta.json").read_text())
    done_rows, _ = load_done_rows(run_dir)
    eval_records, _ = load_schedule_eval_records(run_dir)
    if eval_records is None:
        eval_records = []
    scrapes = load_scrapes(run_dir)
    gpu_rows = load_gpu_samples(run_dir)
    server_log = find_server_log(run_dir)
    # Both arms run EAGLE (draft 4); the boot's server-args dump is the
    # authoritative check when run_meta carries no spec field.
    dump = server_log.read_text(errors="replace")[:200000] if server_log else ""
    speculative = "'EAGLE'" in dump

    report = {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "run": {
            "policy": meta.get("policy"),
            "concurrency": meta.get("concurrency"),
            "seed": meta.get("seed"),
            "duration_s": meta.get("duration_s"),
        },
        "n": {
            "p_done": len(done_rows),
            "eval_records": len(eval_records),
            "server_scrapes": len(scrapes),
            "gpu_samples": len(gpu_rows),
        },
        "tok_per_s": tok_per_s(done_rows)[0],
        "q1_gpu_busy": q1_gpu_busy(scrapes, gpu_rows),
        "q2_stream_gap": q2_stream_gap(done_rows, eval_records),
        "q3_ttft": q3_ttft_decomposition(done_rows, eval_records, scrapes),
        "q4_rank_balance": q4_rank_balance(scrapes),
        "q5_kv": q5_kv(eval_records, scrapes),
        "q6_spec": q6_spec(server_log, speculative),
        "q7_hicache": q7_hicache(eval_records, scrapes),
    }
    report["verdicts"] = {
        name.removeprefix("q").split("_", 1)[0]: block["verdict"]
        for name, block in report.items()
        if name.startswith("q") and isinstance(block, dict) and "verdict" in block
    }
    return _clean_rec(report)


def write_report(run_dir: Path) -> dict:
    report = compute_report(run_dir)
    out_path = run_dir / "bottleneck_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _pct(v: float | None) -> str:
    return f"{100 * v:5.1f}" if isinstance(v, (int, float)) else "  n/a"


def _f2(v: float | None) -> str:
    return f"{v:6.2f}" if isinstance(v, (int, float)) else "   n/a"


def table_row(report: dict) -> str:
    """One cross-cell table line: the numbers the decision tree turned on."""
    q1, q2, q3, q4, q5 = (
        report["q1_gpu_busy"],
        report["q2_stream_gap"],
        report["q3_ttft"],
        report["q4_rank_balance"],
        report["q5_kv"],
    )
    accept = report["q6_spec"]["accept_len_mean"]
    return (
        f"{str(report['run']['policy']):<11} c{str(report['run']['concurrency']):<4} "
        f"busy {_pct(q1['busy_fraction'])}%  cpu {_pct(q1['scheduler_cpu_rate_per_rank'])}%  "
        f"q/rank {_f2(q1['queue_reqs_mean'])}  "
        f"cv {_pct(q4['running_mean_cv'])}%  "
        f"retr {_pct(q5['retracted_request_share'])}%  "
        f"acc {_f2(accept)}  "
        f"itl/tpt {_f2(q2['p50_ratio'])}  "
        f"resid {_pct(q3['front_path_residual_share'])}%  "
        f"tok/s {_f2(report['tok_per_s'])}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--all", action="store_true", help="every run dir with run_meta.json")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "1x-rtx6000pro" / "agentx",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    if args.all:
        run_dirs = sorted(p.parent for p in args.results_root.glob("*/*/run_meta.json"))
        if not run_dirs:
            sys.exit(f"no run dirs under {args.results_root}")
        for run_dir in run_dirs:
            report = write_report(run_dir)
            print(table_row(report))
        return

    if args.run_dir is None:
        parser.error("give a run_dir or --all")
    report = write_report(args.run_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"bottleneck report written to {args.run_dir / 'bottleneck_report.json'}")


if __name__ == "__main__":
    main()
