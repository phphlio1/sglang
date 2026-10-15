#!/usr/bin/env python3
"""Shared computations for the focus report (structure of
docs/focus3320_routing_scheduling.md), usable for any profiling run dir.

Everything here mirrors the one-off analyze_3320_*.py scripts (kept as
historical reference) but returns values instead of printing. Log parsing
strips ANSI codes first; engine/frontend request ids differ (frontend UUID
vs engine 32-hex rid) so joins are timestamp-based greedy-nearest (+-2 s).

Typical flow:
    from focus_report_lib import analyze_run
    rep = analyze_run("<run_dir>")        # dict of every section
    # or per-section functions, each taking (run_dir, t0, t1)

Window derivation: profile_export.jsonl metadata.benchmark_phase is
'warmup' or 'profiling'; T0 = min request_start_ns of profiling records,
T1 = T0 + duration (e2_metrics.json run.duration_s).
"""
import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ENG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
SEL_RE = re.compile(
    r'Selected worker router_mode="kv" request_id="([0-9a-f-]+)" '
    r'worker_id=\d+ worker_type=decode dp_rank=(\d+) logit=([\d.]+)'
    r"(?: overlap_blocks=(\d+))?"
)
COM_RE = re.compile(r'request completed request_id=([0-9a-f-]+) .*ttft_ms="?([\d.]+)"?')
HTTP_RE = re.compile(r"http response sent status=(\d+) latency_ms=(\d+)")
GPU_TS = "%Y/%m/%d %H:%M:%S.%f"

SG_METRICS = [
    "sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:full_token_usage",
    "sglang:pending_prealloc_token_usage", "sglang:cache_hit_rate",
    "sglang:kv_used_tokens", "sglang:hicache_host_used_tokens",
    "sglang:num_prefill_inflight_queue_reqs", "sglang:num_retracted_reqs",
]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


def med(v):
    return pct(v, 50)


def hhmmss(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%H:%M:%S")


def iso_min(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%H:%M")


def fmt_epoch(iso):  # engine-log 'YYYY-MM-DDTHH:MM:SS' (UTC) -> epoch
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------- files
def glob1(run_dir, pattern):
    hits = sorted(glob.glob(os.path.join(run_dir, pattern)))
    if not hits:
        raise FileNotFoundError(f"no match for {pattern} in {run_dir}")
    return hits[0]


def engine_log(run_dir):
    return glob1(run_dir, "engine_engine_*.log")


def frontend_log(run_dir):
    return glob1(run_dir, "frontend_frontend_*.log")


def run_stamp(run_dir):
    return os.path.basename(run_dir.rstrip("/"))


# ---------------------------------------------------------------- window
def derive_window(run_dir):
    """-> dict(t0, t1, duration, n_warmup, n_profiling, warmup_last, gap)."""
    starts, phases = [], Counter()
    with open(os.path.join(run_dir, "profile_export.jsonl")) as f:
        for line in f:
            md = json.loads(line)["metadata"]
            phases[md.get("benchmark_phase")] += 1
            if md.get("benchmark_phase") == "profiling":
                starts.append(md["request_start_ns"] / 1e9)
    if not starts:
        raise ValueError("no profiling-phase records in profile_export.jsonl")
    t0 = min(starts)
    with open(os.path.join(run_dir, "e2_metrics.json")) as f:
        dur = float(json.load(f)["run"]["duration_s"])
    t1 = t0 + dur
    return {
        "t0": t0, "t1": t1, "duration": dur,
        "n_warmup": phases.get("warmup", 0),
        "n_profiling": phases.get("profiling", 0),
    }


def client_metrics(run_dir):
    with open(os.path.join(run_dir, "e2_metrics.json")) as f:
        return json.load(f)


def input_len_median_profile(run_dir):
    v = []
    with open(os.path.join(run_dir, "profile_export.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["metadata"].get("benchmark_phase") == "profiling":
                u = r.get("metrics", {}).get("usage_prompt_tokens", {})
                if u.get("value") is not None:
                    v.append(u["value"])
    return med(v) if v else float("nan")


# ---------------------------------------------------------------- engine log
def engine_minutes(run_dir, t0, t1):
    """Per-minute Decode/Prefill aggregation + examples + WARN/boot scans."""
    mins = defaultdict(lambda: {
        "n_dec": 0, "n_pre": 0, "usage": [], "running": [], "queue": [],
        "gen": [], "accept": [], "pending": [], "cached": 0, "newtok": 0,
        "occ": [], "in_thr": [],
    })
    n_batch = 0
    dec_before_t0 = 0
    cached_w = new_w = 0  # whole-log prefill token totals (warmup incl.)
    retracts = []      # (hhmmss, retracted, gained)
    boot_kv = []       # (hhmmss, tokens, gb) — raw boot alloc lines (per rank)
    boot_hicache = []  # (hhmmss, tokens, gb)
    ex_dec_max, ex_dec_min = None, None
    dec_ex = []  # (running, hhmmss, gen_thr) mọi decode line trong window
    pre_ex = []  # (hhmmss, pending, input_thr) mọi prefill line trong window
    with open(engine_log(run_dir)) as f:
        for line in f:
            line = ANSI.sub("", line)
            t = ENG_TS.match(line)
            if not t:
                continue
            iso = t.group(1)
            ep = fmt_epoch(iso)
            if "Retract requests." in line:
                m = re.search(r"#retracted_reqs: (\d+)(?:, #new_tokens_gained: (\d+))?", line)
                retracts.append((iso[11:19], int(m.group(1)) if m else 0,
                                 int(m.group(2) or 0) if m else 0))
                continue
            if "KV Cache is allocated" in line:
                m = re.search(r"#tokens: (\d+), .*?KV size: ([\d.]+) GB", line)
                if m:
                    boot_kv.append((iso[11:19], int(m.group(1)), float(m.group(2))))
                continue
            if "hierarchical KV host pool" in line:
                m = re.search(r"pool: (\d+) tokens, ([\d.]+) GB", line)
                if m:
                    boot_hicache.append((iso[11:19], int(m.group(1)), float(m.group(2))))
                continue
            if "batch," not in line:
                continue
            n_batch += 1
            if "Decode batch" in line:
                if ep < t0:
                    dec_before_t0 += 1
                    continue
                run_ = int(re.search(r"#running-req: (\d+)", line).group(1))
                gen_ = float(re.search(r"gen throughput \(token/s\): ([\d.]+)", line).group(1))
                if ep <= t1:
                    dec_ex.append((run_, iso[11:19], gen_))
                    if ex_dec_max is None or run_ > ex_dec_max[0]:
                        ex_dec_max = (run_, iso[11:19], gen_)
            elif "Prefill batch" in line:
                m = re.search(r"#cached-token: (\d+)", line)
                c = int(m.group(1)) if m else 0
                m = re.search(r"#new-token: (\d+)", line)
                n = int(m.group(1)) if m else 0
                cached_w += c
                new_w += n
                if ep < t0:
                    continue
                mp = re.search(r"#pending-token: (\d+)", line)
                if mp and ep <= t1:
                    pend = int(mp.group(1))
                    thr = float(re.search(r"input throughput \(token/s\): ([\d.]+)", line).group(1))
                    pre_ex.append((iso[11:19], pend, thr))
            else:
                continue
            if ep > t1:
                continue
            d = mins[iso[:16]]
            if "Decode batch" in line:
                d["n_dec"] += 1
                d["usage"].append(float(re.search(r"token usage: ([\d.]+)", line).group(1)))
                d["running"].append(int(re.search(r"#running-req: (\d+)", line).group(1)))
                d["queue"].append(int(re.search(r"#queue-req: (\d+)", line).group(1)))
                d["gen"].append(float(re.search(r"gen throughput \(token/s\): ([\d.]+)", line).group(1)))
                d["accept"].append(float(re.search(r"accept len: ([\d.]+)", line).group(1)))
                d["occ"].append(float(re.search(r"fwd occupancy: ([\d.]+)%", line).group(1)))
            else:
                d["n_pre"] += 1
                d["usage"].append(float(re.search(r"token usage: ([\d.]+)", line).group(1)))
                d["pending"].append(int(re.search(r"#pending-token: (\d+)", line).group(1)))
                d["cached"] += c
                d["newtok"] += n
                d["occ"].append(float(re.search(r"fwd occupancy: ([\d.]+)%", line).group(1)))
                d["in_thr"].append(float(re.search(r"input throughput \(token/s\): ([\d.]+)", line).group(1)))

    cached = sum(d["cached"] for d in mins.values())
    newtok = sum(d["newtok"] for d in mins.values())
    run_means = [mean(d["running"]) for d in mins.values() if d["running"]]
    occ_means = [mean(d["occ"]) for d in mins.values() if d["occ"]]
    # per-minute mean ranges (doc semantics: "per-minute tổng hợp")
    pend_pm = [mean(d["pending"]) for d in mins.values() if d["pending"]]
    acc_pm = [mean(d["accept"]) for d in mins.values() if d["accept"]]
    # decode examples: max running (first seen); min running với gen median
    ex_dec_min = ex_pre_med = ex_pre_maxpend = None
    if dec_ex:
        rmin = min(x[0] for x in dec_ex)
        cand = sorted([x for x in dec_ex if x[0] == rmin], key=lambda x: x[2])
        ex_dec_min = cand[len(cand) // 2]
    if pre_ex:
        ex_pre_maxpend = max(pre_ex, key=lambda x: x[1])
        pmed = med([x[1] for x in pre_ex])
        ex_pre_med = min(pre_ex, key=lambda x: abs(x[1] - pmed))
    return {
        "n_batch": n_batch,
        "dec_before_t0": dec_before_t0,
        "minutes": dict(mins),
        # whole-log prefill token totals (warmup included — doc "tổng hợp")
        "cache_reuse_pct": 100 * cached_w / (cached_w + new_w) if (cached_w + new_w) else float("nan"),
        "cached_m": cached_w / 1e6, "new_m": new_w / 1e6,
        "cache_reuse_pct_win": 100 * cached / (cached + newtok) if (cached + newtok) else float("nan"),
        "run_mean_min": min(run_means) if run_means else float("nan"),
        "run_mean_max": max(run_means) if run_means else float("nan"),
        "occ_min": min(occ_means) if occ_means else float("nan"),
        "occ_max": max(occ_means) if occ_means else float("nan"),
        "pending_pm_min": min(pend_pm) if pend_pm else float("nan"),
        "pending_pm_max": max(pend_pm) if pend_pm else float("nan"),
        "accept_pm_min": min(acc_pm) if acc_pm else float("nan"),
        "accept_pm_max": max(acc_pm) if acc_pm else float("nan"),
        "ex_dec_max": ex_dec_max, "ex_dec_min": ex_dec_min, "ex_pre_maxpend": ex_pre_maxpend,
        "ex_pre_med": ex_pre_med,
        "retracts": retracts,
        "boot_kv": sorted(boot_kv), "boot_hicache": sorted(boot_hicache),
    }


# ---------------------------------------------------------------- frontend log
def parse_frontend(run_dir):
    """-> events (epoch, rid, rank, logit), ttft_fe, http counter, warns."""
    events = []
    ttft_fe = {}
    http = Counter()
    expiring, failed_prefill = [], []
    with open(frontend_log(run_dir)) as f:
        for line in f:
            line = ANSI.sub("", line)
            t = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z", line)
            if not t:
                continue
            iso = t.group(1)
            ep = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%f").replace(
                tzinfo=timezone.utc).timestamp()
            if "Expiring stale request" in line:
                expiring.append(iso[11:19])
                continue
            if "Failed to mark prefill" in line:
                failed_prefill.append(iso[11:19])
                continue
            m = SEL_RE.search(line)
            if m:
                events.append((ep, m.group(1), int(m.group(2)),
                               float(m.group(3)), float(m.group(4) or 0)))
                continue
            m = COM_RE.search(line)
            if m:
                ttft_fe[m.group(1)] = float(m.group(2)) / 1000.0
                continue
            m = HTTP_RE.search(line)
            if m:
                if int(m.group(1)) == 200:
                    http[int(m.group(2))] += 1
                else:
                    http["other_" + m.group(1)] += 1
    events.sort()
    return {"events": events, "ttft_fe": ttft_fe, "http": http,
            "expiring": expiring, "failed_prefill": failed_prefill}


# ---------------------------------------------------------------- join
def join_frontend_engine(run_dir, fe, t0):
    """Greedy nearest-time match (±2 s), warmup 5% dropped, per-rank stats."""
    events = fe["events"]
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "schedule_eval.jsonl")) if l.strip()]
    arr = sorted(x["arrival_wall_s"] for x in recs)
    warmup_end = arr[int(len(arr) * 0.05)]
    used = set()

    def nearest(epoch):
        best, bd = None, 1e9
        for i, ev in enumerate(events):
            if i in used:
                continue
            d = abs(ev[0] - epoch)
            if d < bd:
                best, bd = i, d
        if best is not None and bd <= 2.0:
            used.add(best)
            return events[best]
        return None

    stats = defaultdict(lambda: {"ttft": [], "qw": [], "in": []})
    fe_ttft, en_ttft = [], []
    n_match = 0
    for x in sorted(recs, key=lambda r: r["arrival_wall_s"]):
        if x["arrival_wall_s"] < warmup_end:
            continue
        ev = nearest(x["arrival_wall_s"])
        if ev is None:
            continue
        n_match += 1
        _, rid, rank, logit, _overlap = ev
        stats[rank]["ttft"].append(x["ttft_s"])
        stats[rank]["qw"].append(x["queue_wait_s"])
        stats[rank]["in"].append(x["input_len"])
        if rid in fe["ttft_fe"]:
            fe_ttft.append(fe["ttft_fe"][rid])
            en_ttft.append(x["ttft_s"])
    per_rank = {}
    for r, s in sorted(stats.items()):
        if not s["ttft"]:
            continue
        per_rank[r] = {
            "n": len(s["ttft"]), "ttft_mean": mean(s["ttft"]),
            "ttft_p95": pct(s["ttft"], 95), "ttft_p99": pct(s["ttft"], 99),
            "qw_mean": mean(s["qw"]), "qw_p95": pct(s["qw"], 95),
            "in_mean": mean(s["in"]),
        }
    diffs = sorted(a - b for a, b in zip(fe_ttft, en_ttft)) if len(fe_ttft) > 20 else []
    return {
        "n_match": n_match, "per_rank": per_rank,
        "rank_req_counts": {r: s["n"] for r, s in per_rank.items()},
        "fe_vs_en_diff_mean": mean(diffs) if diffs else float("nan"),
        "fe_vs_en_diff_p50": pct(diffs, 50) if diffs else float("nan"),
    }


# ---------------------------------------------------------------- router cost
def _corr(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    ma, mb = mean(a), mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / n
    va = sum((x - ma) ** 2 for x in a) / n
    vb = sum((y - mb) ** 2 for y in b) / n
    return cov / (math.sqrt(va) * math.sqrt(vb)) if va > 0 and vb > 0 else 0.0


def router_cost(run_dir, fe):
    events = fe["events"]
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "schedule_eval.jsonl")) if l.strip()]
    arr = sorted(x["arrival_wall_s"] for x in recs)
    warmup_end = arr[int(len(arr) * 0.05)]
    used = set()

    def nearest(epoch):
        best, bd = None, 1e9
        for i, ev in enumerate(events):
            if i in used:
                continue
            d = abs(ev[0] - epoch)
            if d < bd:
                best, bd = i, d
        if best is not None and bd <= 2.0:
            used.add(best)
            return events[best]
        return None

    rows = []
    for x in sorted(recs, key=lambda r: r["arrival_wall_s"]):
        if x["arrival_wall_s"] < warmup_end:
            continue
        ev = nearest(x["arrival_wall_s"])
        if ev is None:
            continue
        _, rid, rank, logit, overlap = ev
        rows.append((logit, overlap, x["queue_wait_s"], x["ttft_s"],
                     x.get("matched_prefix_tokens_first_admission", 0)
                     + x.get("device_matched_prefix_tokens_first_admission", 0),
                     x["input_len"], rank))
    n = len(rows)
    logit = [r[0] for r in rows]
    qw = [r[2] for r in rows]
    ttft = [r[3] for r in rows]
    match = [r[4] for r in rows]
    inl = [r[5] for r in rows]
    rows.sort(key=lambda r: r[0])
    k = max(n // 5, 1)
    quint = []
    for b in range(5):
        grp = rows[b * k:(b + 1) * k] if b < 4 else rows[4 * k:]
        if not grp:
            continue
        quint.append({
            "bucket": b + 1, "n": len(grp),
            "logit_p50": med([r[0] for r in grp]),
            "qw_p50": med([r[2] for r in grp]),
            "ttft_p50": med([r[3] for r in grp]),
        })
    return {
        "n": n,
        "corr_logit_qw": _corr(logit, qw), "corr_logit_ttft": _corr(logit, ttft),
        "corr_logit_inlen": _corr(logit, inl), "corr_inlen_qw": _corr(inl, qw),
        "corr_match_qw": _corr(match, qw),
        "quintiles": quint,
    }


# ---------------------------------------------------------------- rank ts (8081)
def rank_ts(run_dir, t0, t1):
    per = defaultdict(lambda: defaultdict(list))
    by_ts = defaultdict(lambda: defaultdict(dict))
    n_rows = 0
    path = os.path.join(run_dir, "analysis", "sglang_rank_ts.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            n_rows += 1
            ts, metric, rank, val = (float(row["timestamp"]), row["metric"],
                                     int(row["dp_rank"]), float(row["value"]))
            if not (t0 <= ts <= t1):
                continue
            per[rank][metric].append(val)
            by_ts[ts][metric][rank] = val
    run_spread, usage_spread, zero_vs_busy = [], [], 0
    first_100 = {}
    for ts, m in by_ts.items():
        rr = m.get("sglang:num_running_reqs", {})
        if len(rr) == 8:
            vals = list(rr.values())
            run_spread.append(max(vals) - min(vals))
            if min(vals) == 0 and max(vals) >= 3:
                zero_vs_busy += 1
        uu = m.get("sglang:full_token_usage", {})
        if len(uu) == 8:
            usage_spread.append(max(uu.values()) - min(uu.values()))
            for r, v in uu.items():
                if v >= 0.999 and r not in first_100:
                    first_100[r] = ts
    means = {m: {r: (mean(per[r][m]) if per[r][m] else None) for r in range(8)}
             for m in SG_METRICS}
    return {
        "n_rows": n_rows, "n_ts": len(by_ts), "means": means,
        "run_spread_p50": pct(run_spread, 50), "run_spread_p95": pct(run_spread, 95),
        "run_spread_max": max(run_spread) if run_spread else 0,
        "zero_vs_busy": zero_vs_busy,
        "zero_vs_busy_pct": 100 * zero_vs_busy / max(len(by_ts), 1),
        "usage_spread_p50": pct(usage_spread, 50), "usage_spread_p95": pct(usage_spread, 95),
        "usage_spread_max": max(usage_spread) if usage_spread else 0,
        "first_100": {r: hhmmss(ts) for r, ts in sorted(first_100.items(), key=lambda x: x[1])},
        "retracted_max": {r: (max(per[r]["sglang:num_retracted_reqs"])
                              if per[r]["sglang:num_retracted_reqs"] else 0) for r in range(8)},
    }


# ---------------------------------------------------------------- dynamo ts (8888)
def dynamo_ts(run_dir, t0, t1):
    hist = defaultdict(lambda: defaultdict(list))   # base -> key -> [(ts, sum, count)]
    gauge = defaultdict(lambda: defaultdict(list))  # metric -> key -> [(ts, v)]
    path = os.path.join(run_dir, "analysis", "dynamo_ts.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            ts, m, key, val = (float(row["timestamp"]), row["metric"],
                               row["key"], float(row["value"]))
            if not (t0 <= ts <= t1):
                continue
            if m.endswith("_sum"):
                hist[m[:-4]][key].append((ts, val, None))
            elif m.endswith("_count"):
                hist[m[:-6]][key].append((ts, None, val))
            else:
                gauge[m][key].append((ts, val))

    def hist_delta(base):
        out = {}
        for key, rows in hist.get(base, {}).items():
            by = {}
            for ts, s, c in rows:
                d = by.setdefault(ts, {"s": None, "c": None})
                if s is not None:
                    d["s"] = s
                if c is not None:
                    d["c"] = c
            tss = sorted(by)
            ds = (by[tss[-1]]["s"] or 0) - (by[tss[0]]["s"] or 0)
            dc = (by[tss[-1]]["c"] or 0) - (by[tss[0]]["c"] or 0)
            if dc > 0:
                out[key] = {"decisions": dc, "mean_ms": ds / dc}
        return out

    def hist_mean_req(base):
        d = hist_delta(base)
        tot = sum(x["decisions"] for x in d.values())
        if not tot:
            return {"n": 0, "mean": float("nan")}
        return {"n": tot,
                "mean": sum(x["decisions"] * x["mean_ms"] for x in d.values()) / tot}

    def worker_gauge(metric):
        out = {}
        for key, rows in gauge.get(metric, {}).items():
            rank = key.split("|")[0]
            try:
                rank = int(rank)
            except ValueError:
                continue
            out[rank] = mean([v for _, v in rows])
        return out

    qpend = [v for rows in gauge.get("dynamo_frontend_router_queue_pending_requests", {}).values()
             for _, v in rows]
    return {
        "router_overhead": {base: hist_delta(base) for base in sorted(hist)
                            if base.startswith("dynamo_router_overhead")},
        "cached_tokens": hist_mean_req("dynamo_frontend_cached_tokens"),
        "itl": hist_mean_req("dynamo_frontend_inter_token_latency_seconds"),
        "ttft_fe_ms": hist_mean_req("dynamo_frontend_ttft"),
        "active_prefill": worker_gauge("dynamo_frontend_worker_active_prefill_tokens"),
        "active_decode_blocks": worker_gauge("dynamo_frontend_worker_active_decode_blocks"),
        "queue_pending_max": max(qpend) if qpend else 0.0,
        "queue_pending_mean": mean(qpend) if qpend else 0.0,
    }


def scrape_counts(run_dir):
    """Unique scrape timestamps per analysis CSV + small record counts."""
    out = {}
    for name in ("sglang_rank_ts.csv", "dynamo_ts.csv"):
        ts_set = set()
        path = os.path.join(run_dir, "analysis", name)
        with open(path) as f:
            for row in csv.DictReader(f):
                ts_set.add(row["timestamp"])
        out[name] = len(ts_set)
    with open(os.path.join(run_dir, "schedule_eval.jsonl")) as f:
        out["schedule_eval"] = sum(1 for _ in f)
    return out


def input_len_median(run_dir):
    """Median input_len theo schedule_eval (cùng semantics bảng join §2.2)."""
    v = []
    with open(os.path.join(run_dir, "schedule_eval.jsonl")) as f:
        for line in f:
            if line.strip():
                v.append(json.loads(line)["input_len"])
    return med(v) if v else float("nan")


# ---------------------------------------------------------------- gpu
def gpu_stats(run_dir, t0, t1):
    rows = []
    with open(os.path.join(run_dir, "gpu_samples.csv")) as f:
        for r in csv.DictReader(f):
            ts = datetime.strptime(r["timestamp"], GPU_TS).replace(tzinfo=timezone.utc)
            rows.append((ts.timestamp(), int(r[" index"]),
                         float(r[" utilization.gpu [%]"].replace(" %", "")),
                         float(r[" power.draw [W]"].replace(" W", ""))))
    by_gpu = defaultdict(list)
    by_gpu_pow = defaultdict(list)
    by_ts = defaultdict(dict)
    for ts, i, u, p in rows:
        by_gpu[i].append(u)
        by_gpu_pow[i].append(p)
        by_ts[ts][i] = u
    spreads = []
    for ts, m in by_ts.items():
        v = list(m.values())
        if len(v) == 8:
            spreads.append(max(v) - min(v))
    per_g = {g: {"mean": mean(v), "p50": pct(v, 50), "p95": pct(v, 95),
                 "w": mean(by_gpu_pow[g])} for g, v in sorted(by_gpu.items())}
    # per-minute per-GPU mean inside the benchmark window; drop the partial
    # first/last minutes (window starts/ends mid-minute, e.g. 11:51:16.5)
    per_min = defaultdict(lambda: defaultdict(list))
    for ts, i, u, _ in rows:
        if t0 <= ts <= t1:
            per_min[datetime.fromtimestamp(ts, timezone.utc).strftime("%H:%M")][i].append(u)
    t0m = datetime.fromtimestamp(t0, timezone.utc).strftime("%H:%M")
    t1m = datetime.fromtimestamp(t1, timezone.utc).strftime("%H:%M")
    for pm in (t0m, t1m):
        if len(per_min) > 2:
            per_min.pop(pm, None)
    minmin = {m: {g: mean(v) for g, v in gmap.items()} for m, gmap in per_min.items()}
    win_vals = [v for gmap in minmin.values() for v in gmap.values()]
    mid_minute = sorted(minmin)[len(minmin) // 2] if minmin else None
    return {
        "n": len(rows), "per_gpu": per_g,
        "whole_mean_min": min(x["mean"] for x in per_g.values()) if per_g else float("nan"),
        "whole_mean_max": max(x["mean"] for x in per_g.values()) if per_g else float("nan"),
        "win_min": min(win_vals) if win_vals else float("nan"),
        "win_max": max(win_vals) if win_vals else float("nan"),
        "cross_spread_p50": pct(spreads, 50), "cross_spread_p95": pct(spreads, 95),
        "cross_spread_max": max(spreads) if spreads else 0,
        "p50_all": pct([x for v in by_gpu.values() for x in v], 50),
        "mid_minute": mid_minute,
        "mid_minute_vals": minmin.get(mid_minute, {}) if mid_minute else {},
        "power_min": min(x["w"] for x in per_g.values()) if per_g else float("nan"),
        "power_max": max(x["w"] for x in per_g.values()) if per_g else float("nan"),
    }


# ---------------------------------------------------------------- top-level
def analyze_run(run_dir):
    win = derive_window(run_dir)
    t0, t1 = win["t0"], win["t1"]
    fe = parse_frontend(run_dir)
    return {
        "run_dir": run_dir, "stamp": run_stamp(run_dir), "window": win,
        "client": client_metrics(run_dir),
        "input_len_median": input_len_median(run_dir),
        "engine": engine_minutes(run_dir, t0, t1),
        "frontend": fe,
        "join": join_frontend_engine(run_dir, fe, t0),
        "cost": router_cost(run_dir, fe),
        "rank": rank_ts(run_dir, t0, t1),
        "dynamo": dynamo_ts(run_dir, t0, t1),
        "gpu": gpu_stats(run_dir, t0, t1),
    }


def summary(rep):
    """Small dict of headline numbers for cross-run comparison."""
    c, e, r, d, g, j = (rep["client"], rep["engine"], rep["rank"], rep["dynamo"],
                        rep["gpu"], rep["join"])
    ap = d["active_prefill"]
    ap_vals = [v for v in ap.values() if v == v]
    usage = r["means"]["sglang:full_token_usage"]
    usage_vals = [v for v in usage.values() if v is not None]
    run_r = r["means"]["sglang:num_running_reqs"]
    run_vals = [v for v in run_r.values() if v is not None]
    return {
        "stamp": rep["stamp"],
        "throughput_tps": c["throughput"]["output_token_throughput_tps"],
        "ttft_mean_s": c["ttft"]["mean_ms"] / 1e3,
        "ttft_p99_s": c["ttft"]["p99_ms"] / 1e3,
        "run_mean_min": min(run_vals) if run_vals else float("nan"),
        "run_mean_max": max(run_vals) if run_vals else float("nan"),
        "pending_pm_min": e["pending_pm_min"], "pending_pm_max": e["pending_pm_max"],
        "prefill_spread_x": (max(ap_vals) / min(ap_vals)) if ap_vals and min(ap_vals) > 0 else float("nan"),
        "kv_mean_min": min(usage_vals) if usage_vals else float("nan"),
        "kv_mean_max": max(usage_vals) if usage_vals else float("nan"),
        "kv_spread_p50": r["usage_spread_p50"],
        "n_rank_full": len(r["first_100"]),
        "n_retract": len(e["retracts"]),
        "corr_logit_ttft": rep["cost"]["corr_logit_ttft"],
        "cache_reuse_pct": e["cache_reuse_pct"],
        "cached_tokens_req": d["cached_tokens"]["mean"],
        "accept_pm_min": e["accept_pm_min"], "accept_pm_max": e["accept_pm_max"],
        "itl_mean_ms": d["itl"]["mean"] * 1e3 if d["itl"]["n"] else float("nan"),
        "gpu_win_util_min": g["win_min"], "gpu_win_util_max": g["win_max"],
    }
