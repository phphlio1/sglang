#!/usr/bin/env python3
"""Focus-2 retrace battery: reproduce every number in
docs/profile_bottleneck_5runs.md §4 (đính chính 2026-10-20) from raw logs.

Usage:
    python3 focus2_retrace.py [--part P] [<run_dir> ...]
    P in: formula, kvblind, pinned, drain, perreq, channels, hhsplit, verify, all
    (default: all; run_dirs default = 5 cell của consistency campaign)

Mỗi run_dir cần có:
    frontend_frontend_*.log    dòng "Selected worker" của dynamo router
    schedule_eval.jsonl        record per-request phía engine
    analysis/dynamo_ts.csv + analysis/sglang_rank_ts.csv
                               (tạo bằng extract_metrics_timeseries.py <run_dir>,
                                chạy srun trên node)

GOTCHA (quan trọng): schedule_eval.jsonl là file CHUNG bị append qua các run
cùng đợt — mọi part lọc record về window log của chính run đó
[events[0]-60, events[-1]+120] trước khi làm gì khác.

Window 2 mức (khác nhau có chủ đích):
- part `formula` dùng TOÀN BỘ record của run (không cắt warmup): câu hỏi là
  "logit có chứa term uncached không" — mọi request router thực sự chấm điểm
  đều là bằng chứng, gồm cả đoạn warmup (request turn-1 lớn, unc ~ isl).
- các part hiện tượng (pinned/drain/perreq/channels/hhsplit/verify) dùng bench
  window như focus report: cắt 5% warmup đầu + 1800 s.
(Bản tmp đầu của formula cắt warmup theo 5% của CẢ file chung — run 1 bị cắt
~25% while run 2–5 không cắt — semantics lẫn lẫn; số canonical là no-cut.)

Ý nghĩa từng part (chuỗi kết luận §4, kèm cite source):
    formula  logit 1.4.2 deployed CÓ term uncached của request đang đặt:
             corr(logit, state + 16*unc/64) >> corr(logit, state-only)
             — bác kết luận cũ "size-blind placement" (test cũ dùng input_len,
             không thông tin vì matched ~ 0.9*isl trong workload lặp history).
             Source công thức: dynamo 1.4.2 lib/kv-router/src/scheduling/
             selector.rs:138-308 + overlap.rs:201-238 + prefill_load.rs:17-19
             (weight 16 = --router-prefill-load-scale; credit tier host 0.75 /
             disk 0.25 = config.rs:42-48).
    kvblind  logit KHÔNG có term KV-occupancy: corr(logit, full_token_usage
             của rank được chọn) ~ 0/âm.
    pinned   ~90% request pinned về rank giữ prefix (device_matched > 0);
             inflow uncached của nhóm pinned chi phối tải per-rank.
    drain    gauge per-rank phía engine: qw correlate với backlog load-back,
             KHÔNG correlate với inflow. Metric đáng ngờ: prefill_delayer_
             outcomes (delta giống hệt mọi rank — misparse), prefill_effective_
             tokens (gauge không phải counter).
    perreq   corr per-request (qw, host/dev/isl/unc) + qw percentile theo nhóm;
             cơ chế chờ: host_hit_length cần reclaim + load-back toàn bộ span
             LÚC admission (schedule_policy.py:1908-1961 prepare_load_back +
             prefill_budget.py:40-58,338-364 reclaim_for_prealloc;
             schedule_batch.py:1187,1476-1482 needs_host_load_back).
    channels tách 2 kênh gây qw tail: host-tier load-back (per-request) vs
             KV>0.9 pressure (per-rank lúc chọn) — KV-full tự nó là thứ yếu.
    hhsplit  host-hit thuộc pinned hay new; chồng lấp retract (retract marginal:
             5 request / 5 run).
    verify   battery loại confounder: within-rank A/B (host>0 vs host=0 trên
             CÙNG rank), dose-response theo host_hit length, request lớn
             không host-hit vẫn nhanh, share request chậm qw>10s/30s, dung
             lượng hicache host tier của rank nóng.
"""
import argparse
import json
import os
import re
import sys
from bisect import bisect_left
from datetime import datetime, timezone

BASE = ("/shared/loipp/sglang/ph2lio/experiments/e2/results/glm53/"
        "agentx_pilot_preprod/agentx_dynamo/profiling/dfs-weight")
DEFAULT_RUNS = [
    "c64_20261014_112105", "c64_20261015_133031", "c64_20261015_144717",
    "c64_20261015_201358", "c64_20261015_224930",
]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z")
SEL_RE = re.compile(
    r'Selected worker router_mode="kv" request_id="([0-9a-f-]+)" '
    r'worker_id=\d+ worker_type=decode dp_rank=(\d+) logit=([\d.]+) '
    r'host_pinned_blocks=(\d+) disk_blocks=(\d+)')

# trường schedule_eval.jsonl (semantics: schedule_eval_log.py;
# queue_wait_s = first admission - first queue entry, stamp-once)
ISL, DEV, HOST = ("input_len", "device_matched_prefix_tokens_first_admission",
                  "host_hit_prefix_tokens_first_admission")
MATCH, QW, TTFT, RETR = ("matched_prefix_tokens_first_admission", "queue_wait_s",
                         "ttft_s", "num_retractions")


# ---------- helpers ----------

def epoch(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f").replace(
        tzinfo=timezone.utc).timestamp()


def corr(a, b):
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / n
    va = sum((x - ma) ** 2 for x in a) / n
    vb = sum((y - mb) ** 2 for y in b) / n
    return cov / (va ** .5 * vb ** .5) if va > 0 and vb > 0 else 0


def pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def sp(v):
    return f"{min(v):,.0f}-{max(v):,.0f} ({max(v)/max(min(v),1):.1f}x)"


def rget(x, k):
    return x.get(k) or 0


def unc_of(x):
    # unc = input_len - effective_cached; router credit host 0.75 (config.rs:42-48)
    return max(0, x[ISL] - rget(x, DEV) - 0.75 * rget(x, HOST))


def find_frontend_log(rd):
    for f in sorted(os.listdir(rd)):
        if f.startswith("frontend_frontend") and f.endswith(".log"):
            return f
    sys.exit(f"!! khong tim thay frontend_frontend_*.log trong {rd}")


def load_events(rd):
    """Dòng 'Selected worker' của frontend log -> (t, rid, rank, logit,
    host_blk, disk_blk) sorted theo t."""
    ev = []
    with open(os.path.join(rd, find_frontend_log(rd))) as f:
        for line in f:
            line = ANSI.sub("", line)
            t = TS_RE.match(line)
            m = SEL_RE.search(line)
            if t and m:
                ev.append((epoch(t.group(1)), m.group(1), int(m.group(2)),
                           float(m.group(3)), int(m.group(4)), int(m.group(5))))
    ev.sort(key=lambda e: e[0])
    return ev


def load_recs(rd, events):
    """schedule_eval.jsonl (file CHUNG append qua các run) -> chỉ giữ record
    trong window log của run này."""
    recs = []
    with open(os.path.join(rd, "schedule_eval.jsonl")) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    lo, hi = events[0][0] - 60, events[-1][0] + 120
    return [x for x in recs if lo <= (x.get("arrival_wall_s") or 0) <= hi]


def bench_window(recs):
    arr = sorted(x["arrival_wall_s"] for x in recs)
    lo = arr[int(len(arr) * 0.05)]
    return lo, min(lo + 1800, arr[-1])


def join_recs(recs, events, lo=None, hi=None):
    """Mỗi record -> event 'Selected worker' gần nhất ±2 s (một-một)."""
    used = set()
    out = []
    for x in sorted(recs, key=lambda r: r["arrival_wall_s"]):
        t = x["arrival_wall_s"]
        if (lo is not None and t < lo) or (hi is not None and t > hi):
            continue
        best, bd = None, 1e9
        for i, e in enumerate(events):
            if i in used:
                continue
            d = abs(e[0] - t)
            if d < bd:
                best, bd = i, d
        if best is not None and bd <= 2.0:
            used.add(best)
            out.append((x, events[best]))
    return out


def load_dynamo(rd, metrics):
    """dynamo_ts.csv (ts,metric,key,val; key '3|attn') -> {metric: {ts: {rank: v}}}"""
    out = {m: {} for m in metrics}
    with open(os.path.join(rd, "analysis", "dynamo_ts.csv")) as f:
        next(f)
        for line in f:
            ts, metric, key, val = line.rstrip("\n").split(",")
            head = key.split("|")[0]
            if metric in out and head.isdigit():
                out[metric].setdefault(float(ts), {})[int(head)] = float(val)
    return out


def load_sglang(rd, metrics):
    """sglang_rank_ts.csv (ts,metric,rank,val) -> {metric: {ts: {rank: v}}}"""
    out = {m: {} for m in metrics}
    with open(os.path.join(rd, "analysis", "sglang_rank_ts.csv")) as f:
        next(f)
        for line in f:
            ts, metric, rank, val = line.rstrip("\n").split(",")
            if metric in out:
                out[metric].setdefault(float(ts), {})[int(rank)] = float(val)
    return out


def rank_mean(series, rank, lo, hi):
    vals = [vs[rank] for ts, vs in series.items()
            if lo <= ts <= hi and rank in vs]
    return sum(vals) / len(vals) if vals else 0.0


def rank_delta(series, rank, lo, hi):
    ser = sorted((ts, vs[rank]) for ts, vs in series.items()
                 if lo <= ts <= hi and rank in vs)
    return ser[-1][1] - ser[0][1] if len(ser) >= 2 else 0.0


# ---------- parts ----------

PF_METRIC = "dynamo_frontend_worker_active_prefill_tokens"
DEC_METRIC = "dynamo_frontend_worker_active_decode_blocks"
KV_METRIC = "sglang:full_token_usage"


def part_formula(runs):
    """logit có term uncached không? A3/A4 (state + 16*unc/64) vs B (state-only)."""
    for rd, tag in runs:
        events = load_events(rd)
        dyn = load_dynamo(rd, (PF_METRIC, DEC_METRIC))
        pf_g, dec_g = dyn[PF_METRIC], dyn[DEC_METRIC]
        gauges = {}  # ts -> rank -> (pf_tok, dec_blk)
        for ts, ranks in pf_g.items():
            for r, p in ranks.items():
                d = dec_g.get(ts, {}).get(r)
                if d is not None:
                    gauges.setdefault(ts, {})[r] = (p, d)
        gts = sorted(gauges)

        recs = load_recs(rd, events)
        # no-cut: toàn bộ record của run (xem note window 2 mức ở header)
        A_lg, A_pred, A4_pred, B_lg, B_pred, inp, unc_tok = [], [], [], [], [], [], []
        dev_m, host_m, mat_m = [], [], []
        for x, ev in join_recs(recs, events):
            t, _rid, rank, logit, host_blk, disk_blk = ev
            i = bisect_left(gts, t)
            g = None
            for j in (i - 1, i):
                if 0 <= j < len(gts) and abs(gts[j] - t) <= 1.0:
                    gg = gauges[gts[j]].get(rank)
                    if gg:
                        g = gg
                        break
            if not g:
                continue
            pf_tok, dec_blk = g
            isl = x[ISL]
            dev, host = rget(x, DEV), rget(x, HOST)
            dev_m.append(dev); host_m.append(host); mat_m.append(rget(x, MATCH))
            eff3 = dev + 0.75 * host
            unc3 = isl - min(eff3, isl)
            eff4 = dev + 0.75 * 64 * host_blk + 0.25 * 64 * disk_blk
            unc4 = isl - min(eff4, isl)
            state = 16 * pf_tok / 64 + dec_blk
            A_lg.append(logit); A_pred.append(state + 16 * unc3 / 64)
            A4_pred.append(state + 16 * unc4 / 64)
            B_lg.append(logit); B_pred.append(state)
            inp.append(isl); unc_tok.append(16 * unc3 / 64)

        n = len(A_lg)
        print(f"=== formula {tag}  (n={n})")
        print(f"  device_matched={sum(dev_m)/n:,.0f}  host_hit={sum(host_m)/n:,.0f}  "
              f"matched_full={sum(mat_m)/n:,.0f}  isl={sum(inp)/n:,.0f}")
        print(f"  B  corr(logit, state-only)                          = {corr(B_lg, B_pred):+.3f}")
        print(f"  A3 corr(logit, state + 16*(isl-dev-0.75*host)/64)    = {corr(A_lg, A_pred):+.3f}")
        print(f"  A4 corr(logit, state + 16*unc4/64, host/disk tu log)  = {corr(A_lg, A4_pred):+.3f}")
        print(f"  corr(logit, 16*unc/64)                               = {corr(A_lg, unc_tok):+.3f}")
        print(f"  corr(logit, input_len)  (test cu, khong thong tin)   = {corr(A_lg, inp):+.3f}")


def part_kvblind(runs):
    """logit có thấy KV occupancy của rank được chọn không?"""
    for rd, tag in runs:
        events = load_events(rd)
        kv = load_sglang(rd, (KV_METRIC,))[KV_METRIC]
        kts = sorted(kv)
        lg, usage, per_rank, hi_n = [], [], {r: [] for r in range(8)}, 0
        for ev in events:
            t, _rid, rank, logit = ev[0], ev[1], ev[2], ev[3]
            i = bisect_left(kts, t)
            for j in (i - 1, i):
                if 0 <= j < len(kts) and abs(kts[j] - t) <= 2.0:
                    u = kv[kts[j]].get(rank)
                    if u is not None:
                        lg.append(logit); usage.append(u)
                        per_rank[rank].append(u)
                        if u > 0.9:
                            hi_n += 1
                    break
        print(f"=== kvblind {tag}  (n={len(lg)})")
        print(f"  corr(logit, full_token_usage(rank duoc chon)) = {corr(lg, usage):+.3f}")
        print(f"  mean usage (selected) = {sum(usage)/len(usage):.3f}  max = {max(usage):.3f}")
        print(f"  #dat vao rank >0.9 KV: {hi_n}/{len(lg)} ({100*hi_n/len(lg):.1f}%)")
        print("  usage mean per rank: " + " ".join(
            f"r{r}={sum(v)/len(v):.2f}" if v else f"r{r}=-" for r, v in per_rank.items()))


def part_pinned(runs):
    """per-rank: pinned (device_matched>0) vs new; inflow unc, gauge, qw/ttft."""
    for rd, tag in runs:
        events = load_events(rd)
        dyn = load_dynamo(rd, (PF_METRIC,))
        sg = load_sglang(rd, (KV_METRIC,))
        pf_g, kv_g = dyn[PF_METRIC], sg[KV_METRIC]
        recs = load_recs(rd, events)
        lo, hi = bench_window(recs)
        st = {r: dict(n=0, n_pin=0, n_new=0, unc_pin=0, unc_new=0, isl=0,
                      qw=[], ttft=[], retr=0) for r in range(8)}
        for x, ev in join_recs(recs, events, lo, hi):
            rank = ev[2]
            s = st[rank]
            s["n"] += 1; s["isl"] += x[ISL]
            s["qw"].append(rget(x, QW)); s["ttft"].append(rget(x, TTFT))
            s["retr"] += 1 if rget(x, RETR) > 0 else 0
            if rget(x, DEV) > 0:
                s["n_pin"] += 1; s["unc_pin"] += unc_of(x)
            else:
                s["n_new"] += 1; s["unc_new"] += unc_of(x)
        dur = hi - lo
        pf_m = {r: rank_mean(pf_g, r, lo, hi) for r in range(8)}
        kv_m = {r: rank_mean(kv_g, r, lo, hi) for r in range(8)}
        ranks = [r for r in sorted(st) if st[r]["n"] > 0]
        print(f"===== pinned {tag}  (recs={len(recs)}, win dur={dur:.0f}s, "
              f"joined={sum(st[r]['n'] for r in st)} req)")
        print("rank |   n  pin new | unc_pin/s unc_new/s | pf_mean kv_mean | qw_mean ttft_mean | retr")
        for r in ranks:
            s = st[r]
            print(f" r{r}  | {s['n']:4d} {s['n_pin']:4d} {s['n_new']:3d} "
                  f"| {s['unc_pin']/dur:8.0f} {s['unc_new']/dur:8.0f} "
                  f"| {pf_m[r]:7.0f} {kv_m[r]:.2f}  "
                  f"| {sum(s['qw'])/len(s['qw']):6.2f} {sum(s['ttft'])/len(s['ttft']):6.2f}  | {s['retr']}")
        up = [st[r]["unc_pin"] / dur for r in ranks]
        un = [st[r]["unc_new"] / dur for r in ranks]
        pfv = [pf_m[r] for r in ranks]
        kvv = [kv_m[r] for r in ranks]
        qv = [sum(st[r]["qw"]) / len(st[r]["qw"]) for r in ranks]
        isl_r = [st[r]["isl"] / dur for r in ranks]
        npin = [st[r]["n_pin"] for r in ranks]
        tot_n = sum(st[r]["n"] for r in ranks)
        print(f"  inflow unc_pinned tok/s: {sp(up)}   unc_new tok/s: {sp(un)}   isl tok/s: {sp(isl_r)}")
        print(f"  active pf gauge mean   : {sp(pfv)}   kv usage mean: {min(kvv):.2f}-{max(kvv):.2f}")
        print(f"  n_pinned per rank      : {sp(npin)}  (tong pinned {sum(npin)}/{tot_n} = "
              f"{100*sum(npin)/max(tot_n,1):.0f}%)")
        print(f"  corr cross-rank: pf~unc_pin={corr(up,pfv):+.2f}  pf~unc_new={corr(un,pfv):+.2f}  "
              f"pf~isl={corr(isl_r,pfv):+.2f}  qw~pf={corr(pfv,qv):+.2f}  kv~isl={corr(isl_r,kvv):+.2f}  "
              f"kv~unc_pin={corr(up,kvv):+.2f}")


# engine gauges (mean) / counters (delta trong window) cho part drain
DRAIN_GAUGES = [
    KV_METRIC, "sglang:hicache_host_used_tokens", "sglang:num_queue_reqs",
    "sglang:num_prefill_inflight_queue_reqs", "sglang:num_prefill_bootstrap_queue_reqs",
    "sglang:num_running_reqs", "sglang:num_paused_reqs", "sglang:pending_prealloc_token_usage",
]
# 3 metric này ĐÁNG NGỜ (đã loại khỏi kết luận): delayer_outcomes delta giống
# hệt mọi rank (misparse histogram); prefill_effective_tokens là gauge
# (non-monotone) không phải counter; num_retracted_reqs là gauge currently-
# retracted không phải cumulative — số retract dùng per-request num_retractions
# trong schedule_eval.jsonl (xem part pinned/verify).
DRAIN_SUSPECT = ["sglang:num_retracted_reqs", "sglang:prefill_delayer_outcomes",
                 "sglang:prefill_effective_tokens"]


def part_drain(runs):
    """per-rank engine gauges + corr(qw/pf/kv, X) — drain chậm vì backlog nào?"""
    for rd, tag in runs:
        events = load_events(rd)
        dyn = load_dynamo(rd, (PF_METRIC,))
        sg = load_sglang(rd, tuple(DRAIN_GAUGES + DRAIN_SUSPECT))
        pf_g = dyn[PF_METRIC]
        recs = load_recs(rd, events)
        lo, hi = bench_window(recs)
        st = {r: dict(n=0, n_pin=0, unc_pin=0, unc_new=0, host=0, qw=[], ttft=[])
              for r in range(8)}
        for x, ev in join_recs(recs, events, lo, hi):
            rank = ev[2]
            s = st[rank]
            s["n"] += 1
            s["qw"].append(rget(x, QW)); s["ttft"].append(rget(x, TTFT))
            s["host"] += rget(x, HOST)
            if rget(x, DEV) > 0:
                s["n_pin"] += 1; s["unc_pin"] += unc_of(x)
            else:
                s["unc_new"] += unc_of(x)
        dur = hi - lo
        ranks = [r for r in range(8) if st[r]["n"] > 0]
        row = {}
        for r in ranks:
            s = st[r]
            peff_d = rank_delta(sg["sglang:prefill_effective_tokens"], r, lo, hi)
            row[r] = dict(
                n=s["n"], pin=s["n_pin"], up=s["unc_pin"] / dur, un=s["unc_new"] / dur,
                ho=s["host"] / dur,
                qw=sum(s["qw"]) / len(s["qw"]), ttft=sum(s["ttft"]) / len(s["ttft"]),
                pf=rank_mean(pf_g, r, lo, hi),
                kv=rank_mean(sg[KV_METRIC], r, lo, hi),
                hk=rank_mean(sg["sglang:hicache_host_used_tokens"], r, lo, hi),
                q=rank_mean(sg["sglang:num_queue_reqs"], r, lo, hi),
                inf=rank_mean(sg["sglang:num_prefill_inflight_queue_reqs"], r, lo, hi),
                boot=rank_mean(sg["sglang:num_prefill_bootstrap_queue_reqs"], r, lo, hi),
                run=rank_mean(sg["sglang:num_running_reqs"], r, lo, hi),
                pau=rank_mean(sg["sglang:num_paused_reqs"], r, lo, hi),
                retr=rank_delta(sg["sglang:num_retracted_reqs"], r, lo, hi),
                dely=rank_delta(sg["sglang:prefill_delayer_outcomes"], r, lo, hi),
                peff=peff_d / dur if peff_d >= 0 else float("nan"),
            )
        print(f"===== drain {tag}  (win dur={dur:.0f}s, joined={sum(st[r]['n'] for r in ranks)} req)")
        print("rank |   n pin | uncP/s uncN/s hostT/s |  qw   ttft | pfstock | kv  hostK "
              "| qR  infl boot run  pau | retr dely peff  (3 cot cuoi = metric dang ngo)")
        for r in ranks:
            x = row[r]
            print(f" r{r}  | {x['n']:4d} {x['pin']:4d} | {x['up']:6.0f} {x['un']:6.0f} {x['ho']:7.0f} "
                  f"| {x['qw']:5.2f} {x['ttft']:6.2f} | {x['pf']:7.0f} | {x['kv']:.2f} {x['hk']:5.0f} "
                  f"| {x['q']:4.1f} {x['inf']:4.1f} {x['boot']:4.1f} {x['run']:4.1f} {x['pau']:4.1f} "
                  f"| {x['retr']:4.0f} {x['dely']:5.0f} | {x['peff']:7.0f}")

        def L(k):
            return [row[r][k] for r in ranks]
        qwv, pfv, kvv = L("qw"), L("pf"), L("kv")
        print("  corr(qw, X): " + "  ".join(
            f"{k}={corr(qwv, L(k)):+.2f}" for k in
            ("up", "un", "ho", "pf", "kv", "hk", "q", "inf", "boot", "run", "pau", "peff")))
        print("  corr(pf, X): " + "  ".join(
            f"{k}={corr(pfv, L(k)):+.2f}" for k in
            ("up", "un", "ho", "kv", "hk", "q", "inf", "boot", "run", "peff")))
        print("  corr(kv, X): " + "  ".join(
            f"{k}={corr(kvv, L(k)):+.2f}" for k in ("up", "un", "ho", "pf", "hk", "q", "boot", "run")))


def part_perreq(runs):
    """per-request: corr(qw, host/dev/isl/unc) + qw percentile theo nhóm +
    timeline 100s rank xau nhat vs tot nhat."""
    for rd, tag in runs:
        events = load_events(rd)
        dyn = load_dynamo(rd, (PF_METRIC,))
        sg = load_sglang(rd, (KV_METRIC, "sglang:num_queue_reqs",
                              "sglang:hicache_host_used_tokens"))
        pf_g = dyn[PF_METRIC]
        kv_g = sg[KV_METRIC]
        q_g = sg["sglang:num_queue_reqs"]
        hk_g = sg["sglang:hicache_host_used_tokens"]
        recs = load_recs(rd, events)
        lo, hi = bench_window(recs)
        rows = []
        for x, ev in join_recs(recs, events, lo, hi):
            rows.append(dict(t=x["arrival_wall_s"], rank=ev[2], isl=x[ISL],
                             dev=rget(x, DEV), host=rget(x, HOST), unc=unc_of(x),
                             qw=rget(x, QW), ttft=rget(x, TTFT), retr=rget(x, RETR)))
        qw = [r["qw"] for r in rows]
        print(f"===== perreq {tag}  (n={len(rows)}, qw p50={pct(qw,.5):.2f} "
              f"p90={pct(qw,.9):.2f} p99={pct(qw,.99):.2f})")
        print(f"  corr(qw, X)  : host={corr(qw,[r['host'] for r in rows]):+.3f}  "
              f"dev={corr(qw,[r['dev'] for r in rows]):+.3f}  "
              f"isl={corr(qw,[r['isl'] for r in rows]):+.3f}  "
              f"unc={corr(qw,[r['unc'] for r in rows]):+.3f}")
        print(f"  corr(ttft, X): host={corr([r['ttft'] for r in rows],[r['host'] for r in rows]):+.3f}  "
              f"isl={corr([r['ttft'] for r in rows],[r['isl'] for r in rows]):+.3f}")
        for name, sel in (("host>0", lambda r: r["host"] > 0),
                          ("host=0", lambda r: r["host"] == 0),
                          ("pinned", lambda r: r["dev"] > 0),
                          ("new", lambda r: r["dev"] == 0)):
            g = [r["qw"] for r in rows if sel(r)]
            print(f"  qw {name:7s}: n={len(g):4d}  mean={sum(g)/max(len(g),1):5.2f}  "
                  f"p50={pct(g,.5):5.2f}  p90={pct(g,.9):5.2f}  p99={pct(g,.99):5.2f}")
        rk = {}
        for r in rows:
            rk.setdefault(r["rank"], []).append(r)
        print("rank |   n | qw p50  p90 | host%  hostT mean | dev mean | isl mean | ttft p90")
        for k in sorted(rk):
            g = rk[k]
            print(f" r{k}  | {len(g):4d} | {pct([r['qw'] for r in g],.5):6.2f} "
                  f"{pct([r['qw'] for r in g],.9):6.2f} "
                  f"| {100*sum(1 for r in g if r['host']>0)/len(g):5.1f} "
                  f"{sum(r['host'] for r in g)/len(g):10.0f} "
                  f"| {sum(r['dev'] for r in g)/len(g):9.0f} | {sum(r['isl'] for r in g)/len(g):8.0f} "
                  f"| {pct([r['ttft'] for r in g],.9):7.2f}")
        worst = max(rk, key=lambda k: pct([r["qw"] for r in rk[k]], .9))
        best = min(rk, key=lambda k: pct([r["qw"] for r in rk[k]], .9))
        print(f"  timeline 100s (rank xau nhat r{worst} vs tot nhat r{best}):")
        print("      t   | nR  qwMean | pfstock   q  kv hostK |  nB  qwMean | pfstock   q  kv hostK")

        def cell(k, b0, b1):
            g = [r for r in rk.get(k, []) if b0 <= r["t"] < b1]
            pts = [ts for ts in pf_g if b0 <= ts < b1]
            pf = sum(pf_g[ts].get(k, 0) for ts in pts) / max(len(pts), 1)

            def m(d):
                ss = [d[ts][k] for ts in pts if k in d.get(ts, {})]
                return sum(ss) / len(ss) if ss else 0.0
            qv, kvv, hk = m(q_g), m(kv_g), m(hk_g)
            mm = sum(r["qw"] for r in g) / max(len(g), 1)
            return f"{len(g):3d} {mm:6.2f} | {pf:7.0f} {qv:4.1f} {kvv:.2f} {hk/1e6:5.2f}"
        for b0 in range(int(lo), int(hi), 100):
            b1 = b0 + 100
            print(f"  {int(b0-lo):6d} | {cell(worst,b0,b1)} | {cell(best,b0,b1)}")


def part_channels(runs):
    """tách kênh: host-tier load-back (per-request) vs KV>0.9 (per-rank lúc chọn);
    share tổng qw-giây của host-hit."""
    TOT = dict(n=0, qwsum=0, hh_n=0, hh_qwsum=0, hi_n=0)
    for rd, tag in runs:
        events = load_events(rd)
        kv = load_sglang(rd, (KV_METRIC,))[KV_METRIC]
        kts = sorted(kv)
        recs = load_recs(rd, events)
        lo, hi = bench_window(recs)
        rows = []
        for x, ev in join_recs(recs, events, lo, hi):
            t, rank = ev[0], ev[2]
            i = bisect_left(kts, t)
            u = None
            for j in (i - 1, i):
                if 0 <= j < len(kts) and abs(kts[j] - t) <= 5.0:
                    u = kv[kts[j]].get(rank)
                    if u is not None:
                        break
            if u is None:
                continue
            rows.append(dict(host=rget(x, HOST), kv=u, qw=rget(x, QW)))
        hh = [r for r in rows if r["host"] > 0]
        h0 = [r for r in rows if r["host"] == 0]
        qwsum = sum(r["qw"] for r in rows)
        TOT["n"] += len(rows); TOT["qwsum"] += qwsum
        TOT["hh_n"] += len(hh); TOT["hh_qwsum"] += sum(r["qw"] for r in hh)
        TOT["hi_n"] += sum(1 for r in rows if r["kv"] > 0.9)
        print(f"===== channels {tag}  (n={len(rows)})")
        print(f"  share tong qw-giay cua host-hit: "
              f"{100*sum(r['qw'] for r in hh)/max(qwsum,1e-9):.1f}% "
              f"(host-hit {len(hh)}/{len(rows)} = {100*len(hh)/len(rows):.1f}% req)")
        for name, g in (("host>0", hh), ("host=0", h0)):
            print(f"  corr(qw, kv_rank luc chon) [{name}]: n={len(g):4d}  "
                  f"corr={corr([r['qw'] for r in g],[r['kv'] for r in g]):+.3f}")
        print("  bucket                |    n | qw mean  p90 | kv mean")
        for name, sel in (("host>0 & kv>0.9", lambda r: r["host"] > 0 and r["kv"] > 0.9),
                          ("host>0 & kv<0.9", lambda r: r["host"] > 0 and r["kv"] <= 0.9),
                          ("host=0 & kv>0.9", lambda r: r["host"] == 0 and r["kv"] > 0.9),
                          ("host=0 & kv<0.9", lambda r: r["host"] == 0 and r["kv"] <= 0.9)):
            g = [r for r in rows if sel(r)]
            if not g:
                print(f"  {name:22s} |    0 |")
                continue
            print(f"  {name:22s} | {len(g):4d} | {sum(r['qw'] for r in g)/len(g):7.2f} "
                  f"{pct([r['qw'] for r in g],.9):6.2f} | {sum(r['kv'] for r in g)/len(g):.2f}")
    print(f"\n===== channels TONG {len(runs)} run: n={TOT['n']}, host-hit {TOT['hh_n']} "
          f"({100*TOT['hh_n']/TOT['n']:.1f}%) gom {100*TOT['hh_qwsum']/TOT['qwsum']:.1f}% tong qw-giay; "
          f"dat vao rank kv>0.9: {TOT['hi_n']} ({100*TOT['hi_n']/TOT['n']:.1f}%)")


def part_hhsplit(runs):
    """host-hit thuộc pinned hay new? qw/ttft mean 4 nhóm + chồng lấp retract."""
    for rd, tag in runs:
        events = load_events(rd)
        recs = load_recs(rd, events)
        lo, hi = bench_window(recs)
        b = {k: dict(n=0, q=0.0, tt=0.0) for k in ("hh_pin", "hh_new", "hp_pin", "hp_new")}
        rtr = dict(n=0, h=0)
        for x, _ev in join_recs(recs, events, lo, hi):
            dev, host = rget(x, DEV), rget(x, HOST)
            if host > 0:
                k = "hh_pin" if dev > 0 else "hh_new"
            else:
                k = "hp_pin" if dev > 0 else "hp_new"
            b[k]["n"] += 1
            b[k]["q"] += rget(x, QW)
            b[k]["tt"] += rget(x, TTFT)
            if rget(x, RETR) > 0:
                rtr["n"] += 1
                rtr["h"] += 1 if host > 0 else 0

        def s(k):
            g = b[k]
            return f"{k}: n={g['n']:4d} qw={g['q']/max(g['n'],1):6.2f} ttft={g['tt']/max(g['n'],1):6.2f}"
        print(f"== hhsplit {tag}: " + " | ".join(s(k) for k in b)
              + f"  || retract {rtr['n']} req (host-hit {rtr['h']})")


def part_verify(runs):
    """battery loại confounder (A within-rank, B dose-response, C size,
    D slow-share, E host-tier capacity)."""
    print("=== verify A. WITHIN-RANK: qw host>0 vs host=0 tren cung rank")
    for rd, tag in runs:
        events = load_events(rd)
        recs = load_recs(rd, events)
        lo, hi = bench_window(recs)
        per_rank = {}
        for x, ev in join_recs(recs, events, lo, hi):
            per_rank.setdefault(ev[2], []).append(
                (rget(x, HOST), rget(x, QW), x[ISL], rget(x, DEV)))
        lines, slow10, slow30, n_all, hh_n = [], [], [], 0, 0
        dose = {k: [] for k in ("0", "1-10k", "10-40k", "40-80k", "80k+")}
        for rank in sorted(per_rank):
            g = per_rank[rank]
            hh = [r for r in g if r[0] > 0]
            h0 = [r for r in g if r[0] == 0]
            if hh and h0:
                rm_hh = sum(r[1] for r in hh) / len(hh)
                rm_h0 = sum(r[1] for r in h0) / len(h0)
                lines.append(f"r{rank}:{rm_hh:5.1f}/{rm_h0:4.1f}"
                             f"({rm_hh/max(rm_h0,1e-9):4.1f}x,{len(hh)}/{len(h0)})")
            for host, qw, isl, dev in g:
                n_all += 1
                if host > 0:
                    hh_n += 1
                    k = ("0" if host < 64 else "1-10k" if host < 10_000 else
                         "10-40k" if host < 40_000 else "80k+" if host >= 80_000
                         else "40-80k")
                    dose[k].append((qw, host))
                if qw > 10:
                    slow10.append(host > 0)
                if qw > 30:
                    slow30.append(host > 0)
        print(f"  {tag}: " + "  ".join(lines))
        print(f"    D. qw>10s: {sum(slow10)}/{len(slow10)} = "
              f"{100*sum(slow10)/max(len(slow10),1):.0f}% la host-hit   |   "
              f"qw>30s: {sum(slow30)}/{len(slow30)} = "
              f"{100*sum(slow30)/max(len(slow30),1):.0f}%")
        print("    B. dose-response qw theo host_hit length: ", end="")
        for k in ("0", "1-10k", "10-40k", "40-80k", "80k+"):
            v = dose[k]
            if v:
                m = sum(q for q, _ in v) / len(v)
                hl = sum(h for _, h in v) / len(v)
                print(f"{k}(n={len(v)},host~{hl/1000:.0f}k):qw={m:5.1f}s  ", end="")
        print()
        hh_all = [r for rank in per_rank for r in per_rank[rank] if r[0] > 0]
        h0_all = [r for rank in per_rank for r in per_rank[rank] if r[0] == 0]
        print(f"    C. isl mean: host>0 = {sum(r[2] for r in hh_all)/len(hh_all):,.0f}  "
              f"host=0 = {sum(r[2] for r in h0_all)/len(h0_all):,.0f}   | "
              f"dev mean: {sum(r[3] for r in hh_all)/len(hh_all):,.0f} vs "
              f"{sum(r[3] for r in h0_all)/len(h0_all):,.0f}")
        # C2. confounder 'request to': request lon (isl>=120k) KHONG host-hit van nhanh?
        big_h0 = [r[1] for r in h0_all if r[2] >= 120_000]
        big_hh = [r[1] for r in hh_all if r[2] >= 120_000]
        print(f"    C2. isl>=120k: host=0 qw mean = {sum(big_h0)/max(len(big_h0),1):.1f}s (n={len(big_h0)})  "
              f"vs host>0 = {sum(big_hh)/max(len(big_hh),1):.1f}s (n={len(big_hh)})")

    print("\n=== verify E. hicache host tier: muc dung cuoi window vs capacity (rank)"
          "  [rank nóng = qw p90 cao nhat]")
    for rd, tag in runs:
        sg = load_sglang(rd, ("sglang:hicache_host_total_tokens",
                              "sglang:hicache_host_used_tokens"))

        def per_rank_series(series):
            out = {}
            for ts, ranks in series.items():
                for r, v in ranks.items():
                    out.setdefault(r, []).append((ts, v))
            return out
        tot, used = (per_rank_series(sg["sglang:hicache_host_total_tokens"]),
                     per_rank_series(sg["sglang:hicache_host_used_tokens"]))
        out = []
        for rank in sorted(tot):
            t2 = sorted(tot[rank])
            u2 = sorted(used.get(rank, []))
            if not t2 or not u2:
                continue
            last_t, cap = t2[-1]
            uu = [v for t, v in u2 if abs(t - last_t) < 60]
            if uu and cap > 0:
                out.append(f"r{rank}: {uu[-1]/1e6:.2f}/{cap/1e6:.2f}M ({100*uu[-1]/cap:.0f}%)")
        print(f"  {tag}: " + "  ".join(out))


PARTS = {
    "formula": part_formula, "kvblind": part_kvblind, "pinned": part_pinned,
    "drain": part_drain, "perreq": part_perreq, "channels": part_channels,
    "hhsplit": part_hhsplit, "verify": part_verify,
}
# thứ tự chạy = thứ tự chuỗi kết luận §4
ORDER = ["formula", "kvblind", "pinned", "drain", "perreq",
         "channels", "hhsplit", "verify"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", default="all",
                    choices=ORDER + ["all"],
                    help="part nào của battery (mặc định: all)")
    ap.add_argument("run_dirs", nargs="*",
                    help="run dir(s); mặc định = 5 cell consistency campaign dưới BASE")
    args = ap.parse_args()
    if args.run_dirs:
        runs = [(rd, os.path.basename(rd.rstrip("/"))) for rd in args.run_dirs]
    else:
        runs = [(os.path.join(BASE, s), s) for s in DEFAULT_RUNS]
    parts = ORDER if args.part == "all" else [args.part]
    for p in parts:
        PARTS[p](runs)


if __name__ == "__main__":
    main()
