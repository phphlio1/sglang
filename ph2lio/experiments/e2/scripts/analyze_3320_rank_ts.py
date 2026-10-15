import csv, os
from collections import defaultdict

RUN = "/shared/loipp/sglang/ph2lio/experiments/e2/results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261014_112105"
A = f"{RUN}/analysis"
# benchmark window: 2026-10-14 11:51:16Z -> 12:21:16Z (profile_export request_start_ns:
# gap 82.5s giữa warmup cuối 11:49:54 và benchmark đầu 11:51:16; duration 1800s)
T0, T1 = 1791978676.0, 1791980476.0

def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p / 100))]

# ---- sglang per-rank (8081) ----
per = defaultdict(lambda: defaultdict(list))   # rank -> metric -> [v]
by_ts = defaultdict(lambda: defaultdict(dict)) # ts -> metric -> rank -> v
n_sg = 0
with open(f"{A}/sglang_rank_ts.csv") as f:
    for row in csv.DictReader(f):
        n_sg += 1
        ts, metric, rank, val = float(row["timestamp"]), row["metric"], int(row["dp_rank"]), float(row["value"])
        if not (T0 <= ts <= T1):
            continue
        per[rank][metric].append(val)
        by_ts[ts][metric][rank] = val

print(f"sglang rows: {n_sg}, in-window ts: {len(by_ts)}")
METRICS = ["sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:full_token_usage",
           "sglang:pending_prealloc_token_usage", "sglang:cache_hit_rate",
           "sglang:kv_used_tokens", "sglang:hicache_host_used_tokens",
           "sglang:num_prefill_inflight_queue_reqs", "sglang:num_retracted_reqs"]
hdr = f"{'metric':<38}" + "".join(f"r{r:<6}" for r in range(8))
print("\n== per-rank benchmark-window (mean | p95 hoặc max) ==")
print(hdr)
for m in METRICS:
    means = [f"{sum(per[r][m])/len(per[r][m]):.2f}" if per[r][m] else "-" for r in range(8)]
    print(f"{m:<38}" + "".join(f"{v:<7}" for v in means))
    if m in ("sglang:num_retracted_reqs",):
        maxs = [f"max={max(per[r][m]):.0f}" if per[r][m] else "-" for r in range(8)]
        print(f"{'  (max)':<38}" + "".join(f"{v:<7}" for v in maxs))

# ---- cross-rank imbalance per timestamp ----
run_spread, usage_spread, zero_vs_busy = [], [], 0
usage_first_100 = {}
for ts in sorted(by_ts):
    m = by_ts[ts]
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
            if v >= 0.999 and r not in usage_first_100:
                usage_first_100[r] = ts
n_ts = len(by_ts)
print(f"\n== imbalance (qua {n_ts} timestamp, 8 rank) ==")
if run_spread:
    print(f"num_running_reqs spread (max-min): p50={pct(run_spread,50):.0f} p95={pct(run_spread,95):.0f} max={max(run_spread):.0f}")
    print(f"timestamps có rank 0-running trong khi rank khác >=3 running: {zero_vs_busy}/{n_ts} ({100*zero_vs_busy/max(n_ts,1):.1f}%)")
if usage_spread:
    print(f"full_token_usage spread: p50={pct(usage_spread,50):.2f} p95={pct(usage_spread,95):.2f} max={max(usage_spread):.2f}")
if usage_first_100:
    from datetime import datetime, timezone
    for r, ts in sorted(usage_first_100.items(), key=lambda x: x[1]):
        print(f"rank {r} chạm token usage >=1.00 lúc {datetime.fromtimestamp(ts, timezone.utc).strftime('%H:%M:%S')}")

# ---- prefill delayer outcomes: cấu trúc thô ----
dl = defaultdict(list)
with open(f"{A}/sglang_rank_ts.csv") as f:
    for row in csv.DictReader(f):
        if row["metric"] == "sglang:prefill_delayer_outcomes" and T0 <= float(row["timestamp"]) <= T1:
            dl[int(row["dp_rank"])].append(float(row["value"]))
print(f"\nprefill_delayer_outcomes rows/rank: " + ", ".join(f"r{r}:{len(v)}" for r, v in sorted(dl.items())))
for r in sorted(dl):
    v = dl[r]
    print(f"  rank {r}: min={min(v):.0f} max={max(v):.0f} last={v[-1]:.0f}")

# ---- dynamo (8888) inventory ----
print(f"\n== dynamo_ts.csv (8888) ==")
names = defaultdict(int)
with open(f"{A}/dynamo_ts.csv") as f:
    for row in csv.DictReader(f):
        names[row["metric"]] += 1
for m, c in sorted(names.items()):
    print(f"  {m}: {c} rows")
