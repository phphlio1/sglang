import csv
from collections import defaultdict
from datetime import datetime, timezone

RUN = "/shared/loipp/sglang/ph2lio/experiments/e2/results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261014_112105"
A = f"{RUN}/analysis"
T0, T1 = 1791978676.0, 1791980476.0  # benchmark 11:51:16 -> 12:21:16Z
fmt = lambda e: datetime.fromtimestamp(e, timezone.utc).strftime("%H:%M:%S")

# cumulative histograms: (metric base) -> key -> [(ts, sum, count)]
hist = defaultdict(lambda: defaultdict(list))
# gauges: metric -> key -> [(ts, value)]
gauge = defaultdict(lambda: defaultdict(list))

with open(f"{A}/dynamo_ts.csv") as f:
    for row in csv.DictReader(f):
        ts, m, key, val = float(row["timestamp"]), row["metric"], row["key"], float(row["value"])
        if not (T0 <= ts <= T1):
            continue
        if m.endswith("_sum"):
            hist[m[:-4]][key].append((ts, val, None))
        elif m.endswith("_count"):
            hist[m[:-6]][key].append((ts, None, val))
        else:
            gauge[m][key].append((ts, val))

print("== router overhead (histogram, per-routing-decision) ==")
for base in sorted(hist):
    for key, rows in sorted(hist[base].items()):
        rows.sort(key=lambda r: r[0])
        # gộp sum/count theo ts
        by_ts = {}
        for ts, s, c in rows:
            d = by_ts.setdefault(ts, {"s": None, "c": None})
            if s is not None: d["s"] = s
            if c is not None: d["c"] = c
        tss = sorted(by_ts)
        first, last = by_ts[tss[0]], by_ts[tss[-1]]
        ds = (last["s"] or 0) - (first["s"] or 0)
        dc = (last["c"] or 0) - (first["c"] or 0)
        if dc > 0:
            print(f"  {base} [{key or '-'}]: {dc:.0f} decisions, mean {ds/dc:.3f} ms/decision "
                  f"(window {fmt(tss[0])}->{fmt(tss[-1])})")

print("\n== frontend per-request cached_tokens (histogram) ==")
for key, rows in sorted(hist.get("dynamo_frontend_cached_tokens", {}).items()):
    by_ts = {}
    for ts, s, c in rows:
        d = by_ts.setdefault(ts, {"s": None, "c": None})
        if s is not None: d["s"] = s
        if c is not None: d["c"] = c
    tss = sorted(by_ts)
    first, last = by_ts[tss[0]], by_ts[tss[-1]]
    ds = (last["s"] or 0) - (first["s"] or 0)
    dc = (last["c"] or 0) - (first["c"] or 0)
    if dc > 0:
        print(f"  [{key or '-'}]: {dc:.0f} requests, mean cached {ds/dc:.0f} tokens/request")

print("\n== gauge: worker_active_prefill_tokens / active_decode_blocks per key ==")
for m in ("dynamo_frontend_worker_active_prefill_tokens", "dynamo_frontend_worker_active_decode_blocks"):
    for key, rows in sorted(gauge.get(m, {}).items()):
        v = [x for _, x in rows]
        print(f"  {m.split('worker_')[1]} key={key[:24]}: mean={sum(v)/len(v):.0f} max={max(v):.0f} n={len(v)}")

print("\n== gauge: router_queue_pending_requests / backpressure ==")
for m in ("dynamo_frontend_router_queue_pending_requests",
          "dynamo_frontend_router_queue_backpressure"):
    for key, rows in sorted(gauge.get(m, {}).items()):
        v = [x for _, x in rows]
        print(f"  {m.split('queue_')[1]} [{key[:48]}]: mean={sum(v)/len(v):.2f} max={max(v):.0f} n={len(v)}")
