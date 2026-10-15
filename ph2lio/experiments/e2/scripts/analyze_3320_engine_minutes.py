import re
from collections import defaultdict

RUN = "/shared/loipp/sglang/ph2lio/experiments/e2/results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261014_112105"
ANSI = re.compile(r'\x1b\[[0-9;]*m')
ts_re = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})')
num = lambda m: float(m.group(1)) if m else None

# per-minute aggregation of Decode/Prefill batch lines
mins = defaultdict(lambda: {
    "n_dec": 0, "n_pre": 0, "usage": [], "running": [], "queue": [],
    "gen": [], "accept": [], "pending": [], "cached": 0, "newtok": 0,
    "occ": [], "in_thr": [],
})
with open(f"{RUN}/engine_engine_20261014_112107.log") as f:
    for line in f:
        line = ANSI.sub('', line)
        t = ts_re.match(line)
        if not t or "batch," not in line:
            continue
        mkey = t.group(1)[:16]  # per minute
        d = mins[mkey]
        if "Decode batch" in line:
            d["n_dec"] += 1
            d["usage"].append(num(re.search(r'token usage: ([\d.]+)', line)))
            d["running"].append(num(re.search(r'#running-req: (\d+)', line)))
            d["queue"].append(num(re.search(r'#queue-req: (\d+)', line)))
            d["gen"].append(num(re.search(r'gen throughput \(token/s\): ([\d.]+)', line)))
            d["accept"].append(num(re.search(r'accept len: ([\d.]+)', line)))
            d["occ"].append(num(re.search(r'fwd occupancy: ([\d.]+)%', line)))
        elif "Prefill batch" in line:
            d["n_pre"] += 1
            d["usage"].append(num(re.search(r'token usage: ([\d.]+)', line)))
            d["pending"].append(num(re.search(r'#pending-token: (\d+)', line)))
            c = num(re.search(r'#cached-token: (\d+)', line)) or 0
            n = num(re.search(r'#new-token: (\d+)', line)) or 0
            d["cached"] += c; d["newtok"] += n
            d["occ"].append(num(re.search(r'fwd occupancy: ([\d.]+)%', line)))
            d["in_thr"].append(num(re.search(r'input throughput \(token/s\): ([\d.]+)', line)))

mean = lambda v: sum(v) / len(v) if v else 0
print(f"{'minute':>5} {'dec':>4} {'pre':>4} {'usage':>5} {'run':>4} {'queue':>5} "
      f"{'gen':>6} {'acc':>4} {'occ%':>5} {'pend':>6} {'cache%':>6}")
for m in sorted(mins):
    d = mins[m]
    cache_pct = 100 * d["cached"] / (d["cached"] + d["newtok"]) if (d["cached"] + d["newtok"]) else 0
    print(f"{m[-5:]:>5} {d['n_dec']:>4} {d['n_pre']:>4} {mean(d['usage']):>5.2f} "
          f"{mean(d['running']):>4.1f} {mean(d['queue']):>5.1f} {mean(d['gen']):>6.1f} "
          f"{mean(d['accept']):>4.2f} {mean(d['occ']):>5.0f} {mean(d['pending']):>6.0f} {cache_pct:>6.0f}")

# global: cached share of all prefill tokens
tc = sum(d["cached"] for d in mins.values())
tn = sum(d["newtok"] for d in mins.values())
print(f"\ntotal prefill tokens: cached={tc/1e6:.2f}M new={tn/1e6:.2f}M -> cache reuse {100*tc/(tc+tn):.1f}%")
