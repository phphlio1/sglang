import json, re, sys
from collections import defaultdict
from datetime import datetime, timezone

RUN = "/shared/loipp/sglang/ph2lio/experiments/e2/results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261014_112105"
ANSI = re.compile(r'\x1b\[[0-9;]*m')

ts_re = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z')
sel_re = re.compile(r'Selected worker router_mode="kv" request_id="([0-9a-f-]+)" worker_id=\d+ worker_type=decode dp_rank=(\d+) logit=([\d.]+)')
com_re = re.compile(r'request completed request_id=([0-9a-f-]+) .*ttft_ms="?([\d.]+)"?')

events = []  # (epoch, rid, rank, logit)
ttft_fe = {}  # rid -> frontend-observed ttft_s
with open(f"{RUN}/frontend_frontend_20261014_112107.log") as f:
    for line in f:
        line = ANSI.sub('', line)
        t = ts_re.match(line)
        if not t:
            continue
        epoch = datetime.strptime(t.group(1), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        m = sel_re.search(line)
        if m:
            events.append((epoch, m.group(1), int(m.group(2)), float(m.group(3))))
            continue
        m = com_re.search(line)
        if m:
            ttft_fe[m.group(1)] = float(m.group(2)) / 1000.0

events.sort()
print("selector events:", len(events), "| frontend ttft records:", len(ttft_fe))

recs = [json.loads(l) for l in open(f"{RUN}/schedule_eval.jsonl") if l.strip()]
arr = sorted(x["arrival_wall_s"] for x in recs)
warmup_end = arr[int(len(arr) * 0.05)]

# greedy nearest-time one-to-one match: engine arrival <-> frontend selector
used = set()
def nearest(epoch):
    best, bd = None, 1e9
    for i, (e, rid, rank, logit) in enumerate(events):
        if i in used:
            continue
        d = abs(e - epoch)
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
    _, rid, rank, logit = ev
    stats[rank]["ttft"].append(x["ttft_s"])
    stats[rank]["qw"].append(x["queue_wait_s"])
    stats[rank]["in"].append(x["input_len"])
    if rid in ttft_fe:
        fe_ttft.append(ttft_fe[rid]); en_ttft.append(x["ttft_s"])

def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p / 100))]

print(f"matched (post-warmup): {n_match}")
print(f"{'rank':>4} {'n':>4} {'ttft_mean':>9} {'ttft_p95':>8} {'ttft_p99':>8} {'qw_mean':>8} {'qw_p95':>8} {'in_mean':>8}")
for r in sorted(stats):
    s = stats[r]
    if not s["ttft"]:
        continue
    print(f"{r:>4} {len(s['ttft']):>4} {sum(s['ttft'])/len(s['ttft']):>9.2f} {pct(s['ttft'],95):>8.2f} {pct(s['ttft'],99):>8.2f} {sum(s['qw'])/len(s['qw']):>8.2f} {pct(s['qw'],95):>8.2f} {sum(s['in'])/len(s['in']):>8.0f}")

if len(fe_ttft) > 20:
    diffs = sorted(a - b for a, b in zip(fe_ttft, en_ttft))
    n = len(diffs)
    print(f"\nfrontend ttft vs engine ttft (n={n}): diff mean={sum(diffs)/n:.3f}s p50={diffs[n//2]:.3f}s p95={diffs[int(n*0.95)]:.3f}s")
