import json, re
from collections import defaultdict
from datetime import datetime, timezone

RUN = "/shared/loipp/sglang/ph2lio/experiments/e2/results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261014_112105"
ANSI = re.compile(r'\x1b\[[0-9;]*m')
ts_re = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z')
sel_re = re.compile(r'Selected worker router_mode="kv" request_id="([0-9a-f-]+)" worker_id=\d+ worker_type=decode dp_rank=(\d+) logit=([\d.]+)(?: overlap_blocks=(\d+))?')

events = []
with open(f"{RUN}/frontend_frontend_20261014_112107.log") as f:
    for line in f:
        line = ANSI.sub('', line)
        t = ts_re.match(line)
        m = sel_re.search(line)
        if t and m:
            epoch = datetime.strptime(t.group(1), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            events.append((epoch, m.group(1), int(m.group(2)), float(m.group(3)),
                           float(m.group(4) or 0)))
events.sort()
print(f"selector events: {len(events)}, overlap_blocks present: {sum(1 for e in events if e[4] > 0)}")

recs = [json.loads(l) for l in open(f"{RUN}/schedule_eval.jsonl") if l.strip()]
arr = sorted(x["arrival_wall_s"] for x in recs)
warmup_end = arr[int(len(arr) * 0.05)]

used = set()
def nearest(epoch):
    best, bd = None, 1e9
    for i, e in enumerate(events):
        if i in used:
            continue
        d = abs(e[0] - epoch)
        if d < bd:
            best, bd = i, d
    if best is not None and bd <= 2.0:
        used.add(best)
        return events[best]
    return None

rows = []  # (logit, overlap_blocks_pred, qw, ttft, matched_prefix, input_len, rank)
for x in sorted(recs, key=lambda r: r["arrival_wall_s"]):
    if x["arrival_wall_s"] < warmup_end:
        continue
    ev = nearest(x["arrival_wall_s"])
    if ev is None:
        continue
    _, rid, rank, logit, overlap = ev
    rows.append((logit, overlap, x["queue_wait_s"], x["ttft_s"],
                 x.get("matched_prefix_tokens_first_admission", 0) + x.get("device_matched_prefix_tokens_first_admission", 0),
                 x["input_len"], rank))

n = len(rows)
print(f"matched: {n}")

def corr(a, b):
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / n
    va = sum((x - ma) ** 2 for x in a) / n
    vb = sum((y - mb) ** 2 for y in b) / n
    return cov / (va ** 0.5 * vb ** 0.5) if va > 0 and vb > 0 else 0

logit = [r[0] for r in rows]; qw = [r[2] for r in rows]; ttft = [r[3] for r in rows]
match = [r[4] for r in rows]; inl = [r[5] for r in rows]
print(f"corr(logit, queue_wait)  = {corr(logit, qw):+.3f}")
print(f"corr(logit, ttft)        = {corr(logit, ttft):+.3f}")
print(f"corr(logit, input_len)   = {corr(logit, inl):+.3f}")
print(f"corr(input_len, qw)      = {corr(inl, qw):+.3f}")
print(f"corr(matched_prefix, qw) = {corr(match, qw):+.3f}")

# bucketed: does high predicted cost (logit) actually mean high realized wait?
rows.sort(key=lambda r: r[0])
k = n // 5
print(f"\nlogit quintile -> realized outcome (median):")
print(f"{'bucket':>14} {'logit_p50':>9} {'qw_p50':>7} {'ttft_p50':>8} {'inlen_p50':>9} {'match_p50':>9}")
def med(v):
    s = sorted(v); return s[len(s)//2]
for b in range(5):
    grp = rows[b*k:(b+1)*k] if b < 4 else rows[4*k:]
    print(f"Q{b+1} ({len(grp):>3} req) {med([r[0] for r in grp]):>9.0f} {med([r[2] for r in grp]):>7.2f} "
          f"{med([r[3] for r in grp]):>8.2f} {med([r[5] for r in grp]):>9.0f} {med([r[4] for r in grp]):>9.0f}")
