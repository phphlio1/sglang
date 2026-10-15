import csv
from collections import defaultdict

RUN = "/shared/loipp/sglang/ph2lio/experiments/e2/results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261014_112105"

rows = []
with open(f"{RUN}/gpu_samples.csv") as f:
    for r in csv.DictReader(f):
        rows.append((r["timestamp"], int(r[" index"]), float(r[" utilization.gpu [%]"].replace(" %", "")),
                     float(r[" power.draw [W]"].replace(" W", ""))))
print(f"samples: {len(rows)}, window: {rows[0][0]} -> {rows[-1][0]}")

# per-GPU stats
by_gpu = defaultdict(list)
by_gpu_pow = defaultdict(list)
for ts, i, u, p in rows:
    by_gpu[i].append(u)
    by_gpu_pow[i].append(p)

def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p / 100))]

print(f"\n{'GPU':>4} {'mean%':>6} {'p50%':>6} {'p95%':>6} {'W_mean':>7}")
for g in sorted(by_gpu):
    v = by_gpu[g]
    print(f"{g:>4} {sum(v)/len(v):>6.1f} {pct(v,50):>6.0f} {pct(v,95):>6.0f} {sum(by_gpu_pow[g])/len(by_gpu_pow[g]):>7.0f}")

# per-timestamp imbalance across GPUs
by_ts = defaultdict(dict)
for ts, i, u, p in rows:
    by_ts[ts][i] = u
spreads, cvs = [], []
for ts, m in sorted(by_ts.items()):
    v = list(m.values())
    if len(v) == 8:
        mean = sum(v) / 8
        spreads.append(max(v) - min(v))
        cvs.append((sum((x - mean) ** 2 for x in v) / 8) ** 0.5 / mean if mean > 0 else 0)
spreads.sort(); cvs.sort()
n = len(spreads)
print(f"\ncross-GPU spread (util pct-points, n={n} timestamps):")
print(f"  p50={spreads[n//2]:.0f}  p95={spreads[int(n*0.95)]:.0f}  max={spreads[-1]:.0f}")
print(f"  CV: p50={cvs[n//2]:.2f}  p95={cvs[int(n*0.95)]:.2f}  (CV=0.5 nghĩa là độ lệch chuẩn = 50% mean)")

# per-minute per-GPU mean (phase view)
per_min = defaultdict(lambda: defaultdict(list))
for ts, i, u, p in rows:
    per_min[ts[:16]][i].append(u)
print(f"\nper-minute mean util per GPU (chỉ in mỗi 3 phút):")
mins = sorted(per_min)
print("  minute        " + " ".join(f"g{g:d}>4.0f" if False else f"g{g}" for g in range(8)))
sel = [m for k, m in enumerate(mins) if k % 3 == 0]
for m in sel:
    vals = " ".join(f"{sum(per_min[m][g])/max(len(per_min[m][g]),1):4.0f}" for g in range(8))
    print(f"  {m}  {vals}")
