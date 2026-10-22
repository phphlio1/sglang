#!/usr/bin/env python3
"""NSys window summary across the 5 NSys cells (sqlite per cell).

Per cell: collection epoch, memcpy per direction, kernel-time totals,
DeepEP intranode trio share (spin-wait evidence), top non-DeepEP kernels,
per-device HtoD/DtoH spread. All numbers are within the nsys collection
window (+120s, warmup regime — see nsys_window.log anchors).
"""
import sqlite3

BASE = ("/shared/loipp/sglang/ph2lio/experiments/e2/results/glm53/"
        "agentx_dynamo/profiling/dfs-weight")
CELLS = [
    ("c64_20261021_141334", "engine_20261021_142055"),
    ("c64_20261021_152339", "engine_20261021_153200"),
    ("c64_20261021_171014", "engine_20261021_171735"),
    ("c64_20261021_182013", "engine_20261021_182835"),
    ("c64_20261021_193141", "engine_20261021_193952"),
]
KERN = "CUPTI_ACTIVITY_KIND_KERNEL"
MEM = "CUPTI_ACTIVITY_KIND_MEMCPY"

for cell, stamp in CELLS:
    db = sqlite3.connect(f"{BASE}/{cell}/profiling/nsys/{stamp}.sqlite")
    cur = db.cursor()
    enum = dict(cur.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER"))
    kind = {v: k for k, v in enum.items()}
    epoch, = cur.execute(
        "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()

    # kernel totals + DeepEP trio
    tot_n, tot_busy = cur.execute(
        f"SELECT COUNT(*), SUM(end-start) FROM {KERN}").fetchone()
    trio_busy = trio_n = 0
    trio = {}
    for nm, n, busy, avg, mx in cur.execute(f"""
        SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6,
               AVG(k.end-k.start)/1e6, MAX(k.end-k.start)/1e6
        FROM {KERN} k JOIN StringIds s ON k.demangledName=s.id
        WHERE s.value LIKE 'void deep_ep::legacy::intranode::%'
        GROUP BY s.value ORDER BY 3 DESC"""):
        short = nm.split("(")[0].replace("void deep_ep::legacy::intranode::", "")
        trio[short] = (n, busy, avg, mx)
        trio_busy += busy * 1e6
        trio_n += n
    others = cur.execute(f"""
        SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6
        FROM {KERN} k JOIN StringIds s ON k.demangledName=s.id
        WHERE s.value NOT LIKE 'void deep_ep::legacy::intranode::%'
        GROUP BY s.value ORDER BY 3 DESC LIMIT 3""").fetchall()

    # memcpy per direction
    mem = {}
    for k, n, gb, busy in cur.execute(
            f"SELECT copyKind, COUNT(*), SUM(bytes)/1e9, SUM(end-start)/1e9 "
            f"FROM {MEM} GROUP BY copyKind"):
        mem[enum.get(k, k)] = (n, gb, busy)

    # per-device spread for the two host-facing directions
    spread = {}
    for lab, kv in (("HtoD", kind.get("Host-to-Device")),
                    ("DtoH", kind.get("Device-to-Host"))):
        if kv is None:
            continue
        rows = cur.execute(
            f"SELECT deviceId, SUM(bytes)/1e6 FROM {MEM} WHERE copyKind={kv} "
            f"GROUP BY deviceId ORDER BY 2 DESC").fetchall()
        if rows:
            spread[lab] = (rows[0], rows[-1])

    span = cur.execute(f"SELECT MIN(start), MAX(end) FROM {MEM}").fetchone()
    print(f"== {cell} (session {stamp})")
    print(f"   epoch_ns={epoch}  kernel_n={tot_n}  kernel_busy={tot_busy/1e9:.1f}s")
    print(f"   DeepEP intranode trio: {100.0*trio_busy/tot_busy:.1f}% kernel busy"
          f"  n={trio_n}  ({', '.join(f'{k}: {v[1]/1e3:.0f}s avg{v[2]:.1f}ms max{v[3]:.0f}ms' for k,v in trio.items())})")
    for nm, n, busy in others:
        print(f"   other: {busy:7.0f} ms n={n:<6} {nm[:70]}")
    for lab, (n, gb, busy) in mem.items():
        print(f"   memcpy {lab}: n={n} bytes={gb:.2f} GB busy={busy:.2f}s")
    for lab, (hi, lo) in spread.items():
        print(f"   {lab} dev spread: dev{hi[0]}={hi[1]:.1f}MB .. dev{lo[0]}={lo[1]:.1f}MB")
    print(f"   window span {(span[1]-span[0])/1e9:.1f}s (+{span[0]/1e9:.0f}..+{span[1]/1e9:.0f}s)")
    db.close()
