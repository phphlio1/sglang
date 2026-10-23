#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nsys_readout.py — đọc + diễn giải NSys (.sqlite) các cell E2 (23/10).

Quy trình theo instruction "visualize and interpret NSys outputs":
 1) `nsys stats` TRƯỚC (subprocess, CWD=/tmp, mọi output ghi ra --out):
    report cuda_gpu_kern_sum / cuda_api_sum / cuda_gpu_mem_time_sum /
    cuda_gpu_mem_size_sum; nếu tên report không có trong bản nsys đang chạy
    thì fallback bộ default. Danh sách report thật lưu ở out/nsys_stats_help.txt.
 2) Tự tổng hợp từ sqlite, MỞ `?mode=ro` (không bao giờ ghi file gốc):
    - kernel theo tên      : CUPTI_ACTIVITY_KIND_KERNEL (+ StringIds)
    - CUDA API             : CUPTI_ACTIVITY_KIND_RUNTIME (+ StringIds)
    - sync                 : CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (+ ENUM_CUDA_SYNC_TYPE)
    - memcpy theo hướng    : CUPTI_ACTIVITY_KIND_MEMCPY (+ ENUM_CUDA_MEMCPY_OPER)
    - timeline 8 GPU       : hợp nhất interval kernel+memcpy+memset per deviceId
      -> busy thật (không chồng lấn), idle gap >= 1 ms, occupancy bin 100 ms
 3) Xuất PNG/CSV + index.html vào dir --out (riêng, không đụng dir gốc).

Chạy trên node tính toán qua srun (nsys chỉ có ở đó; login node 8 GB RAM),
dùng python của venv có matplotlib, vd:
  /shared/loipp/venv-aiperf/bin/python nsys_readout.py \
    --base <dfs-weight> --cells 141334,... --out <dfs-weight>/nsys_readout
"""
import argparse
import csv
import os
import sqlite3
import subprocess
import sys

REPORTS = ["cuda_gpu_kern_sum", "cuda_api_sum",
           "cuda_gpu_mem_time_sum", "cuda_gpu_mem_size_sum"]
GAP_NS = 1 * 1000 * 1000        # ngưỡng idle gap tính riêng: 1 ms
BIN_NS = 100 * 1000 * 1000      # bin timeline: 100 ms

CAP_TIMELINE = ("Mỗi hàng 1 GPU; nền trắng = idle, nền tối = bận. Dải trắng kéo dài "
    "giữa các bước forward là idle gap — dấu hiệu admission/scheduler giữ prefill "
    "(focus 1) hoặc rank chờ rank khác (MoE EP=8). Chi tiết gap: <code>&lt;cell&gt;_gpu_gaps.csv</code>.")
CAP_KERNEL = ("Top kernel theo TỔNG thời gian (cộng dồn mọi stream nên có thể chồng lấn). "
    "Nhóm communication/sync (vd deep_ep dispatch/notify) chiếm áp đảo ⇒ comm/sync-bound, "
    "không compute-bound (§7 đã thấy deep_ep 94% busy ở window warmup). "
    "GEMM/attention chiếm ⇒ compute-bound thật.")
CAP_API = ("CUDA API theo tổng thời gian. cudaMemcpyAsync/cudaLaunchKernel lớn đi kèm GPU "
    "idle nhiều ⇒ CPU/launch-bound. cudaStreamSynchronize/cudaEventSynchronize lớn ⇒ "
    "sync-bound (CPU chờ GPU).")
CAP_MEMCPY = ("Memcpy theo hướng. HtoD lớn = nạp lại từ host (chuỗi load-back host-tier); "
    "DtoD lớn = write_back/thu hồi chỗ trong device. LƯU Ý: window này thu lúc warmup "
    "(host tier còn 0) — KHÔNG kết luận load-back steady-state từ trace này (§7 đo HtoD ≈ 0).")


def nm_of(sids, v):
    """shortName/nameId trong sqlite là id int trỏ StringIds; một số bản lưu sẵn text."""
    if isinstance(v, int) and v in sids:
        return sids[v]
    return str(v)


def run_nsys_stats(sq, out, cell):
    help_path = os.path.join(out, "nsys_stats_help.txt")
    if not os.path.exists(help_path):
        with open(help_path, "w") as f:
            subprocess.run(["nsys", "stats", "--help"], stdout=f,
                           stderr=subprocess.STDOUT, cwd="/tmp")
    prefix = os.path.join(out, cell + "_")
    log = os.path.join(out, cell + "_nsys_stats.log")
    cmd = (["nsys", "stats", "--format", "csv", "--output", prefix]
           + [x for r in REPORTS for x in ("--report", r)] + [sq])
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd="/tmp").returncode
    made = [x for x in os.listdir(out)
            if x.startswith(cell + "_") and x.endswith(".csv")]
    if rc != 0 or not made:
        cmd = ["nsys", "stats", "--format", "csv", "--output", prefix, sq]
        with open(log, "a") as f:
            f.write("\n-- fallback default reports (rc=%d) --\n$ " % rc
                    + " ".join(cmd) + "\n")
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                cwd="/tmp").returncode
    return rc


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def analyze_cell(sq, out, cell):
    con = sqlite3.connect("file:%s?mode=ro" % sq, uri=True)
    sids = dict(con.execute("SELECT id, value FROM StringIds").fetchall())

    row = con.execute(
        "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    t0, t1 = row[0] or 0, row[1] or 0
    dur_s = (t1 - t0) / 1e9
    if dur_s <= 0:
        print("[cell %s] KHONG co kernel event — bo qua" % cell)
        return None
    anchor = con.execute(
        "SELECT utcEpochNs, systemClockNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()

    # ---- kernel theo tên
    kern = {}
    for sn, cnt, tot in con.execute(
            "SELECT shortName, COUNT(*), SUM(end-start) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY shortName"):
        e = kern.setdefault(nm_of(sids, sn), [0, 0])
        e[0] += cnt or 0
        e[1] += tot or 0
    kern_top = sorted(kern.items(), key=lambda kv: -kv[1][1])
    write_csv(os.path.join(out, cell + "_kernel_top.csv"),
              ["rank", "kernel", "count", "total_s", "pct_of_window"],
              [[i, k, v[0], round(v[1] / 1e9, 3),
                round(100.0 * v[1] / 1e9 / dur_s, 2)]
               for i, (k, v) in enumerate(kern_top)])

    # ---- CUDA API
    api = {}
    for nid, cnt, tot in con.execute(
            "SELECT nameId, COUNT(*), SUM(end-start) "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY nameId"):
        e = api.setdefault(nm_of(sids, nid), [0, 0])
        e[0] += cnt or 0
        e[1] += tot or 0
    api_top = sorted(api.items(), key=lambda kv: -kv[1][1])
    write_csv(os.path.join(out, cell + "_cuda_api.csv"),
              ["rank", "api", "count", "total_s"],
              [[i, k, v[0], round(v[1] / 1e9, 3)]
               for i, (k, v) in enumerate(api_top)])

    # ---- sync
    try:
        sync_enum = {i: (l or n) for i, n, l in con.execute(
            "SELECT id, name, label FROM ENUM_CUPTI_SYNC_TYPE")}
    except sqlite3.DatabaseError:
        sync_enum = {}
    sync = {}
    for st, cnt, tot in con.execute(
            "SELECT syncType, COUNT(*), SUM(end-start) "
            "FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION GROUP BY syncType"):
        sync["%s [%s]" % (sync_enum.get(st, "syncType?"), st)] = [cnt or 0, tot or 0]
    sync_top = sorted(sync.items(), key=lambda kv: -kv[1][1])
    write_csv(os.path.join(out, cell + "_sync.csv"),
              ["rank", "sync_type", "count", "total_s"],
              [[i, k, v[0], round(v[1] / 1e9, 3)]
               for i, (k, v) in enumerate(sync_top)])

    # ---- memcpy theo hướng
    try:
        mc_enum = {i: (l or n) for i, n, l in con.execute(
            "SELECT id, name, label FROM ENUM_CUDA_MEMCPY_OPER")}
    except sqlite3.DatabaseError:
        mc_enum = {}
    mc = {}
    for ck, cnt, by, tot in con.execute(
            "SELECT copyKind, COUNT(*), SUM(bytes), SUM(end-start) "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"):
        mc["%s [%s]" % (mc_enum.get(ck, "copyKind?"), ck)] = [cnt or 0, by or 0, tot or 0]
    write_csv(os.path.join(out, cell + "_memcpy.csv"),
              ["direction", "count", "total_bytes", "total_s"],
              [[k, v[0], v[1], round(v[2] / 1e9, 3)] for k, v in mc.items()])

    ms = con.execute("SELECT COUNT(*), SUM(bytes), SUM(end-start) "
                     "FROM CUPTI_ACTIVITY_KIND_MEMSET").fetchone()
    memset = (ms[0] or 0, ms[1] or 0, (ms[2] or 0) / 1e9)

    nvtx_n = con.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()[0]

    # ---- timeline per GPU: hợp nhất interval
    devs = [r[0] for r in con.execute(
        "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId")]
    tl = {}
    for dev in devs:
        iv = []
        for tbl in ("CUPTI_ACTIVITY_KIND_KERNEL",
                    "CUPTI_ACTIVITY_KIND_MEMCPY",
                    "CUPTI_ACTIVITY_KIND_MEMSET"):
            try:
                iv += [(a, b) for a, b in con.execute(
                    "SELECT start, end FROM %s WHERE deviceId=?" % tbl, (dev,))
                    if a is not None and b is not None]
            except sqlite3.DatabaseError:
                pass
        iv.sort()
        merged = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                if b > merged[-1][1]:
                    merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        busy = sum(b - a for a, b in merged)
        gaps = [(merged[i][1] - t0, merged[i + 1][0] - merged[i][1])
                for i in range(len(merged) - 1)
                if merged[i + 1][0] - merged[i][1] >= GAP_NS]
        # occupancy bin (tỷ lệ bận trong từng bin)
        nb = max(1, int(dur_s * 1e9 // BIN_NS))
        occ = [0.0] * nb
        for a, b in merged:
            i0, i1 = int((a - t0) // BIN_NS), int((b - t0) // BIN_NS)
            for i in range(max(0, i0), min(nb - 1, i1) + 1):
                s = max(a, t0 + i * BIN_NS)
                e = min(b, t0 + (i + 1) * BIN_NS)
                if e > s:
                    occ[i] += (e - s) / float(BIN_NS)
        tl[dev] = (merged, busy, gaps, [min(1.0, x) for x in occ])
    write_csv(os.path.join(out, cell + "_gpu_gaps.csv"),
              ["gpu", "busy_s", "busy_pct", "idle_s", "n_gaps_ge_1ms",
               "max_gap_s", "top5_gap_s"],
              [[d, round(tl[d][1] / 1e9, 3),
                round(100.0 * tl[d][1] / 1e9 / dur_s, 2),
                round(dur_s - tl[d][1] / 1e9, 3), len(tl[d][2]),
                round(max([g for _, g in tl[d][2]] or [0]) / 1e9, 3),
                ", ".join("%.3f" % (g / 1e9) for _, g in
                          sorted(tl[d][2], key=lambda x: -x[1])[:5])]
               for d in devs])

    # ---- summary text
    with open(os.path.join(out, cell + "_summary.txt"), "w") as f:
        f.write("cell %s\nwindow: %.3f s (kernel min..max)\n" % (cell, dur_s))
        if anchor:
            f.write("session anchor utcEpochNs=%s systemClockNs=%s "
                    "(tham khảo ghép wall-clock)\n" % (anchor[0], anchor[1]))
        f.write("kernel: %d tên, %d event, tổng %.3f s (cộng dồn, có thể chồng lấn)\n"
                % (len(kern), sum(v[0] for v in kern.values()),
                   sum(v[1] for v in kern.values()) / 1e9))
        f.write("memcpy: %s\nmemset: %d event, %.3f GB, %.3f s\nnvtx events: %d\n"
                % (mc, memset[0], memset[1] / 1e9, memset[2], nvtx_n))

    # ---- charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def barh_png(items, title, xlabel, path, n=15):
        top = items[:n][::-1]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(top) + 1.5)))
        ax.barh([k[:78] for k, _ in top], [v[1] / 1e9 for _, v in top])
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    barh_png(kern_top, "Cell %s — top kernel theo tên (window %.1f s)" % (cell, dur_s),
             "Tổng thời gian chạy (s) — cộng dồn mọi stream",
             os.path.join(out, cell + "_kernel_top.png"))
    barh_png(api_top, "Cell %s — CUDA API theo tổng thời gian" % cell,
             "Tổng thời gian (s)", os.path.join(out, cell + "_cuda_api.png"))

    M = np.array([tl[d][3] for d in devs]) if devs else np.zeros((1, 1))
    fig, ax = plt.subplots(figsize=(11, 0.5 * max(1, len(devs)) + 1.8))
    im = ax.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                   extent=[0, dur_s, len(devs) - 0.5, -0.5])
    ax.set_yticks(range(len(devs)))
    ax.set_yticklabels(["GPU %s" % d for d in devs])
    ax.set_xlabel("Thời gian trong window (s) — bin %d ms" % (BIN_NS // 1000000))
    ax.set_title("Cell %s — hoạt động GPU (trắng = idle)" % cell, fontsize=10)
    fig.colorbar(im, ax=ax, label="Tỷ lệ bận trong bin")
    fig.tight_layout()
    fig.savefig(os.path.join(out, cell + "_gpu_timeline.png"), dpi=120)
    plt.close(fig)

    if mc:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
        ks = list(mc.keys())
        a1.bar(ks, [v[1] / 1e9 for v in mc.values()])
        a1.set_ylabel("Tổng bytes (GB)"); a1.set_title("memcpy — bytes", fontsize=10)
        a2.bar(ks, [v[2] / 1e9 for v in mc.values()])
        a2.set_ylabel("Tổng thời gian (s)"); a2.set_title("memcpy — thời gian", fontsize=10)
        for ax_ in (a1, a2):
            ax_.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(os.path.join(out, cell + "_memcpy.png"), dpi=120)
        plt.close(fig)

    con.close()
    print("[cell %s] OK: window %.1f s, %d GPU, kernel %.3f s, memcpy %s"
          % (cell, dur_s, len(devs),
             sum(v[1] for v in kern.values()) / 1e9, mc))
    return {
        "cell": cell, "dur_s": round(dur_s, 2),
        "kernel_total_s": round(sum(v[1] for v in kern.values()) / 1e9, 2),
        "busy_pct_min": round(min(100.0 * tl[d][1] / 1e9 / dur_s for d in devs), 1),
        "busy_pct_max": round(max(100.0 * tl[d][1] / 1e9 / dur_s for d in devs), 1),
        "api_top1": api_top[0][0] if api_top else "",
        "api_top1_s": round(api_top[0][1][1] / 1e9, 3) if api_top else 0,
        "kern_top1": kern_top[0][0][:60] if kern_top else "",
        "memcpy": "; ".join("%s=%.3f GB" % (k, v[1] / 1e9) for k, v in mc.items()),
        "nvtx": nvtx_n, "memset_GB": round(memset[1] / 1e9, 3),
    }


def html_table(rows, header):
    h = "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
    h += "<tr>" + "".join("<th>%s</th>" % c for c in header) + "</tr>"
    for r in rows:
        h += "<tr>" + "".join("<td>%s</td>" % c for c in r) + "</tr>"
    return h + "</table>"


def write_html(out, cells, cross):
    parts = ["<html><head><meta charset='utf-8'><title>NSys readout — 5 cell E2</title></head><body>"]
    parts.append("<h1>NSys readout — 5 cell NSys E2 (window thu +120 s, warmup)</h1>")
    parts.append("<p>Cách đọc bottleneck cho SGLang inference: "
                 "<b>compute-bound</b> = GEMM/attention chiếm áp đảo &amp; GPU bận liên tục; "
                 "<b>comm/sync-bound</b> = kernel communication (deep_ep dispatch/notify) chiếm áp đảo "
                 "(rank chờ rank khác, MoE EP=8); <b>CPU/launch-bound</b> = CUDA API launch lớn + GPU idle; "
                 "<b>sync-bound</b> = cuda*Sync API lớn; <b>memory-bound</b> = memcpy lớn.</p>")
    parts.append("<p><b>Hạn chế của trace này (nói rõ để không kết luận sai)</b>: window thu nằm ở "
                 "warmup — host tier còn 0, chuỗi load-back host-tier chưa diễn ra; đo load-back "
                 "steady-state bằng metrics (bottleneck doc §7), không bằng trace này.</p>")
    parts.append("<h2>Tổng hợp 5 cell</h2>")
    if cross:
        parts.append(html_table(
            [[c[k] for k in ("cell", "dur_s", "kernel_total_s", "busy_pct_min",
                             "busy_pct_max", "kern_top1", "api_top1", "api_top1_s",
                             "memcpy", "memset_GB", "nvtx")] for c in cross],
            ["cell", "window(s)", "kernel tổng(s)", "busy% min", "busy% max",
             "kernel top-1", "API top-1", "API top-1 (s)", "memcpy", "memset GB", "nvtx"]))
        parts.append("<p>Bảng đầy đủ: <code>summary_cross.csv</code>; nsys stats CSV từng cell: "
                     "<code>&lt;cell&gt;_*.csv</code> (trong cùng dir).</p>")
    for c in cells:
        parts.append("<hr><h2>Cell %s</h2>" % c)
        for fn, cap in ((c + "_gpu_timeline.png", CAP_TIMELINE),
                        (c + "_kernel_top.png", CAP_KERNEL),
                        (c + "_cuda_api.png", CAP_API),
                        (c + "_memcpy.png", CAP_MEMCPY)):
            p = os.path.join(out, fn)
            if os.path.exists(p):
                parts.append("<h3>%s</h3><img src='%s' style='max-width:100%%'><p>%s</p>"
                             % (fn, fn, cap))
    parts.append("</body></html>")
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dir chứa c64_20261021_*")
    ap.add_argument("--cells", required=True, help="danh sách stamp cách nhau dấu phẩy")
    ap.add_argument("--out", required=True, help="dir output (tạo mới, không đụng file gốc)")
    ap.add_argument("--skip-nsys-stats", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    cross = []
    for cell in cells:
        run_dir = os.path.join(a.base, "c64_20261021_" + cell)
        nsdir = os.path.join(run_dir, "profiling", "nsys")
        sqs = sorted(x for x in os.listdir(nsdir) if x.endswith(".sqlite"))
        if not sqs:
            print("[cell %s] KHONG tim thay sqlite trong %s — bo qua" % (cell, nsdir))
            continue
        sq = os.path.join(nsdir, sqs[0])
        if not a.skip_nsys_stats:
            try:
                rc = run_nsys_stats(sq, a.out, cell)
                print("[cell %s] nsys stats rc=%d" % (cell, rc))
            except Exception as e:
                print("[cell %s] nsys stats LOI: %s (van chay phan python)" % (cell, e))
        try:
            r = analyze_cell(sq, a.out, cell)
            if r:
                cross.append(r)
        except Exception as e:
            print("[cell %s] LOI phan tich: %s" % (cell, e))
    if cross:
        write_csv(os.path.join(a.out, "summary_cross.csv"), list(cross[0].keys()),
                  [[c[k] for k in c] for c in cross])
        write_html(a.out, cells, cross)
        print("DONE: %d/%d cell — xem %s" % (len(cross), len(cells),
                                             os.path.join(a.out, "index.html")))
    else:
        print("FAIL: khong cell nao thanh cong")
        sys.exit(1)


if __name__ == "__main__":
    main()
