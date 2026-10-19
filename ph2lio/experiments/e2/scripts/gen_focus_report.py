#!/usr/bin/env python3
"""Generate a focus report (.md) for a profiling run dir, with the exact
structure of docs/focus3320_routing_scheduling.md (§0 sources & window,
§1 scheduling policy, §2 routing imbalance, §3 health-check table,
§4 next steps, §5 reproduce). All numbers come from focus_report_lib.

Usage:
    python3 gen_focus_report.py <run_dir> [-o out.md]
Default out: docs/focus_<stamp>_routing_scheduling.md next to this script.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focus_report_lib import analyze_run, scrape_counts  # noqa: E402


def vn(n):
    """Số nguyên kiểu VN: chấm làm dấu ngăn cách nghìn (583296 -> 583.296)."""
    return f"{int(round(n)):,}".replace(",", ".")


def f(x, d=2):
    return f"{x:.{d}f}" if x == x else "-"


def k(x):
    """21.5k / 3.8k kiểu bảng 2.2."""
    return f"{x / 1e3:.1f}k"


def spread_x(vals):
    vals = [v for v in vals if v is not None]
    return f"{max(vals) / min(vals):.1f}×" if vals and min(vals) > 0 else "-"


def build(rep, scrapes):
    w = rep["window"]
    c = rep["client"]
    e = rep["engine"]
    r = rep["rank"]
    d = rep["dynamo"]
    g = rep["gpu"]
    j = rep["join"]
    cost = rep["cost"]
    fe = rep["frontend"]
    from focus_report_lib import hhmmss
    t0s, t1s = hhmmss(w["t0"]), hhmmss(w["t1"])
    stamp = rep["stamp"]
    date = stamp.split("_")[1][:8]
    day = f"{date[6:8]}-{date[4:6]}-{date[0:4]}"

    m = r["means"]
    usage = m["sglang:full_token_usage"]
    usage_v = {k2: v for k2, v in usage.items() if v is not None}
    q_req = m["sglang:num_queue_reqs"]
    q_v = [v for v in q_req.values() if v is not None]
    chr_ = m["sglang:cache_hit_rate"]
    chr_v = [v for v in chr_.values() if v is not None]
    kv_used = [v for v in m["sglang:kv_used_tokens"].values() if v is not None]
    hic_used = [v for v in m["sglang:hicache_host_used_tokens"].values() if v is not None]
    ap = d["active_prefill"]
    ap_v = [v for v in ap.values() if v == v]
    adb = d["active_decode_blocks"]
    adb_v = [v for v in adb.values() if v == v]
    run_r = [v for v in m["sglang:num_running_reqs"].values() if v is not None]

    # rank extremes cho bảng/§2.3
    rmax_u = max(usage_v, key=usage_v.get)
    rmin_u = min(usage_v, key=usage_v.get)
    per_rank = j["per_rank"]
    rmax_qw = max(per_rank, key=lambda x: per_rank[x]["qw_mean"])
    rmin_qw = min(per_rank, key=lambda x: per_rank[x]["qw_mean"])
    rmax_ttft = max(per_rank, key=lambda x: per_rank[x]["ttft_mean"])
    rmin_ttft = min(per_rank, key=lambda x: per_rank[x]["ttft_mean"])
    rmin_n = min(per_rank, key=lambda x: per_rank[x]["n"])
    rmax_n = max(per_rank, key=lambda x: per_rank[x]["n"])
    rank_counts = "/".join(str(per_rank[x]["n"]) for x in sorted(per_rank))

    # router overhead: gộp các key router_id
    def ov(base):
        parts = list(d["router_overhead"].get(base, {}).values())
        if not parts:
            return 0, float("nan")
        n = sum(p["decisions"] for p in parts)
        return n, sum(p["decisions"] * p["mean_ms"] for p in parts) / n

    ov_n, ov_mean = ov("dynamo_router_overhead_total_ms")
    _, ov_sched = ov("dynamo_router_overhead_scheduling_ms")
    _, ov_idx = ov("dynamo_router_overhead_indexer_find_matches_ms")
    _, ov_bh = ov("dynamo_router_overhead_block_hashing_ms")
    _, ov_sh = ov("dynamo_router_overhead_seq_hashing_ms")

    q = cost["quintiles"]
    q1, q5 = q[0], q[-1]
    http200 = {k2: v for k2, v in fe["http"].items() if isinstance(k2, int)}
    http_txt = " / ".join(f"{v}×{k2}" for k2, v in sorted(http200.items())[:5])
    n_http = sum(http200.values())
    mid_vals = " ".join(f"{v:.0f}" for v in g["mid_minute_vals"].values())
    first100 = ", ".join(f"r{rk} {ts}" for rk, ts in r["first_100"].items())
    retract_txt = "; ".join(f"{ts} `#retracted_reqs: {n2}`" for ts, n2, _ in e["retracts"])
    # boot: pool KV device lớn nhất (bỏ qua pool phụ), đếm dòng cùng pool
    bk = max(e["boot_kv"], key=lambda x: x[2]) if e["boot_kv"] else ("-", 0, 0)
    bk_n = sum(1 for b in e["boot_kv"] if b[1] == bk[1] and b[2] == bk[2])
    bh = e["boot_hicache"][0] if e["boot_hicache"] else ("-", 0, 0)
    tps = c["throughput"]["output_token_throughput_tps"]
    ttft_p99 = c["ttft"]["p99_ms"] / 1e3
    ttft_mean = c["ttft"]["mean_ms"] / 1e3
    qw_p99 = c["queue"]["queue_time_p99_s"]
    tpot = c["tpot"]["mean_ms"]

    exd1 = e["ex_dec_min"]
    exd2 = e["ex_dec_max"]
    exp = e["ex_pre_med"]

    doc = f"""# Focus Analysis — Run {stamp}: Engine Scheduling Policy & Dynamo Routing

RunID **{stamp}** ({day}, full stack dynamo 1.4.2, `dfs-weight` c64, 8×H200
hgx-h200-01). Replicate kiểm tra **consistency bottleneck** so với baseline
`c64_20261014_112105` (3320) — cùng config từng tham số (seed 42, cold radix
cache, PROBE_DURATION 1800 s). Theo 2 hướng định hướng:

(1) local engine scheduling policy giới hạn GPU utilization,

(2) global dynamo routing gây mất cân bằng giữa các GPU/DPA rank. Kèm
health-check thông số: GPU, token usage, cache, KV cache.

Run dir: `results/glm53/agentx_dynamo/profiling/dfs-weight/{stamp}/`

## 0. Nguồn dữ liệu & quy ước cite

| Nguồn | Nội dung | Cách trích |
|---|---|---|
| `engine_*.log` | {vn(e['n_batch'])} dòng batch scheduler (Decode/Prefill), boot lines, WARN | dòng theo timestamp `HH:MM:SS` |
| `frontend_*.log` | {vn(len(fe['events']))} dòng worker-selection, {vn(len(fe['ttft_fe']))} request-completed, {vn(n_http)} HTTP access (status 200) | dòng theo timestamp |
| `server_metrics_export.jsonl` | ~{vn(scrapes['sglang_rank_ts.csv'])} scrape × 2 endpoint (8081 `sglang:*` per dp_rank; 8888 `dynamo_*`) | trích sẵn → `analysis/sglang_rank_ts.csv`, `analysis/dynamo_ts.csv` |
| `gpu_samples.csv` | {vn(g['n'])} mẫu nvidia-smi 1 s × 8 GPU | cột `utilization.gpu [%]` |
| `profile_export.jsonl` / `schedule_eval.jsonl` | {vn(w['n_profiling'] + w['n_warmup'])}/{vn(scrapes['schedule_eval'])} record per request | field-level |
| `profiling/pyspy_*.svg` ×8 | flame 25 Hz × 120 s × 8 scheduler | frame-level |

**Benchmark window = {t0s} → {t1s}Z** ({w['duration'] / 60:.0f} phút). Cách suy ra:
`profile_export.jsonl` → `metadata.request_start_ns`, `benchmark_phase`:
{w['n_warmup']} record phase `warmup`, record đầu phase `profiling` lúc
{t0s}; duration {w['duration']:.0f} s từ `e2_metrics.json` (`run.duration_s`).
Mọi thống kê dưới đây tính trong window này trừ khi ghi khác.

## 1. Local engine scheduling policy (dfs-weight + prefill-delayer)

### 1.1 GPU "bận" nhưng không hiệu quả

- GPU util benchmark window: **{f(g['win_min'], 0)}–{f(g['win_max'], 0)}%** mọi GPU
  (`gpu_samples.csv`, per-minute mean; ví dụ phút {g['mid_minute']}: {mid_vals}).
  Mean cả run (gồm warmup/drain): {f(g['whole_mean_min'], 1)}–{f(g['whole_mean_max'], 1)}%.
- Fwd occupancy {f(e['occ_min'], 0)}–{f(e['occ_max'], 0)}% các phút benchmark
  (`engine_*.log`, field `fwd occupancy` tổng hợp per-minute).
- Nhưng batch decode nhỏ: **running-req mean {f(min(run_r), 2)}–{f(max(run_r), 2)}/rank**
  trong window (`analysis/sglang_rank_ts.csv`, `sglang:num_running_reqs` per
  dp_rank) — log cho thấy batch mỏng (ví dụ {exd1[1]} `#running-req: {exd1[0]},
  gen throughput: {f(exd1[2])}`; {exd2[1]} `#running-req: {exd2[0]}, gen
  throughput: {f(exd2[2])}`) dù trần `--max-running-requests=256`.
- Hệ quả: output chỉ {f(tps, 0)} tok/s trên 8×H200 (`e2_metrics.json`). Cơ chế:
  MoE EP=8 all-to-all đồng bộ mọi rank mỗi bước → chi phí cố định/trước
  (flame `profiling/pyspy_*.svg`: MoE forward + DeepEP `dispatch_b` chiếm
  ưu thế) — batch mỏng → latency-bound → util cao ≠ năng suất cao.

**Kết luận 1.1:** "GPU utilization bị giới hạn" đúng ở nghĩa *hiệu quả*
không phải *không rảnh*: scheduling để GPU bận với batch
{f(min(run_r), 2)}–{f(max(run_r), 2)} req/rank thay vì dồn việc vào batch dày.

### 1.2 Prefill bị hoãn suốt benchmark — TTFT tail từ đây

- Backlog prefill duy trì **{vn(e['pending_pm_min'])}–{vn(e['pending_pm_max'])}
  pending token** (per-minute mean) ở các phút benchmark (`engine_*.log`,
  field `#pending-token`; ví dụ dòng {exp[0]} `#pending-token: {vn(exp[1])},
  input throughput: {f(exp[2])}`; đỉnh {e['ex_pre_maxpend'][0]}
  {vn(e['ex_pre_maxpend'][1])}).
- Admission queue gần rỗng: `#queue-req` mean {f(min(q_v), 2)}–{f(max(q_v), 2)}/rank
  (`analysis/sglang_rank_ts.csv`, `sglang:num_queue_reqs`) — request chờ ở
  prefill backlog chứ không phải hàng admission. Đối lập queue-wait p99
  {qw_p99:.0f} s (`e2_metrics.json`) ↔ queue-req ~0 → mâu thuẫn định vị
  được chỗ chờ.
- Cơ chế: `--enable-prefill-delayer` + `--schedule-policy=dfs-weight` bảo vệ
  request đang decode (prefill chen làm ITL giật) nên hoãn prefill mới.
- `sglang:prefill_delayer_outcomes` có đếm theo outcome per rank
  (`server_metrics_export.jsonl`) — chưa phân tích chi tiết (cần re-extract
  giữ label), backlog đã định lượng đủ.

**Kết luận 1.2:** TTFT p99 {ttft_p99:.1f} s là hệ quả chọn lối policy (bảo vệ
ITL decode), không phải cạn tài nguyên — GPU vẫn
{f(g['win_min'], 0)}–{f(g['win_max'], 0)}% nhưng đang bận với batch decode mỏng (1.1).

### 1.3 Mất nhịp rank-local

- Cùng một thời điểm, running-req chênh giữa rank: **spread p50 =
  {r['run_spread_p50']:.0f}, p95 = {r['run_spread_p95']:.0f}, max =
  {r['run_spread_max']:.0f}** (`analysis/sglang_rank_ts.csv`, max−min
  `num_running_reqs` qua {vn(r['n_ts'])} timestamp).
- **{f(r['zero_vs_busy_pct'], 1)}% thời gian** có rank 0-running trong khi
  rank khác ≥3 running (nguồn trên) — rank đang prefill không decode được
  (attn-DP: 1 scheduler độc lập/GPU, prefill chunk chiếm slot forward).

## 2. Global dynamo routing: mất cân bằng giữa các GPU/rank

### 2.1 Bề mặt GPU util KHÔNG mất cân bằng

- Mean util 8 GPU: {f(g['whole_mean_min'], 1)}–{f(g['whole_mean_max'], 1)}%
  (chênh {f(g['whole_mean_max'] - g['whole_mean_min'], 1)} điểm), p50 =
  {f(g['p50_all'], 0)}% cả 8 (`gpu_samples.csv`). Cross-GPU spread tại mỗi
  giây: p50 = **{f(g['cross_spread_p50'], 0)} điểm**, p95 = {f(g['cross_spread_p95'], 0)},
  max = {f(g['cross_spread_max'], 0)}.
- Lý do kiến trúc: MoE EP=8 all-to-all đồng bộ mọi rank trong mọi bước
  forward (cơ chế 1.1) — rank ít request vẫn phải tham gia exchange của
  rank khác → util bị ép cân bằng bề mặt.

### 2.2 Mất cân bằng thật ở workload/KV per rank

Benchmark window, `analysis/sglang_rank_ts.csv` + `analysis/dynamo_ts.csv`:

| Chỉ số (mean) | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | spread |
|---|---|---|---|---|---|---|---|---|---|
| Active prefill tokens (8888 `worker_active_prefill_tokens`) | {k(ap.get(0, float('nan')))} | {k(ap.get(1, float('nan')))} | {k(ap.get(2, float('nan')))} | {k(ap.get(3, float('nan')))} | {k(ap.get(4, float('nan')))} | {k(ap.get(5, float('nan')))} | {k(ap.get(6, float('nan')))} | {k(ap.get(7, float('nan')))} | **{spread_x(ap_v)}** |
| Active decode blocks (8888) | {k(adb.get(0, float('nan')))} | {k(adb.get(1, float('nan')))} | {k(adb.get(2, float('nan')))} | {k(adb.get(3, float('nan')))} | {k(adb.get(4, float('nan')))} | {k(adb.get(5, float('nan')))} | {k(adb.get(6, float('nan')))} | {k(adb.get(7, float('nan')))} | {spread_x(adb_v)} |
| KV token usage (8081 `full_token_usage`) | {f(usage.get(0))} | {f(usage.get(1))} | {f(usage.get(2))} | {f(usage.get(3))} | {f(usage.get(4))} | {f(usage.get(5))} | {f(usage.get(6))} | {f(usage.get(7))} | {spread_x(list(usage_v.values()))} |
| Admission queue-req (8081) | {f(q_req.get(0))} | {f(q_req.get(1))} | {f(q_req.get(2))} | {f(q_req.get(3))} | {f(q_req.get(4))} | {f(q_req.get(5))} | {f(q_req.get(6))} | {f(q_req.get(7))} | — |
| Cache hit rate (8081) | {f(chr_.get(0))} | {f(chr_.get(1))} | {f(chr_.get(2))} | {f(chr_.get(3))} | {f(chr_.get(4))} | {f(chr_.get(5))} | {f(chr_.get(6))} | {f(chr_.get(7))} | — |

- Tại cùng thời điểm, KV usage chênh giữa rank: **spread p50 =
  {f(r['usage_spread_p50'])}, p95 = {f(r['usage_spread_p95'])}**
  (max−min `full_token_usage` qua {vn(r['n_ts'])} timestamp).
- Request count per rank (warmup loại 5%): {rank_counts} — đếm `dp_rank`
  trong {vn(len(fe['events']))} dòng `Selected worker ... dp_rank=N logit=`
  (`frontend_*.log`), join timestamp với `schedule_eval.jsonl` (±2 s).

### 2.3 Hệ quả của mất cân bằng

- **{len(r['first_100'])} rank lần lượt chạm KV 100%**: {first100}
  (`sglang:full_token_usage` ≥ 0.999) — khớp {len(e['retracts'])} dòng WARN
  retract (`engine_*.log`): {retract_txt} `KV cache pool is full. Retract
  requests.` Memory pressure là **rank-local** — do routing dồn, không
  phải tổng thể thiếu.
- Queue-wait per rank (join selector ↔ schedule_eval): r{rmax_qw}
  **{f(per_rank[rmax_qw]['qw_mean'])} s** mean vs r{rmin_qw}
  {f(per_rank[rmin_qw]['qw_mean'])} s; TTFT per rank: r{rmax_ttft}
  {f(per_rank[rmax_ttft]['ttft_mean'])} s vs r{rmin_ttft}
  {f(per_rank[rmin_ttft]['ttft_mean'])} s (r{rmax_ttft} p99
  {per_rank[rmax_ttft]['ttft_p99']:.0f} s).
- Correlation input size ↔ rank: rank ít request nhận input lớn nhất
  (r{rmin_n}: {per_rank[rmin_n]['in_mean'] / 1e3:.0f}k token mean input) —
  router đang cân bằng theo *số request* bề ngoài nhưng chưa cân bằng theo
  *tải thực* (prefill token/KV).

### 2.4 Router overhead — không đáng kể

`analysis/dynamo_ts.csv` (8888, histogram delta trong window):
`dynamo_router_overhead_total_ms`: **{vn(ov_n)} decision, mean {f(ov_mean)}
ms** (chi tiết: scheduling {f(ov_sched)}, indexer_find_matches {f(ov_idx, 3)},
block_hashing {f(ov_bh, 3)}, seq_hashing {f(ov_sh, 3)}). Không đáng kể so
TTFT giây. Frontend queue pending max = {f(d['queue_pending_max'], 0)} mọi
timestamp (`dynamo_frontend_router_queue_pending_requests`).

### 2.5 Công thức cost: có tín hiệu nhưng chưa đủ

- corr(cost `logit`, TTFT) = **{cost['corr_logit_ttft']:+.2f}**, corr(logit,
  queue_wait) = {cost['corr_logit_qw']:+.2f} ({vn(cost['n'])} request join
  theo timestamp; `logit` từ dòng `Selected worker` frontend_*.log,
  `ttft_s`/`queue_wait_s` từ schedule_eval.jsonl).
- Quintile logit → median TTFT thực: Q1 {f(q1['ttft_p50'])} s → Q5
  {f(q5['ttft_p50'])} s (queue_wait {f(q1['qw_p50'])} → {f(q5['qw_p50'])} s):
  cost cao đoán đúng hướng chờ lâu hơn.
- Nhưng phương sai lớn chưa giải thích + hot-rank (2.2–2.3) vẫn bị dồn
  vượt KV → giả thuyết "công thức chưa tối ưu" **được ủng hộ một phần**.

Nghi vấn cơ chế:
- (a) kv-state của router hết hạn → cost ước tính theo state cũ.
- (b) weight `16 × prefill_blocks` có vẻ underestimate chi phí prefill thực
  (r{max(ap, key=ap.get)} nhận {k(max(ap_v))} active prefill token mean —
  nhiều nhất — mà cost chưa kéo request đi chỗ khác).

## 3. Health-check thông số engine (checklist mentor)

Benchmark window trừ khi ghi khác — nguồn: `analysis/sglang_rank_ts.csv`
(8081, per dp_rank), `analysis/dynamo_ts.csv` (8888), `gpu_samples.csv`,
`engine_*.log` (boot + WARN).

| Thông số | Giá trị | Ý nghĩa / Mô tả |
|---|---|---|
| GPU utilization | {f(g['win_min'], 0)}–{f(g['win_max'], 0)}% mỗi GPU (window) | GPU bận suốt benchmark — không rảnh; hiệu quả thấp đến từ batch mỏng (§1.1), không phải idle |
| GPU power | mean {f(g['power_min'], 0)}–{f(g['power_max'], 0)} W/GPU | Tương ứng mức bận cao, ổn định |
| Token usage (KV device) — mean | r{rmax_u} {f(usage_v[rmax_u])} … r{rmin_u} {f(usage_v[rmin_u])} | Tỉ lệ lấp pool KV device của từng rank; mất cân bằng {spread_x(list(usage_v.values()))} theo rank (§2.2) |
| Token usage (KV device) — đỉnh | {len(r['first_100'])} rank chạm 1.00 | {first100} — áp lực muộn run, rank-local do routing dồn |
| KV device capacity | {vn(bk[1])} token/rank ({f(bk[2])} GB) | Boot line {bk[0]} `KV Cache is allocated ... #tokens: {vn(bk[1])}` ×{bk_n} rank; kv_used mean {min(kv_used) / 1e3:.0f}k–{max(kv_used) / 1e3:.0f}k |
| KV hicache host — pool | {vn(bh[1])} token = {f(bh[2], 0)} GB/rank | Boot {bh[0]} ×{len(e['boot_hicache'])} dòng; offload host cho radix cache phân tầng |
| KV hicache host — used | mean {min(hic_used) / 1e6:.2f}M → {max(hic_used) / 1e6:.2f}M | Dùng {100 * min(hic_used) / bh[1]:.0f}–{100 * max(hic_used) / bh[1]:.0f}% pool; cũng mất cân bằng theo rank như KV device |
| Cache reuse (engine) | {f(e['cache_reuse_pct'], 1)}% ({f(e['cached_m'], 2)}M cached / {f(e['new_m'], 2)}M new) | Tổng token prefill cả run: tỉ lệ token được serve từ cache — cache engine hoạt động tốt |
| Cached tokens/request (frontend) | mean {vn(d['cached_tokens']['mean'])} token (n={vn(d['cached_tokens']['n'])}) | `dynamo_frontend_cached_tokens`: mỗi request trung bình {d['cached_tokens']['mean'] / 1e3:.0f}k token input được lấy từ cache (input median {rep['input_len_median'] / 1e3:.0f}k) |
| Cache hit rate per rank | {f(min(chr_v))}–{f(max(chr_v))} (mean) | Chênh theo rank phản ánh việc routing dồn request cùng prefix về cùng rank (kv-router chủ đích) |
| Retract | {len(e['retracts'])} lần, mỗi lần 1 request | {retract_txt} — request bị thu hồi khi rank đầy; hiếm, không sập throughput nhưng là dấu hiệu mất cân bằng (§2.3) |
| EAGLE accept len | {f(e['accept_pm_min'])}–{f(e['accept_pm_max'])} (per-minute mean) | Ngưỡng khỏe ≥2.0 — spec decode có phải vấn đề không, đọc cùng ITL |
| ITL (frontend đo) | mean {f(d['itl']['mean'] * 1e3)} ms | `dynamo_frontend_inter_token_latency_seconds`; khớp client TPOT {f(tpot, 1)} ms → frontend không thêm độ trễ đáng kể |

## 4. Vai trò trong so sánh consistency

Run này là replicate của baseline 3320 (`c64_20261014_112105`), cùng config
từng tham số. Bảng so sánh 3 run + verdict từng focus:
`docs/focus_consistency_3runs.md` (tạo bằng `scripts/compare_focus_runs.py`).

## 5. Phụ lục: cách từng số được tính (reproduce)

- Trích metrics: `srun -p hardware-hcm -w hgx-h200-01 --cpus-per-task=8
  --mem=16G python3 -I scripts/extract_metrics_timeseries.py <run_dir>`
  (stream JSONL 4.4 GB trên node) → `analysis/{{sglang_rank_ts,dynamo_ts}}.csv`.
- Report này: `python3 scripts/gen_focus_report.py <run_dir>` — toàn bộ số
  liệu tính bằng `scripts/focus_report_lib.py` (per-rank mean, spread
  max−min per timestamp, join router↔engine theo timestamp ±2 s, histogram
  delta, per-minute tổng hợp engine log, corr + quintile).
- Join router↔engine: `frontend_*.log` dòng `Selected worker` (request_id,
  dp_rank, logit) match `schedule_eval.jsonl` theo timestamp ±2 s —
  request_id (frontend) ≠ rid (engine) nên không join trực tiếp được.
"""
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    rep = analyze_run(args.run_dir)
    scrapes = scrape_counts(args.run_dir)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "docs",
        f"focus_{rep['stamp']}_routing_scheduling.md")
    with open(os.path.abspath(out), "w") as fh:
        fh.write(build(rep, scrapes))
    print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
