# Focus Analysis — Run c64_20261015_224930: Engine Scheduling Policy & Dynamo Routing

RunID **c64_20261015_224930** (15-10-2026, full stack dynamo 1.4.2, `dfs-weight` c64, 8×H200
hgx-h200-01). Replicate kiểm tra **consistency bottleneck** so với baseline
`c64_20261014_112105` (3320) — cùng config từng tham số (seed 42, cold radix
cache, PROBE_DURATION 1800 s). Theo 2 hướng định hướng:

(1) local engine scheduling policy giới hạn GPU utilization,

(2) global dynamo routing gây mất cân bằng giữa các GPU/DPA rank. Kèm
health-check thông số: GPU, token usage, cache, KV cache.

Run dir: `results/glm53/agentx_pilot_preprod/agentx_dynamo/profiling/dfs-weight/c64_20261015_224930/`

## 0. Nguồn dữ liệu & quy ước cite

| Nguồn | Nội dung | Cách trích |
|---|---|---|
| `engine_*.log` | 9.490 dòng batch scheduler (Decode/Prefill), boot lines, WARN | dòng theo timestamp `HH:MM:SS` |
| `frontend_*.log` | 1.255 dòng worker-selection, 1.252 request-completed, 10.392 HTTP access (status 200) | dòng theo timestamp |
| `server_metrics_export.jsonl` | ~8.931 scrape × 2 endpoint (8081 `sglang:*` per dp_rank; 8888 `dynamo_*`) | trích sẵn → `analysis/sglang_rank_ts.csv`, `analysis/dynamo_ts.csv` |
| `gpu_samples.csv` | 30.424 mẫu nvidia-smi 1 s × 8 GPU | cột `utilization.gpu [%]` |
| `profile_export.jsonl` / `schedule_eval.jsonl` | 1.252/6.333 record per request | field-level |
| `profiling/pyspy_*.svg` ×8 | flame 25 Hz × 120 s × 8 scheduler | frame-level |

**Benchmark window = 23:21:41 → 23:51:41Z** (30 phút). Cách suy ra:
`profile_export.jsonl` → `metadata.request_start_ns`, `benchmark_phase`:
65 record phase `warmup`, record đầu phase `profiling` lúc
23:21:41; duration 1800 s từ `e2_metrics.json` (`run.duration_s`).
Mọi thống kê dưới đây tính trong window này trừ khi ghi khác.

## 1. Local engine scheduling policy (dfs-weight + prefill-delayer)

### 1.1 GPU "bận" nhưng không hiệu quả

- GPU util benchmark window: **77–97%** mọi GPU
  (`gpu_samples.csv`, per-minute mean; ví dụ phút 23:36: 97 91 91 89 89 91 91 91).
  Mean cả run (gồm warmup/drain): 70.9–73.3%.
- Fwd occupancy 83–96% các phút benchmark
  (`engine_*.log`, field `fwd occupancy` tổng hợp per-minute).
- Nhưng batch decode nhỏ: **running-req mean 2.45–3.66/rank**
  trong window (`analysis/sglang_rank_ts.csv`, `sglang:num_running_reqs` per
  dp_rank) — log cho thấy batch mỏng (ví dụ 23:44:36 `#running-req: 1,
  gen throughput: 48.30`; 23:48:56 `#running-req: 11, gen
  throughput: 516.49`) dù trần `--max-running-requests=256`.
- Hệ quả: output chỉ 528 tok/s trên 8×H200 (`e2_metrics.json`). Cơ chế:
  MoE EP=8 all-to-all đồng bộ mọi rank mỗi bước → chi phí cố định/trước
  (flame `profiling/pyspy_*.svg`: MoE forward + DeepEP `dispatch_b` chiếm
  ưu thế) — batch mỏng → latency-bound → util cao ≠ năng suất cao.

**Kết luận 1.1:** "GPU utilization bị giới hạn" đúng ở nghĩa *hiệu quả*
không phải *không rảnh*: scheduling để GPU bận với batch
2.45–3.66 req/rank thay vì dồn việc vào batch dày.

### 1.2 Prefill bị hoãn suốt benchmark — TTFT tail từ đây

- Backlog prefill duy trì **11.324–83.241
  pending token** (per-minute mean) ở các phút benchmark (`engine_*.log`,
  field `#pending-token`; ví dụ dòng 23:46:20 `#pending-token: 4.537,
  input throughput: 1650.92`; đỉnh 23:45:59
  329.927).
- Admission queue gần rỗng: `#queue-req` mean 0.14–0.71/rank
  (`analysis/sglang_rank_ts.csv`, `sglang:num_queue_reqs`) — request chờ ở
  prefill backlog chứ không phải hàng admission. Đối lập queue-wait p99
  22 s (`e2_metrics.json`) ↔ queue-req ~0 → mâu thuẫn định vị
  được chỗ chờ.
- Cơ chế: `--enable-prefill-delayer` + `--schedule-policy=dfs-weight` bảo vệ
  request đang decode (prefill chen làm ITL giật) nên hoãn prefill mới.
- `sglang:prefill_delayer_outcomes` có đếm theo outcome per rank
  (`server_metrics_export.jsonl`) — chưa phân tích chi tiết (cần re-extract
  giữ label), backlog đã định lượng đủ.

**Kết luận 1.2:** TTFT p99 96.9 s là hệ quả chọn lối policy (bảo vệ
ITL decode), không phải cạn tài nguyên — GPU vẫn
77–97% nhưng đang bận với batch decode mỏng (1.1).

### 1.3 Mất nhịp rank-local

- Cùng một thời điểm, running-req chênh giữa rank: **spread p50 =
  3, p95 = 5, max =
  9** (`analysis/sglang_rank_ts.csv`, max−min
  `num_running_reqs` qua 4.289 timestamp).
- **2.7% thời gian** có rank 0-running trong khi
  rank khác ≥3 running (nguồn trên) — rank đang prefill không decode được
  (attn-DP: 1 scheduler độc lập/GPU, prefill chunk chiếm slot forward).

## 2. Global dynamo routing: mất cân bằng giữa các GPU/rank

### 2.1 Bề mặt GPU util KHÔNG mất cân bằng

- Mean util 8 GPU: 70.9–73.3%
  (chênh 2.4 điểm), p50 =
  98% cả 8 (`gpu_samples.csv`). Cross-GPU spread tại mỗi
  giây: p50 = **1 điểm**, p95 = 23,
  max = 100.
- Lý do kiến trúc: MoE EP=8 all-to-all đồng bộ mọi rank trong mọi bước
  forward (cơ chế 1.1) — rank ít request vẫn phải tham gia exchange của
  rank khác → util bị ép cân bằng bề mặt.

### 2.2 Mất cân bằng thật ở workload/KV per rank

Benchmark window, `analysis/sglang_rank_ts.csv` + `analysis/dynamo_ts.csv`:

| Chỉ số (mean) | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | spread |
|---|---|---|---|---|---|---|---|---|---|
| Active prefill tokens (8888 `worker_active_prefill_tokens`) | 26.5k | 5.3k | 22.7k | 5.3k | 16.4k | 7.0k | 10.0k | 9.6k | **5.0×** |
| Active decode blocks (8888) | 7.4k | 5.0k | 6.0k | 6.1k | 6.4k | 6.1k | 7.2k | 5.6k | 1.5× |
| KV token usage (8081 `full_token_usage`) | 0.68 | 0.56 | 0.67 | 0.68 | 0.63 | 0.64 | 0.79 | 0.66 | 1.4× |
| Admission queue-req (8081) | 0.62 | 0.14 | 0.71 | 0.27 | 0.41 | 0.18 | 0.64 | 0.35 | — |
| Cache hit rate (8081) | 0.53 | 0.61 | 0.42 | 0.70 | 0.78 | 0.78 | 0.72 | 0.71 | — |

- Tại cùng thời điểm, KV usage chênh giữa rank: **spread p50 =
  0.51, p95 = 0.78**
  (max−min `full_token_usage` qua 4.289 timestamp).
- Request count per rank (warmup loại 5%): 146/191/113/217/115/184/130/137 — đếm `dp_rank`
  trong 1.255 dòng `Selected worker ... dp_rank=N logit=`
  (`frontend_*.log`), join timestamp với `schedule_eval.jsonl` (±2 s).

### 2.3 Hệ quả của mất cân bằng

- **0 rank lần lượt chạm KV 100%**: 
  (`sglang:full_token_usage` ≥ 0.999) — khớp 0 dòng WARN
  retract (`engine_*.log`):  `KV cache pool is full. Retract
  requests.` Memory pressure là **rank-local** — do routing dồn, không
  phải tổng thể thiếu.
- Queue-wait per rank (join selector ↔ schedule_eval): r2
  **13.00 s** mean vs r1
  2.84 s; TTFT per rank: r2
  16.52 s vs r1
  5.29 s (r2 p99
  151 s).
- Correlation input size ↔ rank: rank ít request nhận input lớn nhất
  (r2: 97k token mean input) —
  router đang cân bằng theo *số request* bề ngoài nhưng chưa cân bằng theo
  *tải thực* (prefill token/KV).

### 2.4 Router overhead — không đáng kể

`analysis/dynamo_ts.csv` (8888, histogram delta trong window):
`dynamo_router_overhead_total_ms`: **1.188 decision, mean 13.33
ms** (chi tiết: scheduling 13.10, indexer_find_matches 0.160,
block_hashing 0.060, seq_hashing 0.010). Không đáng kể so
TTFT giây. Frontend queue pending max = 0 mọi
timestamp (`dynamo_frontend_router_queue_pending_requests`).

### 2.5 Công thức cost: có tín hiệu nhưng chưa đủ

- corr(cost `logit`, TTFT) = **+0.42**, corr(logit,
  queue_wait) = +0.19 (1.233 request join
  theo timestamp; `logit` từ dòng `Selected worker` frontend_*.log,
  `ttft_s`/`queue_wait_s` từ schedule_eval.jsonl).
- Quintile logit → median TTFT thực: Q1 1.90 s → Q5
  12.79 s (queue_wait 1.08 → 3.65 s):
  cost cao đoán đúng hướng chờ lâu hơn.
- Nhưng phương sai lớn chưa giải thích + hot-rank (2.2–2.3) vẫn bị dồn
  vượt KV → giả thuyết "công thức chưa tối ưu" **được ủng hộ một phần**.

Nghi vấn cơ chế:
- (a) kv-state của router hết hạn → cost ước tính theo state cũ.
- (b) weight `16 × prefill_blocks` có vẻ underestimate chi phí prefill thực
  (r0 nhận 26.5k active prefill token mean —
  nhiều nhất — mà cost chưa kéo request đi chỗ khác).

## 3. Health-check thông số engine (checklist mentor)

Benchmark window trừ khi ghi khác — nguồn: `analysis/sglang_rank_ts.csv`
(8081, per dp_rank), `analysis/dynamo_ts.csv` (8888), `gpu_samples.csv`,
`engine_*.log` (boot + WARN).

| Thông số | Giá trị | Ý nghĩa / Mô tả |
|---|---|---|
| GPU utilization | 77–97% mỗi GPU (window) | GPU bận suốt benchmark — không rảnh; hiệu quả thấp đến từ batch mỏng (§1.1), không phải idle |
| GPU power | mean 325–372 W/GPU | Tương ứng mức bận cao, ổn định |
| Token usage (KV device) — mean | r6 0.79 … r1 0.56 | Tỉ lệ lấp pool KV device của từng rank; mất cân bằng 1.4× theo rank (§2.2) |
| Token usage (KV device) — đỉnh | 0 rank chạm 1.00 |  — áp lực muộn run, rank-local do routing dồn |
| KV device capacity | 583.296 token/rank (33.39 GB) | Boot line 22:52:02 `KV Cache is allocated ... #tokens: 583.296` ×8 rank; kv_used mean 326k–458k |
| KV hicache host — pool | 2.469.952 token = 128 GB/rank | Boot 22:53:58 ×8 dòng; offload host cho radix cache phân tầng |
| KV hicache host — used | mean 0.86M → 1.68M | Dùng 35–68% pool; cũng mất cân bằng theo rank như KV device |
| Cache reuse (engine) | 87.5% (104.84M cached / 14.98M new) | Tổng token prefill cả run: tỉ lệ token được serve từ cache — cache engine hoạt động tốt |
| Cached tokens/request (frontend) | mean 96.645 token (n=1.039) | `dynamo_frontend_cached_tokens`: mỗi request trung bình 97k token input được lấy từ cache (input median 87k) |
| Cache hit rate per rank | 0.42–0.78 (mean) | Chênh theo rank phản ánh việc routing dồn request cùng prefix về cùng rank (kv-router chủ đích) |
| Retract | 0 lần, mỗi lần 1 request |  — request bị thu hồi khi rank đầy; hiếm, không sập throughput nhưng là dấu hiệu mất cân bằng (§2.3) |
| EAGLE accept len | 1.99–2.55 (per-minute mean) | Ngưỡng khỏe ≥2.0 — spec decode có phải vấn đề không, đọc cùng ITL |
| ITL (frontend đo) | mean 40.01 ms | `dynamo_frontend_inter_token_latency_seconds`; khớp client TPOT 38.2 ms → frontend không thêm độ trễ đáng kể |

## 4. Vai trò trong so sánh consistency

Run này là replicate của baseline 3320 (`c64_20261014_112105`), cùng config
từng tham số. Bảng so sánh 3 run + verdict từng focus:
`docs/focus_consistency_3runs.md` (tạo bằng `scripts/compare_focus_runs.py`).

## 5. Phụ lục: cách từng số được tính (reproduce)

- Trích metrics: `srun -p hardware-hcm -w hgx-h200-01 --cpus-per-task=8
  --mem=16G python3 -I scripts/extract_metrics_timeseries.py <run_dir>`
  (stream JSONL 4.4 GB trên node) → `analysis/{sglang_rank_ts,dynamo_ts}.csv`.
- Report này: `python3 scripts/gen_focus_report.py <run_dir>` — toàn bộ số
  liệu tính bằng `scripts/focus_report_lib.py` (per-rank mean, spread
  max−min per timestamp, join router↔engine theo timestamp ±2 s, histogram
  delta, per-minute tổng hợp engine log, corr + quintile).
- Join router↔engine: `frontend_*.log` dòng `Selected worker` (request_id,
  dp_rank, logit) match `schedule_eval.jsonl` theo timestamp ±2 s —
  request_id (frontend) ≠ rid (engine) nên không join trực tiếp được.
