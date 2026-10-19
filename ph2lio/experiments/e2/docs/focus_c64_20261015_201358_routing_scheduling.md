# Focus Analysis — Run c64_20261015_201358: Engine Scheduling Policy & Dynamo Routing

RunID **c64_20261015_201358** (15-10-2026, full stack dynamo 1.4.2, `dfs-weight` c64, 8×H200
hgx-h200-01). Replicate kiểm tra **consistency bottleneck** so với baseline
`c64_20261014_112105` (3320) — cùng config từng tham số (seed 42, cold radix
cache, PROBE_DURATION 1800 s). Theo 2 hướng định hướng:

(1) local engine scheduling policy giới hạn GPU utilization,

(2) global dynamo routing gây mất cân bằng giữa các GPU/DPA rank. Kèm
health-check thông số: GPU, token usage, cache, KV cache.

Run dir: `results/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261015_201358/`

## 0. Nguồn dữ liệu & quy ước cite

| Nguồn | Nội dung | Cách trích |
|---|---|---|
| `engine_*.log` | 9.376 dòng batch scheduler (Decode/Prefill), boot lines, WARN | dòng theo timestamp `HH:MM:SS` |
| `frontend_*.log` | 1.271 dòng worker-selection, 1.268 request-completed, 10.572 HTTP access (status 200) | dòng theo timestamp |
| `server_metrics_export.jsonl` | ~9.123 scrape × 2 endpoint (8081 `sglang:*` per dp_rank; 8888 `dynamo_*`) | trích sẵn → `analysis/sglang_rank_ts.csv`, `analysis/dynamo_ts.csv` |
| `gpu_samples.csv` | 30.536 mẫu nvidia-smi 1 s × 8 GPU | cột `utilization.gpu [%]` |
| `profile_export.jsonl` / `schedule_eval.jsonl` | 1.268/5.078 record per request | field-level |
| `profiling/pyspy_*.svg` ×8 | flame 25 Hz × 120 s × 8 scheduler | frame-level |

**Benchmark window = 20:45:37 → 21:15:37Z** (30 phút). Cách suy ra:
`profile_export.jsonl` → `metadata.request_start_ns`, `benchmark_phase`:
65 record phase `warmup`, record đầu phase `profiling` lúc
20:45:37; duration 1800 s từ `e2_metrics.json` (`run.duration_s`).
Mọi thống kê dưới đây tính trong window này trừ khi ghi khác.

## 1. Local engine scheduling policy (dfs-weight + prefill-delayer)

### 1.1 GPU "bận" nhưng không hiệu quả

- GPU util benchmark window: **81–97%** mọi GPU
  (`gpu_samples.csv`, per-minute mean; ví dụ phút 21:00: 87 87 87 88 89 85 88 87).
  Mean cả run (gồm warmup/drain): 72.4–73.1%.
- Fwd occupancy 89–95% các phút benchmark
  (`engine_*.log`, field `fwd occupancy` tổng hợp per-minute).
- Nhưng batch decode nhỏ: **running-req mean 2.55–4.50/rank**
  trong window (`analysis/sglang_rank_ts.csv`, `sglang:num_running_reqs` per
  dp_rank) — log cho thấy batch mỏng (ví dụ 20:58:27 `#running-req: 1,
  gen throughput: 47.19`; 21:10:01 `#running-req: 14, gen
  throughput: 354.61`) dù trần `--max-running-requests=256`.
- Hệ quả: output chỉ 520 tok/s trên 8×H200 (`e2_metrics.json`). Cơ chế:
  MoE EP=8 all-to-all đồng bộ mọi rank mỗi bước → chi phí cố định/trước
  (flame `profiling/pyspy_*.svg`: MoE forward + DeepEP `dispatch_b` chiếm
  ưu thế) — batch mỏng → latency-bound → util cao ≠ năng suất cao.

**Kết luận 1.1:** "GPU utilization bị giới hạn" đúng ở nghĩa *hiệu quả*
không phải *không rảnh*: scheduling để GPU bận với batch
2.55–4.50 req/rank thay vì dồn việc vào batch dày.

### 1.2 Prefill bị hoãn suốt benchmark — TTFT tail từ đây

- Backlog prefill duy trì **5.273–78.131
  pending token** (per-minute mean) ở các phút benchmark (`engine_*.log`,
  field `#pending-token`; ví dụ dòng 21:14:22 `#pending-token: 4.418,
  input throughput: 6176.22`; đỉnh 21:08:36
  279.246).
- Admission queue gần rỗng: `#queue-req` mean 0.17–0.58/rank
  (`analysis/sglang_rank_ts.csv`, `sglang:num_queue_reqs`) — request chờ ở
  prefill backlog chứ không phải hàng admission. Đối lập queue-wait p99
  88 s (`e2_metrics.json`) ↔ queue-req ~0 → mâu thuẫn định vị
  được chỗ chờ.
- Cơ chế: `--enable-prefill-delayer` + `--schedule-policy=dfs-weight` bảo vệ
  request đang decode (prefill chen làm ITL giật) nên hoãn prefill mới.
- `sglang:prefill_delayer_outcomes` có đếm theo outcome per rank
  (`server_metrics_export.jsonl`) — chưa phân tích chi tiết (cần re-extract
  giữ label), backlog đã định lượng đủ.

**Kết luận 1.2:** TTFT p99 99.4 s là hệ quả chọn lối policy (bảo vệ
ITL decode), không phải cạn tài nguyên — GPU vẫn
81–97% nhưng đang bận với batch decode mỏng (1.1).

### 1.3 Mất nhịp rank-local

- Cùng một thời điểm, running-req chênh giữa rank: **spread p50 =
  3, p95 = 7, max =
  12** (`analysis/sglang_rank_ts.csv`, max−min
  `num_running_reqs` qua 4.388 timestamp).
- **3.9% thời gian** có rank 0-running trong khi
  rank khác ≥3 running (nguồn trên) — rank đang prefill không decode được
  (attn-DP: 1 scheduler độc lập/GPU, prefill chunk chiếm slot forward).

## 2. Global dynamo routing: mất cân bằng giữa các GPU/rank

### 2.1 Bề mặt GPU util KHÔNG mất cân bằng

- Mean util 8 GPU: 72.4–73.1%
  (chênh 0.8 điểm), p50 =
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
| Active prefill tokens (8888 `worker_active_prefill_tokens`) | 15.7k | 10.3k | 11.9k | 11.8k | 5.3k | 9.4k | 20.2k | 21.0k | **3.9×** |
| Active decode blocks (8888) | 6.1k | 6.9k | 6.2k | 6.7k | 5.0k | 6.1k | 6.8k | 6.3k | 1.4× |
| KV token usage (8081 `full_token_usage`) | 0.60 | 0.74 | 0.68 | 0.67 | 0.55 | 0.73 | 0.66 | 0.66 | 1.4× |
| Admission queue-req (8081) | 0.51 | 0.58 | 0.46 | 0.39 | 0.17 | 0.42 | 0.45 | 0.39 | — |
| Cache hit rate (8081) | 0.63 | 0.75 | 0.66 | 0.61 | 0.58 | 0.56 | 0.73 | 0.59 | — |

- Tại cùng thời điểm, KV usage chênh giữa rank: **spread p50 =
  0.57, p95 = 0.81**
  (max−min `full_token_usage` qua 4.388 timestamp).
- Request count per rank (warmup loại 5%): 188/79/197/181/226/105/148/124 — đếm `dp_rank`
  trong 1.271 dòng `Selected worker ... dp_rank=N logit=`
  (`frontend_*.log`), join timestamp với `schedule_eval.jsonl` (±2 s).

### 2.3 Hệ quả của mất cân bằng

- **0 rank lần lượt chạm KV 100%**: 
  (`sglang:full_token_usage` ≥ 0.999) — khớp 0 dòng WARN
  retract (`engine_*.log`):  `KV cache pool is full. Retract
  requests.` Memory pressure là **rank-local** — do routing dồn, không
  phải tổng thể thiếu.
- Queue-wait per rank (join selector ↔ schedule_eval): r1
  **11.19 s** mean vs r2
  4.89 s; TTFT per rank: r1
  15.70 s vs r2
  7.32 s (r1 p99
  267 s).
- Correlation input size ↔ rank: rank ít request nhận input lớn nhất
  (r1: 123k token mean input) —
  router đang cân bằng theo *số request* bề ngoài nhưng chưa cân bằng theo
  *tải thực* (prefill token/KV).

### 2.4 Router overhead — không đáng kể

`analysis/dynamo_ts.csv` (8888, histogram delta trong window):
`dynamo_router_overhead_total_ms`: **1.206 decision, mean 7.37
ms** (chi tiết: scheduling 7.16, indexer_find_matches 0.130,
block_hashing 0.056, seq_hashing 0.009). Không đáng kể so
TTFT giây. Frontend queue pending max = 0 mọi
timestamp (`dynamo_frontend_router_queue_pending_requests`).

### 2.5 Công thức cost: có tín hiệu nhưng chưa đủ

- corr(cost `logit`, TTFT) = **+0.37**, corr(logit,
  queue_wait) = +0.15 (1.248 request join
  theo timestamp; `logit` từ dòng `Selected worker` frontend_*.log,
  `ttft_s`/`queue_wait_s` từ schedule_eval.jsonl).
- Quintile logit → median TTFT thực: Q1 1.89 s → Q5
  11.22 s (queue_wait 1.09 → 2.75 s):
  cost cao đoán đúng hướng chờ lâu hơn.
- Nhưng phương sai lớn chưa giải thích + hot-rank (2.2–2.3) vẫn bị dồn
  vượt KV → giả thuyết "công thức chưa tối ưu" **được ủng hộ một phần**.

Nghi vấn cơ chế:
- (a) kv-state của router hết hạn → cost ước tính theo state cũ.
- (b) weight `16 × prefill_blocks` có vẻ underestimate chi phí prefill thực
  (r7 nhận 21.0k active prefill token mean —
  nhiều nhất — mà cost chưa kéo request đi chỗ khác).

## 3. Health-check thông số engine (checklist mentor)

Benchmark window trừ khi ghi khác — nguồn: `analysis/sglang_rank_ts.csv`
(8081, per dp_rank), `analysis/dynamo_ts.csv` (8888), `gpu_samples.csv`,
`engine_*.log` (boot + WARN).

| Thông số | Giá trị | Ý nghĩa / Mô tả |
|---|---|---|
| GPU utilization | 81–97% mỗi GPU (window) | GPU bận suốt benchmark — không rảnh; hiệu quả thấp đến từ batch mỏng (§1.1), không phải idle |
| GPU power | mean 321–372 W/GPU | Tương ứng mức bận cao, ổn định |
| Token usage (KV device) — mean | r1 0.74 … r4 0.55 | Tỉ lệ lấp pool KV device của từng rank; mất cân bằng 1.4× theo rank (§2.2) |
| Token usage (KV device) — đỉnh | 0 rank chạm 1.00 |  — áp lực muộn run, rank-local do routing dồn |
| KV device capacity | 583.296 token/rank (33.39 GB) | Boot line 20:16:15 `KV Cache is allocated ... #tokens: 583.296` ×8 rank; kv_used mean 318k–434k |
| KV hicache host — pool | 2.469.952 token = 128 GB/rank | Boot 20:18:12 ×8 dòng; offload host cho radix cache phân tầng |
| KV hicache host — used | mean 0.58M → 1.71M | Dùng 23–69% pool; cũng mất cân bằng theo rank như KV device |
| Cache reuse (engine) | 87.0% (103.91M cached / 15.56M new) | Tổng token prefill cả run: tỉ lệ token được serve từ cache — cache engine hoạt động tốt |
| Cached tokens/request (frontend) | mean 95.644 token (n=1.049) | `dynamo_frontend_cached_tokens`: mỗi request trung bình 96k token input được lấy từ cache (input median 87k) |
| Cache hit rate per rank | 0.56–0.75 (mean) | Chênh theo rank phản ánh việc routing dồn request cùng prefix về cùng rank (kv-router chủ đích) |
| Retract | 0 lần, mỗi lần 1 request |  — request bị thu hồi khi rank đầy; hiếm, không sập throughput nhưng là dấu hiệu mất cân bằng (§2.3) |
| EAGLE accept len | 1.98–2.43 (per-minute mean) | Ngưỡng khỏe ≥2.0 — spec decode có phải vấn đề không, đọc cùng ITL |
| ITL (frontend đo) | mean 41.55 ms | `dynamo_frontend_inter_token_latency_seconds`; khớp client TPOT 40.1 ms → frontend không thêm độ trễ đáng kể |

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
