# Focus Analysis — Run c64_20261015_144717: Engine Scheduling Policy & Dynamo Routing

RunID **c64_20261015_144717** (15-10-2026, full stack dynamo 1.4.2, `dfs-weight` c64, 8×H200
hgx-h200-01). Replicate kiểm tra **consistency bottleneck** so với baseline
`c64_20261014_112105` (3320) — cùng config từng tham số (seed 42, cold radix
cache, PROBE_DURATION 1800 s). Theo 2 hướng định hướng:

(1) local engine scheduling policy giới hạn GPU utilization,

(2) global dynamo routing gây mất cân bằng giữa các GPU/DPA rank. Kèm
health-check thông số: GPU, token usage, cache, KV cache.

Run dir: `results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261015_144717/`

## 0. Nguồn dữ liệu & quy ước cite

| Nguồn | Nội dung | Cách trích |
|---|---|---|
| `engine_*.log` | 9.241 dòng batch scheduler (Decode/Prefill), boot lines, WARN | dòng theo timestamp `HH:MM:SS` |
| `frontend_*.log` | 1.288 dòng worker-selection, 1.285 request-completed, 10.711 HTTP access (status 200) | dòng theo timestamp |
| `server_metrics_export.jsonl` | ~9.314 scrape × 2 endpoint (8081 `sglang:*` per dp_rank; 8888 `dynamo_*`) | trích sẵn → `analysis/sglang_rank_ts.csv`, `analysis/dynamo_ts.csv` |
| `gpu_samples.csv` | 29.696 mẫu nvidia-smi 1 s × 8 GPU | cột `utilization.gpu [%]` |
| `profile_export.jsonl` / `schedule_eval.jsonl` | 1.285/3.807 record per request | field-level |
| `profiling/pyspy_*.svg` ×8 | flame 25 Hz × 120 s × 8 scheduler | frame-level |

**Benchmark window = 15:16:44 → 15:46:44Z** (30 phút). Cách suy ra:
`profile_export.jsonl` → `metadata.request_start_ns`, `benchmark_phase`:
65 record phase `warmup`, record đầu phase `profiling` lúc
15:16:44; duration 1800 s từ `e2_metrics.json` (`run.duration_s`).
Mọi thống kê dưới đây tính trong window này trừ khi ghi khác.

## 1. Local engine scheduling policy (dfs-weight + prefill-delayer)

### 1.1 GPU "bận" nhưng không hiệu quả

- GPU util benchmark window: **79–96%** mọi GPU
  (`gpu_samples.csv`, per-minute mean; ví dụ phút 15:31: 91 82 81 83 80 83 81 87).
  Mean cả run (gồm warmup/drain): 73.6–77.4%.
- Fwd occupancy 86–95% các phút benchmark
  (`engine_*.log`, field `fwd occupancy` tổng hợp per-minute).
- Nhưng batch decode nhỏ: **running-req mean 2.60–3.93/rank**
  trong window (`analysis/sglang_rank_ts.csv`, `sglang:num_running_reqs` per
  dp_rank) — log cho thấy batch mỏng (ví dụ 15:22:53 `#running-req: 1,
  gen throughput: 46.10`; 15:44:55 `#running-req: 11, gen
  throughput: 319.15`) dù trần `--max-running-requests=256`.
- Hệ quả: output chỉ 531 tok/s trên 8×H200 (`e2_metrics.json`). Cơ chế:
  MoE EP=8 all-to-all đồng bộ mọi rank mỗi bước → chi phí cố định/trước
  (flame `profiling/pyspy_*.svg`: MoE forward + DeepEP `dispatch_b` chiếm
  ưu thế) — batch mỏng → latency-bound → util cao ≠ năng suất cao.

**Kết luận 1.1:** "GPU utilization bị giới hạn" đúng ở nghĩa *hiệu quả*
không phải *không rảnh*: scheduling để GPU bận với batch
2.60–3.93 req/rank thay vì dồn việc vào batch dày.

### 1.2 Prefill bị hoãn suốt benchmark — TTFT tail từ đây

- Backlog prefill duy trì **1.886–73.559
  pending token** (per-minute mean) ở các phút benchmark (`engine_*.log`,
  field `#pending-token`; ví dụ dòng 15:35:20 `#pending-token: 2.642,
  input throughput: 644.65`; đỉnh 15:26:43
  354.600).
- Admission queue gần rỗng: `#queue-req` mean 0.11–0.54/rank
  (`analysis/sglang_rank_ts.csv`, `sglang:num_queue_reqs`) — request chờ ở
  prefill backlog chứ không phải hàng admission. Đối lập queue-wait p99
  79 s (`e2_metrics.json`) ↔ queue-req ~0 → mâu thuẫn định vị
  được chỗ chờ.
- Cơ chế: `--enable-prefill-delayer` + `--schedule-policy=dfs-weight` bảo vệ
  request đang decode (prefill chen làm ITL giật) nên hoãn prefill mới.
- `sglang:prefill_delayer_outcomes` có đếm theo outcome per rank
  (`server_metrics_export.jsonl`) — chưa phân tích chi tiết (cần re-extract
  giữ label), backlog đã định lượng đủ.

**Kết luận 1.2:** TTFT p99 76.0 s là hệ quả chọn lối policy (bảo vệ
ITL decode), không phải cạn tài nguyên — GPU vẫn
79–96% nhưng đang bận với batch decode mỏng (1.1).

### 1.3 Mất nhịp rank-local

- Cùng một thời điểm, running-req chênh giữa rank: **spread p50 =
  3, p95 = 6, max =
  10** (`analysis/sglang_rank_ts.csv`, max−min
  `num_running_reqs` qua 4.262 timestamp).
- **1.7% thời gian** có rank 0-running trong khi
  rank khác ≥3 running (nguồn trên) — rank đang prefill không decode được
  (attn-DP: 1 scheduler độc lập/GPU, prefill chunk chiếm slot forward).

## 2. Global dynamo routing: mất cân bằng giữa các GPU/rank

### 2.1 Bề mặt GPU util KHÔNG mất cân bằng

- Mean util 8 GPU: 73.6–77.4%
  (chênh 3.8 điểm), p50 =
  98% cả 8 (`gpu_samples.csv`). Cross-GPU spread tại mỗi
  giây: p50 = **1 điểm**, p95 = 41,
  max = 100.
- Lý do kiến trúc: MoE EP=8 all-to-all đồng bộ mọi rank trong mọi bước
  forward (cơ chế 1.1) — rank ít request vẫn phải tham gia exchange của
  rank khác → util bị ép cân bằng bề mặt.

### 2.2 Mất cân bằng thật ở workload/KV per rank

Benchmark window, `analysis/sglang_rank_ts.csv` + `analysis/dynamo_ts.csv`:

| Chỉ số (mean) | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | spread |
|---|---|---|---|---|---|---|---|---|---|
| Active prefill tokens (8888 `worker_active_prefill_tokens`) | 3.5k | 8.6k | 11.5k | 5.0k | 5.3k | 14.3k | 15.9k | 10.3k | **4.5×** |
| Active decode blocks (8888) | 5.7k | 7.4k | 7.6k | 4.8k | 6.5k | 6.6k | 5.6k | 5.1k | 1.6× |
| KV token usage (8081 `full_token_usage`) | 0.63 | 0.73 | 0.75 | 0.76 | 0.71 | 0.67 | 0.57 | 0.63 | 1.3× |
| Admission queue-req (8081) | 0.14 | 0.39 | 0.31 | 0.54 | 0.11 | 0.47 | 0.33 | 0.26 | — |
| Cache hit rate (8081) | 0.77 | 0.73 | 0.67 | 0.41 | 0.65 | 0.67 | 0.51 | 0.65 | — |

- Tại cùng thời điểm, KV usage chênh giữa rank: **spread p50 =
  0.51, p95 = 0.79**
  (max−min `full_token_usage` qua 4.262 timestamp).
- Request count per rank (warmup loại 5%): 206/112/77/166/140/125/156/279 — đếm `dp_rank`
  trong 1.288 dòng `Selected worker ... dp_rank=N logit=`
  (`frontend_*.log`), join timestamp với `schedule_eval.jsonl` (±2 s).

### 2.3 Hệ quả của mất cân bằng

- **3 rank lần lượt chạm KV 100%**: r3 15:25:51, r6 15:41:34, r7 15:45:51
  (`sglang:full_token_usage` ≥ 0.999) — khớp 2 dòng WARN
  retract (`engine_*.log`): 15:25:53 `#retracted_reqs: 1`; 15:41:36 `#retracted_reqs: 1` `KV cache pool is full. Retract
  requests.` Memory pressure là **rank-local** — do routing dồn, không
  phải tổng thể thiếu.
- Queue-wait per rank (join selector ↔ schedule_eval): r5
  **8.65 s** mean vs r0
  2.52 s; TTFT per rank: r5
  11.62 s vs r0
  4.86 s (r5 p99
  119 s).
- Correlation input size ↔ rank: rank ít request nhận input lớn nhất
  (r2: 139k token mean input) —
  router đang cân bằng theo *số request* bề ngoài nhưng chưa cân bằng theo
  *tải thực* (prefill token/KV).

### 2.4 Router overhead — không đáng kể

`analysis/dynamo_ts.csv` (8888, histogram delta trong window):
`dynamo_router_overhead_total_ms`: **1.221 decision, mean 9.59
ms** (chi tiết: scheduling 9.39, indexer_find_matches 0.121,
block_hashing 0.059, seq_hashing 0.010). Không đáng kể so
TTFT giây. Frontend queue pending max = 0 mọi
timestamp (`dynamo_frontend_router_queue_pending_requests`).

### 2.5 Công thức cost: có tín hiệu nhưng chưa đủ

- corr(cost `logit`, TTFT) = **+0.52**, corr(logit,
  queue_wait) = +0.18 (1.261 request join
  theo timestamp; `logit` từ dòng `Selected worker` frontend_*.log,
  `ttft_s`/`queue_wait_s` từ schedule_eval.jsonl).
- Quintile logit → median TTFT thực: Q1 1.95 s → Q5
  10.18 s (queue_wait 1.12 → 1.78 s):
  cost cao đoán đúng hướng chờ lâu hơn.
- Nhưng phương sai lớn chưa giải thích + hot-rank (2.2–2.3) vẫn bị dồn
  vượt KV → giả thuyết "công thức chưa tối ưu" **được ủng hộ một phần**.

Nghi vấn cơ chế:
- (a) kv-state của router hết hạn → cost ước tính theo state cũ.
- (b) weight `16 × prefill_blocks` có vẻ underestimate chi phí prefill thực
  (r6 nhận 15.9k active prefill token mean —
  nhiều nhất — mà cost chưa kéo request đi chỗ khác).

## 3. Health-check thông số engine (checklist mentor)

Benchmark window trừ khi ghi khác — nguồn: `analysis/sglang_rank_ts.csv`
(8081, per dp_rank), `analysis/dynamo_ts.csv` (8888), `gpu_samples.csv`,
`engine_*.log` (boot + WARN).

| Thông số | Giá trị | Ý nghĩa / Mô tả |
|---|---|---|
| GPU utilization | 79–96% mỗi GPU (window) | GPU bận suốt benchmark — không rảnh; hiệu quả thấp đến từ batch mỏng (§1.1), không phải idle |
| GPU power | mean 329–358 W/GPU | Tương ứng mức bận cao, ổn định |
| Token usage (KV device) — mean | r3 0.76 … r6 0.57 | Tỉ lệ lấp pool KV device của từng rank; mất cân bằng 1.3× theo rank (§2.2) |
| Token usage (KV device) — đỉnh | 3 rank chạm 1.00 | r3 15:25:51, r6 15:41:34, r7 15:45:51 — áp lực muộn run, rank-local do routing dồn |
| KV device capacity | 583.296 token/rank (33.39 GB) | Boot line 14:49:15 `KV Cache is allocated ... #tokens: 583.296` ×8 rank; kv_used mean 331k–446k |
| KV hicache host — pool | 2.469.952 token = 128 GB/rank | Boot 14:51:11 ×8 dòng; offload host cho radix cache phân tầng |
| KV hicache host — used | mean 0.58M → 1.74M | Dùng 23–70% pool; cũng mất cân bằng theo rank như KV device |
| Cache reuse (engine) | 87.9% (107.76M cached / 14.87M new) | Tổng token prefill cả run: tỉ lệ token được serve từ cache — cache engine hoạt động tốt |
| Cached tokens/request (frontend) | mean 96.302 token (n=1.067) | `dynamo_frontend_cached_tokens`: mỗi request trung bình 96k token input được lấy từ cache (input median 87k) |
| Cache hit rate per rank | 0.41–0.77 (mean) | Chênh theo rank phản ánh việc routing dồn request cùng prefix về cùng rank (kv-router chủ đích) |
| Retract | 2 lần, mỗi lần 1 request | 15:25:53 `#retracted_reqs: 1`; 15:41:36 `#retracted_reqs: 1` — request bị thu hồi khi rank đầy; hiếm, không sập throughput nhưng là dấu hiệu mất cân bằng (§2.3) |
| EAGLE accept len | 1.91–2.45 (per-minute mean) | Ngưỡng khỏe ≥2.0 — spec decode có phải vấn đề không, đọc cùng ITL |
| ITL (frontend đo) | mean 42.78 ms | `dynamo_frontend_inter_token_latency_seconds`; khớp client TPOT 38.7 ms → frontend không thêm độ trễ đáng kể |

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
