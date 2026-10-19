# Focus Analysis — Cell 3320: Engine Scheduling Policy & Dynamo Routing

RunID **3320** (14-10-2026, full stack dynamo 1.4.2, `dfs-weight` c64, 8×H200
hgx-h200-01). Theo 2 hướng định hướng:

(1) local engine scheduling policy giới hạn GPU utilization, 

(2) global dynamo routing
gây mất cân bằng giữa các GPU/DPA rank. Kèm health-check thông số: GPU,
token usage, cache, KV cache.

Run dir: `results/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261014_112105/`

## 0. Nguồn dữ liệu & quy ước cite

| Nguồn | Nội dung | Cách trích |
|---|---|---|
| `engine_*.log` | 9.518 dòng batch scheduler (Decode/Prefill), boot lines, WARN | dòng theo timestamp `HH:MM:SS` |
| `frontend_*.log` | 16.011 dòng: 1.237 worker-selection, 1.234 request-completed, 10.944 HTTP access | dòng theo timestamp |
| `server_metrics_export.jsonl` (4.4 GB) | 9.588 scrape × 2 endpoint (8081 `sglang:*` per dp_rank; 8888 `dynamo_*`) | trích sẵn → `analysis/sglang_rank_ts.csv`, `analysis/dynamo_ts.csv` |
| `gpu_samples.csv` | 30.232 mẫu nvidia-smi 1 s × 8 GPU | cột `utilization.gpu [%]` |
| `profile_export.jsonl` / `schedule_eval.jsonl` | 1.234/1.237 record per request | field-level |
| `profiling/pyspy_*.svg` ×8 | flame 25 Hz × 120 s × 8 scheduler | frame-level |

**Benchmark window = 11:51:16 → 12:21:16Z** (30 phút). Cách suy ra:
`profile_export.jsonl` → `metadata.request_start_ns` — gap 82.5 s giữa
warmup cuối (11:49:54) và benchmark đầu (11:51:16); trước đó là warmup
prefill-only (engine log: 11:27–11:50 không có dòng `Decode batch` nào).
Mọi thống kê dưới đây tính trong window này trừ khi ghi khác.

## 1. Local engine scheduling policy (dfs-weight + prefill-delayer)

### 1.1 GPU "bận" nhưng không hiệu quả

- GPU util benchmark window: **90–98%** mọi GPU (`gpu_samples.csv`, per-minute
  mean; ví dụ phút 12:05: 95/94/94/95/93/94/94/93). Mean cả run (gồm
  warmup/drain): 74.8–77.0%.
- Fwd occupancy 88–95% các phút benchmark (`engine_*.log`, field `fwd
  occupancy` tổng hợp per-minute).
- Nhưng batch decode nhỏ: **running-req mean 2.75–3.75/rank** trong window
  (`analysis/sglang_rank_ts.csv`, `sglang:num_running_reqs` per dp_rank) —
  log cho thấy 1–4 request/step (dòng `Decode batch, #running-req: 1–4`,
  11:51:37 `#running-req: 4 ... gen throughput: 35.88`; 11:51:58
  `#running-req: 1 ... 27.04`) dù trần `--max-running-requests=256`.
- Hệ quả: output chỉ 522 tok/s trên 8×H200 (`e2_metrics.json`). Cơ chế:
  MoE EP=8 all-to-all đồng bộ mọi rank mỗi bước → chi phí cố định/trước
  (flame `profiling/pyspy_*.svg`: ~25% samples trong DeepEP `dispatch_b`,
  ~52% MoE forward) — batch mỏng → latency-bound → util cao ≠ năng suất cao.

**Kết luận 1.1:** "GPU utilization bị giới hạn" đúng ở nghĩa *hiệu quả*
không phải *không rảnh*: scheduling để GPU bận với batch 2.75–3.75 req/rank
thay vì dồn việc vào batch dày.

### 1.2 Prefill bị hoãn suốt benchmark — TTFT tail từ đây

- Backlog prefill duy trì **5–112k pending token** ở mọi phút benchmark
  (`engine_*.log`, field `#pending-token` tổng hợp per-minute; ví dụ dòng
  12:00:11 `#pending-token: 44029, input throughput: 7172.36`).
- Admission queue gần rỗng: `#queue-req` mean 0.09–0.56/rank
  (`analysis/sglang_rank_ts.csv`, `sglang:num_queue_reqs`) — request chờ ở
  prefill backlog chứ không phải hàng admission. Đối lập queue-wait p99
  247 s (`e2_metrics.json`) ↔ queue-req ~0 → mâu thuẫn định vị được chỗ chờ.
- Cơ chế: `--enable-prefill-delayer` + `--schedule-policy=dfs-weight` bảo vệ
  request đang decode (prefill chen làm ITL giật) nên hoãn prefill mới.
- `sglang:prefill_delayer_outcomes` có đếm theo outcome per rank
  (`server_metrics_export.jsonl`, labels `actual_execution`/
  `input_estimation`/`output_allow`) — chưa phân tích chi tiết (cần
  re-extract giữ label), backlog đã định lượng đủ.

**Kết luận 1.2:** TTFT p99 75.4 s là hệ quả chọn lối policy (bảo vệ ITL
decode), không phải cạn tài nguyên — GPU vẫn 90–98% nhưng đang bận với
batch decode mỏng (1.1).

### 1.3 Mất nhịp rank-local

- Cùng một thời điểm, running-req chênh giữa rank: **spread p50 = 3, p95 =
  6, max = 9** (`analysis/sglang_rank_ts.csv`, max−min `num_running_reqs`
  qua 4.433 timestamp).
- **3.3% thời gian** có rank 0-running trong khi rank khác ≥3 running
  (nguồn trên) — rank đang prefill không decode được (attn-DP: 1 scheduler
  độc lập/GPU, prefill chunk chiếm slot forward).

## 2. Global dynamo routing: mất cân bằng giữa các GPU/rank

### 2.1 Bề mặt GPU utilization KHÔNG mất cân bằng

- Mean util 8 GPU: 74.8–77.0% (chênh 2.2 điểm), p50 = 98% cả 8
  (`gpu_samples.csv`). Cross-GPU spread tại mỗi giây: p50 = **1 điểm**,
  p95 = 48, max = 100.
- Lý do kiến trúc: MoE EP=8 all-to-all đồng bộ mọi rank trong mọi bước
  forward (cơ chế 1.1) — rank ít request vẫn phải tham gia exchange của
  rank khác → util bị ép cân bằng bề mặt.

### 2.2 Mất cân bằng ở workload/KV per rank

Benchmark window, `analysis/sglang_rank_ts.csv` + `analysis/dynamo_ts.csv`:

| Chỉ số (mean) | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | spread |
|---|---|---|---|---|---|---|---|---|---|
| Active prefill tokens (8888 `worker_active_prefill_tokens`) | 15.0k | 11.3k | **21.5k** | 6.5k | 5.7k | 11.8k | 13.3k | **3.8k** | **5.6×** |
| Active decode blocks (8888) | **7.5k** | 6.3k | 6.8k | 5.4k | 5.5k | 6.2k | 5.9k | 5.3k | 1.4× |
| KV token usage (8081 `full_token_usage`) | **0.82** | 0.77 | 0.68 | 0.58 | 0.58 | 0.63 | 0.74 | 0.61 | 1.4× |
| Admission queue-req (8081) | 0.48 | 0.48 | 0.56 | 0.16 | 0.15 | 0.32 | 0.22 | 0.09 | 6× |
| Cache hit rate (8081) | 0.74 | 0.66 | 0.64 | 0.53 | 0.76 | 0.54 | 0.59 | 0.69 | — |

- Tại cùng thời điểm, KV usage chênh giữa rank: **spread p50 = 0.56, p95 =
  0.79** (max−min `full_token_usage` qua 4.433 timestamp) — rank đầy 0.9
  trong khi rank khác 0.3.
- Request count per rank (warmup loại 5%): 84/130/152/221/171/151/72/173 —
  đếm `dp_rank` trong 1.237 dòng `Selected worker ... dp_rank=N logit=`
  (`frontend_*.log`), join timestamp với `schedule_eval.jsonl` (±2 s).

### 2.3 Hệ quả của mất cân bằng

- **4 rank lần lượt chạm KV 100% → retract**: r2 12:10:11, r6 12:18:02,
  r1 12:20:04, r0 12:21:12 (`sglang:full_token_usage` ≥ 0.999) — khớp 3
  dòng WARN retract (`engine_*.log`): 12:10:11, 12:18:16, 12:20:06
  `KV cache pool is full. Retract requests. #retracted_reqs: 1`. Memory
  pressure là **rank-local** — do routing dồn, không phải tổng thể thiếu.
- Queue-wait per rank (join selector ↔ schedule_eval): r0 **9.61 s** mean
  vs r3 2.70 s; TTFT per rank: r0 11.25 s vs r3/7 ~4.0 s (r2 p99 264 s).
- Correlation input size ↔ rank: rank ít request nhận input lớn nhất
  (r0: 107k token mean input) — router đang cân bằng theo *số request*
  bề ngoài nhưng chưa cân bằng theo *tải thực* (prefill token/KV).

### 2.4 Router overhead — không đáng kể

`analysis/dynamo_ts.csv` (8888, histogram delta trong window):
`dynamo_router_overhead_total_ms`: **1.172 decision, mean 4.80 ms**
(chi tiết: scheduling 4.61, indexer_find_matches 0.118, block_hashing
0.060, seq_hashing 0.010). Không đáng kể so TTFT giây. Frontend queue
pending = 0 mọi timestamp (`dynamo_frontend_router_queue_pending_requests`).

### 2.5 Công thức cost: có tín hiệu nhưng chưa đủ

- corr(cost `logit`, TTFT) = **+0.40**, corr(logit, queue_wait) = +0.32
  (1.154 request join theo timestamp; `logit` từ dòng `Selected worker`
  frontend_*.log, `ttft_s`/`queue_wait_s` từ schedule_eval.jsonl).
- Quintile logit → median TTFT thực: Q1 1.94 s → Q5 6.35 s (queue_wait
  1.09 → 1.92 s): cost cao đoán đúng hướng chờ lâu hơn.
- Nhưng phương sai lớn chưa giải thích + hot-rank (2.2–2.3) vẫn bị dồn
  vượt KV → giả thuyết "công thức chưa tối ưu" **được ủng hộ một phần**.
  
  
Nghi vấn cơ chế: 
- (a) kv-state của router hết hạn → cost ước tính theo state cũ.
- (b) weight `16 × prefill_blocks` có vẻ underestimate chi phí prefill thực
  (r2 nhận 21.5k active prefill token mean — nhiều nhất — mà cost chưa
  kéo request đi chỗ khác).

## 3. Health-check thông số engine

`analysis/sglang_rank_ts.csv`
(8081, per dp_rank), `analysis/dynamo_ts.csv` (8888), `gpu_samples.csv`,
`engine_*.log` (boot + WARN).

| Thông số | Giá trị | Ý nghĩa / Mô tả |
|---|---|---|
| GPU utilization | 90–98% mỗi GPU (window) | GPU bận suốt benchmark — không rảnh; hiệu quả thấp đến từ batch mỏng (§1.1), không phải idle |
| GPU power | mean 324–372 W/GPU | Tương ứng mức bận cao, ổn định |
| Token usage (KV device) — mean | r0 0.82 … r3/r4 0.58 | Tỉ lệ lấp pool KV device của từng rank; mất cân bằng 1.4× theo rank (§2.2) |
| Token usage (KV device) — đỉnh | 4 rank chạm 1.00 (12:10–12:21) | r2 12:10:11, r6 12:18:02, r1 12:20:04, r0 12:21:12 — áp lực muộn run, rank-local do routing dồn |
| KV device capacity | 583.296 token/rank (33.39 GB) | Boot line 11:23:09 `KV Cache is allocated ... #tokens: 583296` ×8 rank; kv_used mean 336k–477k |
| KV hicache host — pool | 2.469.952 token = 128 GB/rank | Boot 11:25:06 ×8 rank; offload host cho radix cache phân tầng |
| KV hicache host — used | mean 0.50M (r3) → 1.59M (r0) | Dùng 20–64% pool; cũng mất cân bằng theo rank như KV device |
| Cache reuse (engine) | 87.5% (103.88M cached / 14.87M new) | Tổng token prefill: tỉ lệ token được serve từ cache — cache engine hoạt động rất tốt |
| Cached tokens/request (frontend) | mean 96.109 token (n=1.032) | `dynamo_frontend_cached_tokens`: mỗi request trung bình 96k token input được lấy từ cache (input median ~93k) → gần như trọn prompt được reuse |
| Cache hit rate per rank | 0.53–0.76 (mean) | Chênh theo rank phản ánh việc routing dồn request cùng prefix về cùng rank (kv-router chủ đích) |
| Retract | 3 lần, mỗi lần 1 request | WARN 12:10:11 / 12:18:16 / 12:20:06 `KV cache pool is full. Retract requests.` — request bị thu hồi khi rank đầy; hiếm, không sập throughput nhưng là dấu hiệu mất cân bằng (§2.3) |
| EAGLE accept len | 1.90–2.58 | ≥ ngưỡng khỏe 2.0 gần hết các phút benchmark — spec decode không phải vấn đề |
| ITL (frontend đo) | mean 42 ms | `dynamo_frontend_inter_token_latency_seconds`; khớp client TPOT 38.4 ms → frontend không thêm độ trễ đáng kể |