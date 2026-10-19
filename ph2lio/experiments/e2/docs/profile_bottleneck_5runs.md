# [Report] - Profiling Bottleneck trên config Prod GLM5.3 (13/10/2026)

So sánh thông số Engine SGLang để profile bottleneck giữa 5 run có cùng cấu hình Production GLM5.3, workload AgentX

`BASE_DIR = /shared/loipp/sglang/ph2lio/experiments/e2/results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/`

| Run | Stamp | Run dir |
|---|---|---|
| 1 | c64_20261014_112105 | `$BASE_DIR/c64_20261014_112105` |
| 2 | c64_20261015_133031 | `$BASE_DIR/c64_20261015_133031` |
| 3 | c64_20261015_144717 | `$BASE_DIR/c64_20261015_144717` |
| 4 | c64_20261015_201358 | `$BASE_DIR/c64_20261015_201358` |
| 5 | c64_20261015_224930 | `$BASE_DIR/c64_20261015_224930` |

## 1. Bảng so sánh thông số giữa các run

| Chỉ số | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Consistent? |
|---|---|---|---|---|---|---|
| Request/output throughput (tok/s) | 522 | 532 | 531 | 520 | 528 | ✓ |
| TTFT mean (s) | 7.1 | 7.6 | 6.8 | 8.2 | 8.2 | ✓ |
| TTFT p99 (s) | 75.4 | 89.7 | 76.0 | 99.4 | 96.9 | ✓ |
| Running-req mean/rank (min–max) | 2.75–3.75 | 2.31–4.68 | 2.60–3.93 | 2.55–4.50 | 2.45–3.66 | ✓ |
| Pending backlog per-minute (min–max) | 5.970–132.265 | 5.443–99.600 | 5.886–73.559 | 5.273–78.131 | 11.324–83.241 | ✓ |
| Active prefill token spread | 5.6 | 5.1 | 4.5 | 3.9 | 5.0 | ✓ |
| KV usage mean (min–max) | 0.58–0.82 | 0.47–0.78 | 0.57–0.76 | 0.55–0.74 | 0.56–0.79 | ✓ |
| KV usage spread p50 | 0.56 | 0.56 | 0.51 | 0.57 | 0.51 | ✓ |
| #rank chạm KV 1.00 | 4 | 0 | 3 | 0 | 0 |  |
| #retract | 3 | 0 | 2 | 0 | 0 |  |
| corr(logit, TTFT) | +0.40 | +0.26 | +0.52 | +0.37 | +0.42 | ✓ cùng dấu |
| Cache reuse engine (%) | 87.5 | 88.1 | 87.9 | 87.0 | 87.5 | ✓ |
| Cached tokens/request | 96111 | 96206 | 96302 | 95644 | 96645 | ✓ |
| EAGLE accept len (per-minute min–max) | 1.97–2.31 | 2.00–2.35 | 1.91–2.45 | 1.98–2.43 | 1.99–2.55 | ✓ |
| ITL mean (ms) | 42.0 | 37.7 | 42.8 | 41.5 | 40.0 | ✓ |
| GPU util window (min–max %) | 79–97 | 77–97 | 79–96 | 81–97 | 77–97 | ✓ |

## 2. Mất cân bằng workload/KV per rank

Thống kê per rank trong benchmark window (hàng workload = min–max mean qua 8 rank, hàng join = rank cực đại vs cực tiểu)

| Chỉ số (per rank) | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|---|---|---|---|---|---|
| Active prefill tokens | 3.8k–21.6k (5.6×) | 4.3k–21.5k (5.1×) | 3.5k–15.9k (4.5×) | 5.3k–21.0k (3.9×) | 5.3k–26.5k (5.0×) |
| Active decode blocks | 5.3k–7.5k (1.4×) | 4.2k–7.4k (1.8×) | 4.8k–7.6k (1.6×) | 5.0k–6.9k (1.4×) | 5.0k–7.4k (1.5×) |
| KV token usage `full_token_usage` | 0.58–0.82 | 0.47–0.78 | 0.57–0.76 | 0.55–0.74 | 0.56–0.79 |
| Cache hit rate | 0.53–0.76 | 0.58–0.80 | 0.41–0.77 | 0.56–0.75 | 0.42–0.78 |
| Admission queue-req | 0.09–0.56 | 0.10–0.92 | 0.11–0.54 | 0.17–0.58 | 0.14–0.71 |
| Queue-wait mean/rank (rank max vs min) | r0 9.6 s / r7 2.4 s | r7 10.5 s / r6 3.3 s | r5 8.7 s / r0 2.5 s | r1 11.2 s / r2 4.9 s | r2 13.0 s / r1 2.8 s |
| TTFT mean/rank (rank max, p99 của nó vs rank min) | r0 11.2 s (p99 119) / r3 4.0 s | r7 14.7 s (p99 213) / r3 5.2 s | r5 11.6 s (p99 119) / r0 4.9 s | r1 15.7 s (p99 267) / r2 7.3 s | r2 16.5 s (p99 151) / r1 5.3 s |
| Request count per rank (min–max) | 72–221 | 70–270 | 77–279 | 79–226 | 113–217 |
| #rank chạm KV 1.00 | 4 | 0 | 3 | 0 | 0 |
| #retract | 3 | 0 | 2 | 0 | 0 |

- **Router overhead - không đáng kể**: 1172 dec/4.8 ms, 1216 dec/12.1 ms, 1221 dec/9.6 ms, 1206 dec/7.4 ms, 1188 dec/13.3 ms - quá nhỏ so TTFT giây.
- **Công thức cost - có tín hiệu nhưng chưa đủ**: corr(logit, TTFT) = +0.40, +0.26, +0.52, +0.37, +0.42; corr(logit, queue_wait) = +0.32, +0.08, +0.18, +0.15, +0.19. Cost cao đoán đúng hướng chờ lâu hơn ở mọi run, nhưng corr(logit, input_len) chỉ +0.18–0.32 (tín hiệu gần như không mang thông tin kích thước request — kiểm chứng chi tiết ở §4) nên hot-rank vẫn bị dồn.

## 3. Focus 1 - scheduling policy dfs-weight dẫn tới batch mỏng và đói prefill

- Batch decode mỏng mọi run (running-req mean ≤ 4.68/rank).
- Backlog prefill duy trì mọi run (đỉnh per-minute 73,559–132,265 token).
- Throughput chênh ≤ 2%, TTFT p99 chênh ≤ 32%.

**Nguyên nhân**:

- Cặp `--schedule-policy=dfs-weight` + `--enable-prefill-delayer` bảo vệ
  request đang decode (prefill chen ngang làm ITL giật) → scheduler hoãn
  prefill mới → backlog `#pending-token` duy trì suốt benchmark  dù admission queue gần rỗng (queue-req
  0.09–0.92/rank). TTFT tail 75–99 s đến từ backlog này, không phải cạn tài nguyên - GPU
  vẫn 77–97%.
- Batch decode mỏng 2.31–4.68 req/rank
  (dù trần `--max-running-requests=256`): prefill bị hoãn nên không có việc dồn vào batch.
- **Tóm lại:** scheduling policy đánh đổi TTFT lấy ITL ổn định. Ở CCU 64 nó giữ batch mỏng kéo dài suốt window nên là bottleneck là hành vi cố ý của policy.

**Verdict:** Consistent giữa 5 run

## 4. Focus 2 - routing config gây mất cân bằng về workload / KV per rank

**Mô tả:** Router của Dynamo có công thức:
`logit = active_decode_blocks + 16 × adjusted_prefill_blocks − cache_credits`

Logit **chỉ phản ánh trạng thái hiện tại của rank** và **không chứa chi phí của request đang được đặt**. Router chọn rank "đang nhẹ" mà không biết request sắp đặt vào nặng bao nhiêu (**size-blind placement**) → với workload không đồng nhất (input dài, ~96k cached token/request), token lệch 3.9–5.6× giữa rank mọi run. Lệch KV là hệ quả tích lũy, trễ thời gian của lệch token đó, chỉ escalate thành KV-full + retract khi quỹ đạo KV của run chạm ngưỡng (2/5 run). Cache affinity là lệch **có chủ đích cộng thêm** (chỉ 9.4–11.8% decision), không phải nguyên nhân chính.

- Active prefill token spread ≥ 3.9× mọi run.
- KV usage spread p50 ≥ 0.51 mọi run.
- corr(logit, TTFT) cùng dấu dương mọi run: +0.40, +0.26, +0.52, +0.37, +0.42.


Ví dụ (run 4):

| Chỉ số (mean) | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | spread |
|---|---|---|---|---|---|---|---|---|---|
| Active prefill tokens `worker_active_prefill_tokens` | 15.7k | 10.3k | 11.9k | 11.8k | 5.3k | 9.4k | 20.2k | 21.0k | **3.9×** |
| Active decode blocks | 6.1k | 6.9k | 6.2k | 6.7k | 5.0k | 6.1k | 6.8k | 6.3k | 1.4× |
| KV token usage `full_token_usage` | 0.60 | 0.74 | 0.68 | 0.67 | 0.55 | 0.73 | 0.66 | 0.66 | 1.4× |
| Admission queue-req | 0.51 | 0.58 | 0.46 | 0.39 | 0.17 | 0.42 | 0.45 | 0.39 | — |
| Cache hit rate | 0.63 | 0.75 | 0.66 | 0.61 | 0.58 | 0.56 | 0.73 | 0.59 | — |

Weight `16 × prefill_blocks` áp cho active prefill blocks (state của rank) — vấn đề không phải weight sai mà là công thức **thiếu hẳn term chi phí của request đang đặt** (corr(logit, input_len) +0.18–0.32) → size-blind placement là cơ chế chính, r7 nhận 21.0k active prefill token mean (nhiều nhất run 4) mà logit vẫn thấy rank đó "nhẹ".

**Nguyên nhân**:

- Router (kv-mode) cố gắng giữ request cùng prefix về cùng rank (cache hit
  per rank chênh 0.41–0.80; nhưng affinity chỉ kích hoạt 9.4–11.8% decision,
  `host_pinned_blocks > 0`) và chọn rank theo **argmin logit** — tín hiệu chỉ là
  trạng thái rank tại thời điểm đó, không chứa tải của request đang đặt. Kết quả
  bề ngoài giống "cân bằng theo số request": mọi run, rank ít request nhất đều thuộc
  nhóm nhận input mean lớn nhất. Request count per rank chênh 1.9–3.9× - tải bị lệch.
- **Hệ quả**: Active prefill token lệch 3.9–5.6× →
  KV usage lệch p50 0.51–0.57 →
  hot-rank chờ lâu hơn hẳn (queue-wait/TTFT per rank), và khi áp lực
  đủ lớn thì escalate thành KV-full + retract. Lớp offload hicache host (128 GB/rank) cũng lệch theo:
  used 0.5M–1.6M / 0.6M–2.1M / 0.6M–1.7M / 0.6M–1.7M / 0.9M–1.7M token.
- Công thức tính cost của Dynamo phản ánh đúng hướng (corr dương với TTFT; cấu trúc state-only đã kiểm chứng corr +0.50–0.74 với gauge) nhưng
  thiếu term chi phí request đang đặt nên chưa đủ kéo request khỏi hot-rank. Do đó, giả thuyết "công thức cost chưa tối ưu" được ủng hộ một phần nhưng chưa thể khẳng định hoàn toàn, hướng sửa: thêm estimated prompt cost (nhân theo KV-room còn lại của từng rank) vào logit, hoặc cân bằng theo prefill token/KV thay
  vì trạng thái tức thời.

**Verdict:** Consistent giữa 5 run

## 5. Bottleneck khác ngoài phạm vi Scheduling (maybe future work?)

Các hiện tượng dưới đây có đo được trong 5 run nhưng **không thuộc** phạm vi scheduler/routing - ghi lại để theo dõi sau:


- **EAGLE accept len** 1.91–2.55
  (per-minute mean): vài phút dưới ngưỡng khỏe ≥2.0 - spec decode chưa là
  nghẽn chính (ITL ~38–43 ms ổn định) nhưng đáng theo dõi.
- **Thước đo còn hở**: `sglang:fwd_occupancy` thỉnh thoảng bị drop mẫu NaN
  trong export; client không report `usage.prompt_tokens_details.cached_tokens`
  (cần `--enable-cache-report` phía server); `sglang:prefill_delayer_outcomes`
  có đếm per outcome/rank nhưng chưa được phân tích chi tiết.
- **MoE expert imbalance**: EP=8 → mỗi bước forward, token mọi rank all-to-all tới đúng expert, nếu 1 expert nóng, rank chứa expert đó chậm → gate cả bước của cả 8 rank. Với workload AgentX (agent/tool-use) phân bố expert theo chủ đề rất dễ lệch. Trong stack hiện tại không có metric per-expert nào được register - chỉ có sglang:eplb_balancedness (metrics_collector.py:923, Summary per forward_mode) nhưng chỉ register khi bật EPLB và ở moe_ep_rank=0. 


## 6. Nguồn dữ liệu & quy ước cite (compact)

| Nguồn | Nội dung | Dùng ở |
|---|---|---|
| `analysis/sglang_rank_ts.csv` + `analysis/dynamo_ts.csv` | trích từ `server_metrics_export.jsonl` (8081 `sglang:*` per dp_rank; 8888 `dynamo_*`), benchmark window | §1–§2 per-rank mean/spread, KV, cache hit, queue |
| `engine_*.log` / `frontend_*.log` (symlink trong run dir) | batch scheduler per-minute, pending-token, boot lines, WARN retract; dòng `Selected worker router_mode="kv"` (logit, host_pinned_blocks, dp_rank) + `Expiring stale request`; join router↔engine theo timestamp ±2 s; join `schedule_eval.jsonl` (`input_len`, `matched_prefix_tokens`, `output_len`) theo arrival ±2 s | §1 pending/running, §2 queue-wait/TTFT per rank, corr(logit,·), §4 kiểm chứng router (size-blind, affinity) |
| `e2_metrics.json` | throughput/TTFT/E2E/TPOT/fairness client | §1 |
| `gpu_samples.csv` | nvidia-smi 1 s × 8 GPU | §1 GPU util |
| `profiling/pyspy_*.svg` | flame 25 Hz × 120 s × 8 scheduler (4/5 run; run 2 probes bị skip) | §3 cơ chế |
| `profile_export.jsonl` | `benchmark_phase` + `request_start_ns` → window; duration từ `e2_metrics.json` (`run.duration_s`) | mọi bảng |

Benchmark window mỗi run = record đầu phase `profiling` → +1800 s. Chi tiết cách tính từng số + cite file/dòng: focus report từng run - run 1
`docs/focus3320_routing_scheduling.md`, run 2–5
`docs/focus_c64_<stamp>_routing_scheduling.md`. Reproduce: `extract_metrics_timeseries.py <run_dir>`
(srun trên node) → `gen_focus_report.py <run_dir>` → `compare_focus_runs.py
<run_dir...>` 
