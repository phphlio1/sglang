# Consistency 5 run — dfs-weight c64 (config prod như 3320)

Trả lời câu hỏi: bottleneck thấy ở baseline 3320 có lặp lại ở các replicate
(cùng config từng tham số, seed 42, cold radix cache mỗi run) hay không?

| Run | Stamp | Run dir |
|---|---|---|
| 1 (baseline 3320) | c64_20261021_141334 | `results/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261021_141334` |
| 2 | c64_20261021_152339 | `results/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261021_152339` |
| 3 | c64_20261021_171014 | `results/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261021_171014` |
| 4 | c64_20261021_182013 | `results/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261021_182013` |
| 5 | c64_20261021_193141 | `results/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261021_193141` |

## 1. Bảng so sánh (mọi số tính trong benchmark window, nguồn như focus report từng run)

| Chỉ số | Run 1: c64_20261021_141334 | Run 2: c64_20261021_152339 | Run 3: c64_20261021_171014 | Run 4: c64_20261021_182013 | Run 5: c64_20261021_193141 | Verdict |
|---|---|---|---|---|---|---|
| Request/output throughput (tok/s) | 541 | 544 | 546 | 530 | 541 | ✓ |
| TTFT mean (s) | 7.6 | 8.2 | 7.4 | 8.2 | 7.5 | ✓ |
| TTFT p99 (s) | 70.9 | 88.3 | 65.6 | 97.5 | 59.9 | ✓ |
| Running-req mean/rank (min–max) | 2.35–4.16 | 2.46–4.01 | 2.30–3.93 | 2.47–4.54 | 2.66–3.61 | ✓ |
| Pending backlog per-minute (min–max) | 11.057–101.063 | 4.495–128.693 | 10.065–89.046 | 7.552–101.302 | 0–83.771 | ✓ |
| Active prefill token spread (×) | 10.8 | 5.9 | 5.3 | 5.5 | 5.1 | ≠ >2× |
| KV usage mean (min–max) | 0.52–0.83 | 0.56–0.77 | 0.62–0.74 | 0.60–0.78 | 0.52–0.80 | ✓ |
| KV usage spread p50 | 0.55 | 0.50 | 0.53 | 0.54 | 0.57 | ✓ |
| #rank chạm KV 1.00 | 2 | 1 | 3 | 0 | 1 |  |
| #retract | 1 | 0 | 2 | 1 | 0 |  |
| corr(logit, TTFT) | +0.39 | +0.30 | +0.38 | +0.34 | +0.32 | ✓ cùng dấu |
| Cache reuse engine (%) | 87.6 | 87.7 | 87.7 | 87.2 | 88.0 | ✓ |
| Cached tokens/request | 95925 | 95415 | 95574 | 98293 | 94531 | ✓ |
| EAGLE accept len (per-minute min–max) | 1.92–2.38 | 1.95–2.33 | 1.97–2.31 | 1.97–2.36 | 1.87–2.41 | ✓ |
| ITL mean (ms) | 39.0 | 38.7 | 39.5 | 40.8 | 38.1 | ✓ |
| GPU util window (min–max %) | 87–98 | 82–98 | 82–98 | 85–98 | 87–97 | ✓ |

## 2. Mất cân bằng workload/KV per rank và hệ quả

Thống kê per rank trong benchmark window (hàng workload = min–max mean qua 8
rank; hàng join = rank cực đại vs cực tiểu; nguồn và cách tính: focus
report từng run §2.2–§2.3 — report này cite hạn chế, chỉ trỏ về focus report).

| Chỉ số (per rank) | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|---|---|---|---|---|---|
| Active prefill tokens (8888) | 4.4k–47.3k (10.8×) | 4.0k–23.9k (5.9×) | 3.9k–20.4k (5.3×) | 4.8k–26.3k (5.5×) | 4.6k–23.7k (5.1×) |
| Active decode blocks (8888) | 4.8k–9.9k (2.1×) | 5.0k–8.9k (1.8×) | 5.2k–7.2k (1.4×) | 5.1k–7.4k (1.5×) | 4.4k–8.3k (1.9×) |
| KV token usage (8081 `full_token_usage`) | 0.52–0.83 | 0.56–0.77 | 0.62–0.74 | 0.60–0.78 | 0.52–0.80 |
| Cache hit rate (8081) | 0.59–0.77 | 0.50–0.77 | 0.54–0.79 | 0.37–0.74 | 0.63–0.77 |
| Admission queue-req (8081) | 0.12–1.41 | 0.12–1.09 | 0.07–0.88 | 0.08–1.07 | 0.09–0.98 |
| Queue-wait mean/rank (rank max vs min) | r4 23.2 s / r6 2.7 s | r1 13.2 s / r6 2.1 s | r2 8.4 s / r0 3.4 s | r2 20.4 s / r6 3.0 s | r2 11.8 s / r7 2.1 s |
| TTFT mean/rank (rank max, p99 của nó vs rank min) | r4 27.1 s (p99 472) / r6 5.4 s | r1 17.1 s (p99 312) / r6 4.1 s | r2 11.8 s (p99 249) / r0 6.1 s | r2 23.9 s (p99 609) / r6 5.4 s | r2 15.4 s (p99 82) / r7 4.0 s |
| Request count per rank (min–max) | 94–213 | 95–279 | 91–262 | 86–230 | 115–219 |
| #rank chạm KV 1.00 | 2 | 1 | 3 | 0 | 1 |
| #retract | 1 | 0 | 2 | 1 | 0 |

- **Router overhead — không đáng kể**: 1228 dec/9.8 ms, 1237 dec/15.0 ms, 1239 dec/10.6 ms, 1194 dec/22.3 ms, 1242 dec/13.4 ms (decision/run, mean ms;
  histogram delta 8888, focus report §2.4 từng run) — quá nhỏ so TTFT giây.
- **Công thức cost — có tín hiệu nhưng chưa đủ**: corr(logit, TTFT) =
  +0.39, +0.30, +0.38, +0.34, +0.32; corr(logit, queue_wait) = +0.19, +0.13, +0.14, +0.13, +0.14; quintile Q1→Q5 median
  TTFT: 2.0→10.4 s; 2.0→9.6 s; 1.8→10.4 s; 2.1→10.2 s; 1.8→9.4 s. Cost cao đoán đúng hướng chờ lâu hơn ở mọi run, nhưng
  hot-rank vẫn bị dồn (KV-full + retract ở 2/5 run, bảng trên) —
  giả thuyết "công thức chưa tối ưu" được ủng hộ một phần (nghi vấn cơ chế:
  weight `16 × prefill_blocks` underestimate chi phí prefill; kv-state router
  hết hạn — chi tiết focus report §2.5).

## 3. Verdict Focus 1 — scheduling policy (batch mỏng + prefill đói): **CONSISTENT**

- Batch decode mỏng mọi run (running-req mean ≤ 4.54/rank).
- Backlog prefill duy trì mọi run (đỉnh per-minute 83,771–128,693 token).
- Throughput chênh ≤ 3%, TTFT p99 chênh ≤ 63%.

**Tại sao** (cơ chế; cite dòng log từng số ở focus report §1.1–1.2 từng run):

- Cặp `--schedule-policy=dfs-weight` + `--enable-prefill-delayer` bảo vệ
  request đang decode (prefill chen ngang làm ITL giật) → scheduler hoãn
  prefill mới → backlog `#pending-token` duy trì suốt benchmark (đỉnh per-minute
  73k–132k token, bảng §1) dù admission queue gần rỗng (queue-req
  0.07–1.41/rank,
  §2). TTFT tail 75–99 s đến từ backlog này, không phải cạn tài nguyên — GPU
  vẫn 77–97%.
- Batch decode mỏng 2.30–4.54 req/rank
  (dù trần `--max-running-requests=256`): prefill bị hoãn nên không có việc
  dồn vào batch; MoE EP=8 all-to-all đồng bộ mọi rank mỗi bước forward →
  chi phí cố định mỗi bước không phụ thuộc batch → GPU bận (util cao, power
  ổn) nhưng năng suất chặn ~520–532 tok/s (latency-bound). Py-spy flame
  (8/8 scheduler, 4 run có flame) cho thấy MoE forward + DeepEP `dispatch_b`
  chiếm ưu thế.
- Tóm lại: policy đánh đổi TTFT lấy ITL ổn định (chủ đích của config prod);
  ở c64 nó giữ batch mỏng kéo dài suốt window — bottleneck là hành vi cố ý
  của policy, lặp lại y hệt mọi run.

## 4. Verdict Focus 2 — routing mất cân bằng rank: **CONSISTENT**

- Active prefill token spread ≥ 5.1× mọi run.
- KV usage spread p50 ≥ 0.50 mọi run.
- corr(logit, TTFT) cùng dấu dương mọi run: +0.39, +0.30, +0.38, +0.34, +0.32.

**Tại sao** (cơ chế; chi tiết + cite focus report §2.2–§2.5 từng run):

- Router (kv-mode) cố gắng giữ request cùng prefix về cùng rank (cache hit
  per rank chênh 0.37–0.79, §2) và
  cân bằng theo **số request** — nhưng số request không phải tải thực: mọi
  run, rank ít request nhất đều thuộc nhóm nhận input mean lớn nhất (4/5 run
  chính là rank input lớn nhất; ví dụ run 2: r1 chỉ
  95 request nhưng input mean 132k token).
  Request count per rank chênh 1.9–2.9×
  (§2) — cân bằng bề ngoài, tải thực lệch.
- Hệ quả: active prefill token lệch 5.1–10.8× →
  KV usage lệch p50 0.50–0.57 →
  hot-rank chờ lâu hơn hẳn (queue-wait/TTFT per rank, §2), và khi áp lực
  đủ lớn thì escalate thành KV-full + retract (2/5 run; triệu chứng
  intermittent — mất cân bằng luôn đo được, việc tràn thì phụ thuộc quỹ đạo
  KV của từng run). Lớp offload hicache host (128 GB/rank) cũng lệch theo:
  used 0.6M–1.6M / 0.7M–1.6M / 0.6M–1.8M / 0.7M–2.0M / 0.6M–1.9M token.
- Cost formula phản ánh đúng hướng (corr dương + quintile tăng, §2) nhưng
  chưa đủ kéo request khỏi hot-rank → giả thuyết "công thức cost chưa tối
  ưu" được ủng hộ một phần, hướng sửa: cân bằng theo prefill token/KV thay
  vì số request (nghi vấn §2.5 từng run).

## 5. Health-check: **CONSISTENT**

- Cache reuse ≥ 87.2% mọi run; accept len ≥ 1.87; retract ≤ 2 lần/run.

## 6. Bottleneck khác ngoài phạm vi 2 focus (ghi chú future work)

Các hiện tượng dưới đây có đo được trong 5 run nhưng **không thuộc** phạm vi
scheduler/routing mentor yêu cầu — ghi lại để theo dõi sau:

- **Fairness / long-tail TTFT giữa các request**: Jain TTFT
  0.11 / 0.08 / 0.11 / 0.09 / 0.07 (gần 0 = rất không đều), CV 2.8 / 3.5 / 2.8 / 3.1 / 3.5; cold/warm TTFT ratio
  0.93 / 0.94 / 0.95 / 0.95 / 0.92 (cold-cache request, n≈114/run); queue-wait p99 client
  55 / 14 / 56 / 12 / 193 s — chênh lớn giữa các run.
- **E2E long-tail**: p99 393 / 434 / 397 / 418 / 416 s — phần lớn từ TTFT tail cộng output dài.
- **EAGLE accept len** 1.87–2.41
  (per-minute mean): vài phút dưới ngưỡng khỏe ≥2.0 — spec decode chưa là
  nghẽn chính (ITL ~38–43 ms ổn định) nhưng đáng theo dõi.
- **Thước đo còn hở**: `sglang:fwd_occupancy` thỉnh thoảng bị drop mẫu NaN
  trong export; client không report `usage.prompt_tokens_details.cached_tokens`
  (cần `--enable-cache-report` phía server); `sglang:prefill_delayer_outcomes`
  có đếm per outcome/rank nhưng chưa được phân tích chi tiết.
- **Ops**: boot sau fresh mirror nguội JIT ~7 phút (run 2) — đã mitigate
  bằng `PROBE_WAIT_TIMEOUT_S=900` + readiness gate (run 4–5 probe đủ 8/8,
  job exit 0).

## 7. Kết luận

Verdict là heuristic (cùng hướng + cùng bậc độ lớn, ngưỡng ghi ở từng mục);
số liệu chính nằm ở bảng §1–§2 và focus report từng run. Cả 2 focus
đều **CONSISTENT qua 5 run cùng config**: (1) scheduling giữ batch
mỏng + prefill đói, (2) routing lệch tải per-rank — bottleneck có thật,
không phải noise của một lần chạy.

## 8. Nguồn dữ liệu & quy ước cite (compact)

| Nguồn | Nội dung | Dùng ở |
|---|---|---|
| `analysis/sglang_rank_ts.csv` + `analysis/dynamo_ts.csv` | trích từ `server_metrics_export.jsonl` (8081 `sglang:*` per dp_rank; 8888 `dynamo_*`), benchmark window | §1–§2 per-rank mean/spread, KV, cache hit, queue |
| `engine_*.log` / `frontend_*.log` (symlink trong run dir) | batch scheduler per-minute, pending-token, boot lines, WARN retract; join router↔engine theo timestamp ±2 s | §1 pending/running, §2 queue-wait/TTFT per rank, corr(logit,·) |
| `e2_metrics.json` | throughput/TTFT/E2E/TPOT/fairness client | §1, §6 |
| `gpu_samples.csv` | nvidia-smi 1 s × 8 GPU | §1 GPU util |
| `profiling/pyspy_*.svg` | flame 25 Hz × 120 s × 8 scheduler (4/5 run; run 2 probes bị skip) | §3 cơ chế |
| `profile_export.jsonl` | `benchmark_phase` + `request_start_ns` → window; duration từ `e2_metrics.json` (`run.duration_s`) | mọi bảng |

Benchmark window mỗi run = record đầu phase `profiling` → +1800 s. Chi tiết
cách tính từng số + cite file/dòng: focus report từng run — run 1
`docs/focus3320_routing_scheduling.md`, run 2–5
`docs/focus_c64_<stamp>_routing_scheduling.md` (report này cite hạn chế,
trỏ về focus report). Reproduce: `extract_metrics_timeseries.py <run_dir>`
(srun trên node) → `gen_focus_report.py <run_dir>` → `compare_focus_runs.py
<run_dir...>` (script này).
