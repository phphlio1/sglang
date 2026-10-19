# Consistency 3 run — dfs-weight c64 (config prod như 3320)

Trả lời câu hỏi: bottleneck thấy ở baseline 3320 có lặp lại ở các replicate
(cùng config từng tham số, seed 42, cold radix cache mỗi run) hay không?

| Run | Stamp | Run dir |
|---|---|---|
| 1 (baseline 3320) | c64_20261014_112105 | `results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261014_112105` |
| 2 | c64_20261015_133031 | `results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261015_133031` |
| 3 | c64_20261015_144717 | `results/8x-h200/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261015_144717` |

## 1. Bảng so sánh (mọi số tính trong benchmark window, nguồn như focus report từng run)

| Chỉ số | Run 1: c64_20261014_112105 | Run 2: c64_20261015_133031 | Run 3: c64_20261015_144717 | Verdict |
|---|---|---|---|---|
| Request/output throughput (tok/s) | 522 | 532 | 531 | ✓ |
| TTFT mean (s) | 7.1 | 7.6 | 6.8 | ✓ |
| TTFT p99 (s) | 75.4 | 89.7 | 76.0 | ✓ |
| Running-req mean/rank (min–max) | 2.75–3.75 | 2.31–4.68 | 2.60–3.93 | ✓ |
| Pending backlog per-minute (min–max) | 597–132.265 | 5.443–99.600 | 1.886–73.559 | ✓ |
| Active prefill token spread (×) | 5.6 | 5.1 | 4.5 | ✓ |
| KV usage mean (min–max) | 0.58–0.82 | 0.47–0.78 | 0.57–0.76 | ✓ |
| KV usage spread p50 | 0.56 | 0.56 | 0.51 | ✓ |
| #rank chạm KV 1.00 | 4 | 0 | 3 |  |
| #retract | 3 | 0 | 2 |  |
| corr(logit, TTFT) | +0.40 | +0.26 | +0.52 | ✓ cùng dấu |
| Cache reuse engine (%) | 87.5 | 88.1 | 87.9 | ✓ |
| Cached tokens/request | 96111 | 96206 | 96302 | ✓ |
| EAGLE accept len (per-minute min–max) | 1.97–2.31 | 2.00–2.35 | 1.91–2.45 | ✓ |
| ITL mean (ms) | 42.0 | 37.7 | 42.8 | ✓ |
| GPU util window (min–max %) | 79–97 | 77–97 | 79–96 | ✓ |

## 2. Verdict Focus 1 — scheduling policy (batch mỏng + prefill đói): **CONSISTENT**

- Batch decode mỏng mọi run (running-req mean ≤ 4.68/rank).
- Backlog prefill duy trì mọi run (đỉnh per-minute 73,559–132,265 token).
- Throughput chênh ≤ 2%, TTFT p99 chênh ≤ 19%.

## 3. Verdict Focus 2 — routing mất cân bằng rank: **CONSISTENT**

- Active prefill token spread ≥ 4.5× mọi run.
- KV usage spread p50 ≥ 0.51 mọi run.
- corr(logit, TTFT) cùng dấu dương mọi run: +0.40, +0.26, +0.52.

## 4. Health-check: **CONSISTENT**

- Cache reuse ≥ 87.5% mọi run; accept len ≥ 1.91; retract ≤ 3 lần/run.

## 5. Kết luận

Verdict là heuristic (cùng hướng + cùng bậc độ lớn, ngưỡng ghi ở từng mục);
số liệu chính nằm ở bảng §1 và focus report từng run
(`docs/focus_<stamp>_routing_scheduling.md`).
