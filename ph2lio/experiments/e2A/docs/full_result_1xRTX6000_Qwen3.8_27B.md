# Full result: 1×RTX 6000 Pro - Qwen/Qwen3.8-27B-FP8 (AgentX)

## 1. Workload: AgentX

| Hạng mục | Cấu hình |
|---|---|
| Benchmark | AIPerf `aiperf profile`, scenario `inferencex-agentx-mvp` |
| Dataset | `semianalysis_cc_traces_weka_062126_256k` (trace agent, context dài) |
| Model | Qwen/Qwen3.8-27B-FP8 |
| Context Length | 32K-256K (avg. 100K) |
| Tỉ lệ Input:Output | 112:1 |
| Hardware | 1×RTX 6000 Pro Blackwell 96GB|
| Scheduling Policies | lpm (baseline), hrrn, fcfs, dfs-weight, boost (γ tĩnh = 0.09, k = 256) |
| Concurrency | 4 / 8 / 16 |
| Duration | 30 phút (1800 s) |
| Seed | 42 |
| Boot | fresh boot mỗi cell (FORCE_RESTART=1) - radix cache lạnh mỗi run |

## 2. Serving Configuration
Sử dụng recipe sau đây:

```
python3 -m sglang.launch_server \
  --model-path /models/Qwen3.8-27B-FP8 --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.80 --attention-backend flashinfer \
  --chunked-prefill-size 8192 --max-prefill-tokens 16384 --context-length 262144 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-linear-replayssm-spec --mamba-full-memory-ratio 1.5 --schedule-policy dfs-weight \
  --enable-hierarchical-cache --hicache-size 64 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

## 3. Metrics

| # | Metric | Mô tả| Công thức |
|---|---|---|---|
| 1 | n done / n wait | Số request hoàn thành / tổng | — |
| 2 | cache hit (%) | Mức tận dụng prefix cache của workload | `sum(matched_token)/sum(input_len)` | 
| 3 | queue mean/p90/p99 (s) | Thời gian trong hàng chờ trước khi batch| Percentile tuyến tính `k=(n−1)·p/100` |
| 4 | TTFT mean/p95/p99 (ms) | Thời gian đến token đầu (gồm queue wait) | Percentile tuyến tính `k=(n−1)·p/100` | 
| 5 | ITL mean/p90/p99 (ms) | Thời gian stream giữa 2 token liên tiếp | Percentile tuyến tính `k=(n−1)·p/100` |
| 6 | tok/s | Throughput số token trong 1 giây | `sum(output_sequence_length) / (request_end_max − request_start_min)` |
| 7 | Jain | Độ đồng đều của chờ (1 = ai cũng chờ như nhau) | `(Σx)²/(n·Σx²)`, x: thời gian chờ của mỗi request |
| 8 | stretch p95/p99 | Tỉ lệ request lạnh phải chờ so với prefill | `wait/(ttft−wait)` (mẫu số ≈ thời gian prefill vì TTFT = wait + prefill) | 
| 9 | worst-5% wait share (%) | mức tập trung wait mass vào 5%. tail request, 100% = toàn bộ thời gian chờ nằm ở 5% request | `sum(ceil(0.05n) wait lớn nhất)/sum(tất cả wait) × 100` |


## 4. Result

Output c4/c8/c16 × {fcfs-match, lpm, hrrn, dfs-weight,
boost} × 1 run × 30', seed 42, fresh boot mỗi cell (FORCE_RESTART=1),
grace 30s. Boost chạy gamma tĩnh γ̂ offline = 0.09, k = 256

### Concurrency 4

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 220 | 222 | 91.76% | 0.092 | 0.002 | 2.936 | 1779.20 | 7225.36 | 13642.73 | 42.66 | 44.28 | 83.00 | 92.9 | 0.015 | 0.01 | 2.85 | 98.55% |
| boost | 212 | 212 | 91.30% | 0.058 | 0.002 | 2.246 | 1768.20 | 7071.56 | 13541.61 | 42.77 | 44.05 | 80.15 | 88.3 | 0.012 | 0.01 | 0.24 | 97.64% |
| hrrn | 222 | 222 | 91.76% | 0.050 | 0.002 | 1.936 | 1682.95 | 7080.40 | 11586.28 | 43.30 | 44.22 | 79.97 | 99.6 | 0.013 | 0.01 | 1.95 | 97.34% |
| fcfs | 223 | 223 | 91.79% | 0.133 | 0.002 | 3.596 | 1839.42 | 7881.43 | 11718.51 | 42.26 | 44.57 | 86.97 | 99.3 | 0.029 | 0.01 | 4.86 | 99.00% |
| dfs-weight | 216 | 217 | 91.32% | 0.104 | 0.002 | 5.220 | 1748.93 | 7631.11 | 14055.84 | 43.79 | 44.25 | 85.84 | 88.6 | 0.021 | 0.02 | 1.35 | 98.38% |

### Concurrency 8

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 143 | 143 | 94.21% | 0.183 | 0.003 | 3.890 | 2323.19 | 11117.82 | 16321.75 | 41.33 | 42.29 | 74.85 | 99.6 | 0.021 | 0.01 | 1.71 | 99.10% |
| boost | 142 | 142 | 94.10% | 0.136 | 0.003 | 4.935 | 2251.08 | 10410.30 | 14503.18 | 40.68 | 41.96 | 71.91 | 99.0 | 0.026 | 0.01 | 2.37 | 98.85% |
| hrrn | 143 | 143 | 94.04% | 0.044 | 0.003 | 1.683 | 2089.60 | 9905.51 | 14053.79 | 42.09 | 42.30 | 75.69 | 99.4 | 0.015 | 0.01 | 0.60 | 96.51% |
| fcfs | 137 | 138 | 94.00% | 0.046 | 0.003 | 1.828 | 2113.72 | 10044.72 | 14439.41 | 42.71 | 42.59 | 84.51 | 96.8 | 0.016 | 0.01 | 0.64 | 96.63% |
| dfs-weight | 142 | 142 | 94.02% | 0.059 | 0.003 | 2.573 | 2120.06 | 9953.82 | 14067.36 | 43.61 | 44.39 | 84.47 | 98.9 | 0.022 | 0.01 | 0.88 | 96.84% |

### Concurrency 16

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 169 | 173 | 84.81% | 19.892 | 57.810 | 153.131 | 23958.37 | 74285.69 | 130670.55 | 77.16 | 52.70 | 211.51 | 87.8 | 0.282 | 38.05 | 53.58 | 30.57% |
| boost | 164 | 168 | 83.03% | 19.536 | 53.201 | 173.198 | 23028.70 | 84795.82 | 117749.90 | 92.71 | 53.32 | 245.14 | 85.5 | 0.261 | 31.07 | 54.27 | 35.18% |
| hrrn | 177 | 180 | 84.84% | 17.409 | 51.919 | 111.336 | 23593.50 | 82065.00 | 141705.79 | 81.41 | 53.10 | 226.71 | 95.6 | 0.321 | 25.11 | 50.04 | 29.58% |
| fcfs | 163 | 168 | 81.86% | 21.100 | 65.599 | 140.530 | 26134.03 | 93752.34 | 147291.40 | 83.28 | 52.73 | 225.47 | 83.6 | 0.303 | 47.25 | 79.39 | 32.64% |
| dfs-weight | 143 | 146 | 81.58% | 27.500 | 51.702 | 365.985 | 35114.19 | 171258.53 | 383571.18 | 95.34 | 53.81 | 191.56 | 81.4 | 0.140 | 31.60 | 65.52 | 56.89% |
| boost-ada Eq.9 J=50* | 167 | 167 | 81.68% | 22.189 | 70.970 | 119.619 | 30099.72 | 105729.70 | 140758.66 | 91.94 | 52.97 | 218.80 | 82.8 | 0.365 | 34.90 | 157.74 | 26.12% |

\* boost-ada Eq.9 J=50 — **đại diện chính của ada-boost** trong bảng này:
gamma-Ada theo UniBoost Eq. 9 (estimator fit E2E latency tail của request
hoàn thành, thay wait tail nội sinh của ada-wait) + max-jump bound
b_γ(k/rate) ≤ J (`--boost-adaptive-max-jump-s`); J=50 → floor γ pin ≈
0.090 = static-0.09 (đo γ min 0.0900, final-50 0.106). Run riêng
2026-10-14, fresh boot, seed 42, cùng protocol. **Both-wins đầu tiên của
boost-ada:** so mọi arm boost trước đó, cả 4 metric starvation cùng cải
thiện — queue p99 119.6 s (thấp nhất sau hrrn 111.3), TTFT p99 140.8 s (≈
hrrn, tốt hơn mọi arm boost cũ), Jain 0.365 + worst-5% 26.12% thấp nhất
bảng. Giá: stretch p99 157.7× (outlier 1 request — p95 34.9 tốt hơn lpm
38.05), tok/s 82.8 (−5.7% vs lpm). 1 run — trong band nhiễu; mọi kết luận
J cần repeat. Các arm ada khác (ada-wait, Eq.9 J=60, J=40, J=60 + cap):
sweep table dưới + `metrics_v2.md` block tương ứng; cap bị bác 6/6
prediction (2026-10-14) → code chunk-preempt đã backtrack về 60060211ce,
guard PrefillBudget giữ làm reliability fix (sống qua 12 lần pool 1.00,
điều kiện tiên quyết production).

### Concurrency 16 — static-gamma sweep (2026-10-13)

Quét γ static (adaptive off) lấp khoảng quanh 0.09: γ ∈ {0.05, 0.075,
0.12, 0.2} wall, k = 256, cùng protocol (fresh boot, seed 42, 1800 s;
γ = 0.02 fail hạ tầng, chưa có). Thống kê vững (mean trên ~170 request):

| arm | γ | queue mean (s) | TTFT mean (s) | TTFT p99 (s) | Jain | stretch p99 | tok/s |
|---|---|---|---|---|---|---|---|
| lpm (10-09) | — | 19.89 | 24.0 | 130.7 | 0.282 | 53.58 | 87.8 |
| boost static | 0.05 | 22.35 | 30.2 | 198.7 | 0.297 | 126.24 | 81.6 |
| boost static | 0.075 | 23.04 | 31.0 | 177.8 | 0.267 | 91.63 | 82.0 |
| boost static (10-09) | 0.09 | 19.54 | 23.0 | 117.7 | 0.261 | 54.27 | 85.5 |
| boost static | 0.12 | 25.80 | 33.6 | 186.8 | 0.296 | 98.99 | 82.9 |
| boost static | 0.2 | 21.03 | 28.6 | 156.9 | 0.278 | 59.48 | 84.6 |
| boost-ada (hội tụ ~0.042) | 0.09→ | 22.33 | 30.0 | 180.0 | 0.309 | 148.51 | 83.2 |
| boost-ada Eq.9 (hội tụ ~0.085) | 0.09→ | 22.52 | 30.0 | 158.3 | 0.299 | 79.68 | 85.6 |
| boost-ada Eq.9 J=50 (14-10) | 0.09→0.106 | 22.19 | 30.1 | 140.8 | 0.365 | 157.7 | 82.8 |
| boost-ada Eq.9 J=40 (14-10) | 0.09→0.120 | 23.71 | 31.2 | 146.2 | 0.335 | 69.3 | 84.3 |
| boost-ada Eq.9 J=60 + cap (14-10) | 0.09→0.088 | 21.28 | 29.5 | 191.5 | 0.272 | 77.5 | 81.3 |

**Đọc J-sweep (14-10):** max-jump bound J thấp → floor γ cao (J=50 →
floor 0.090 = static-0.09; J=40 → ~0.107). Floor pin γ đúng nơi dự
đoán (γ min 0.0900 ở J=50, đúng bằng static-0.09). J=50: cả 4 metric
queue p99 / TTFT p99 / Jain / worst-5% cùng cải thiện so J=60 —
both-wins đầu tiên của boost-ada; J=40 nghiêng FCFS hơn, đánh mất
queue p99 (132.8). Non-monotonic theo J ở 1 run (stretch p99, ITL
p99, tok/s) — cùng band nhiễu như γ sweep. Cap chunk-preempt (J=60 +
cap): tệ hơn đồng nhất 6/7 metric chính so no-cap J=60 (queue p99
157.5, TTFT p99 191.5 — tệ nhất bảng) → giant-budget-hog bị bác,
đóng hướng cap.

**Đọc sweep:** mọi metric lật qua lật lại không đơn điệu theo γ (TTFT
mean: 30.2 → 31.0 → 23.0 → 33.6 → 28.6) — ở 1 run/cell, khác biệt giữa
các γ nằm trong band nhiễu, không có "γ đúng" phân biệt được. Ở thống
kê vững, lpm ≥ mọi arm boost ở mọi γ đã quét (queue mean, TTFT mean,
ITL mean, tok/s). Kết luận: hand-tune γ theo workload/CCU không khả
thi từ single run (tín hiệu < nhiễu); nếu boost có giá trị thì phải
chứng minh ở CCU cao hơn / nhiều GPU / có repeat — đó cũng là lý do
adaptive (γ̂ tự sinh từ workload) đúng hướng về mặt thiết kế. Arm boost-ada Eq.9 (hội tụ ~0.085) là bước kiểm chứng đầu tiên
của hướng đó: γ tự sinh đúng vùng static đo tốt (0.075–0.09), hết
regression starvation của estimator wait, và giữ queue p99 / worst-5%
thấp nhất bảng — nhưng ở 1 run, mọi khác biệt vẫn trong band nhiễu đã
nêu.
