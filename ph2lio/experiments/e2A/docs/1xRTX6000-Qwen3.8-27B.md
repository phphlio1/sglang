# 1×RTX 6000 Pro - Qwen/Qwen3.8-27B-FP8 Report

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

## 3b. Quy ước "% so với baseline"

Ba bảng dưới in `giá_trị (±x.x%)` trong mỗi cell (trừ dòng baseline và
hai cột population và cột cache hit): % tốt hơn so với **lpm (baseline)** ở cùng
concurrency, dấu chuẩn hoá về phía "tốt hơn":

- metric càng-thấp-càng-tốt (queue, TTFT, ITL, stretch, worst-5%):
  `% = (base − arm) / base × 100`;
- metric càng-cao-càng-tốt (tok/s, Jain):
  `% = (arm − base) / base × 100`;
- **+ = tốt hơn baseline, − = tệ hơn baseline**;
- cột cache hit in giá trị không kèm % — tính chất workload, không phải hiệu ứng policy.

## 4. Result 

### 4a) Concurrency 16

| Policy | n done | n wait | cache hit (%) | queue p99 (s) | TTFT p99 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 169 | 173 | 84.81 | 153.131 | 130671 | 211.51 | 87.8 | 0.282 | 53.58 | 30.57 |
| boost | 164 | 168 | 83.03 | 173.198 (-13.1%) | 117750 (+9.9%) | 245.14 (-15.9%) | 85.5 (-2.6%) | 0.261 (-7.4%) | 54.27 (-1.3%) | 35.18 (-15.1%) |
| hrrn | 177 | 180 | 84.84 | 111.336 (+27.3%) | 141706 (-8.4%) | 226.71 (-7.2%) | 95.6 (+8.9%) | 0.321 (+13.8%) | 50.04 (+6.6%) | 29.58 (+3.3%) |
| fcfs | 163 | 168 | 81.86 | 140.530 (+8.2%) | 147291 (-12.7%) | 225.47 (-6.6%) | 83.6 (-4.8%) | 0.303 (+7.5%) | 79.39 (-48.2%) | 32.64 (-6.8%) |
| dfs-weight | 143 | 146 | 81.58 | 365.985 (-139.0%) | 383571 (-193.5%) | 191.56 (+9.4%) | 81.4 (-7.2%) | 0.140 (-50.3%) | 65.52 (-22.3%) | 56.89 (-86.1%) |
| boost-ada Eq.9 J=50* | 167 | 167 | 81.68 | 119.619 (+21.9%) | 140759 (-7.7%) | 218.80 (-3.4%) | 82.8 (-5.7%) | 0.365 (+29.4%) | 157.74 (-194.4%) | 26.12 (+14.6%) |

Regime đổi hẳn so với c4/c8: queue p99 111–366 s, TTFT p99 118–384 s,
worst-5% 29.6–56.9% (wait trải đều hơn), stretch p99 50–66× (request
lạnh chờ gấp ~50–66 lần prefill của chính mình). 16 request đồng thời
trên 1 GPU → hàng luôn có người → policy thật sự có quyền chọn →
khác biệt arm bung ra.

- **hrrn — tốt nhất nhóm xếp hàng + throughput:** queue p99 +27.3%
  (111.3 s), Jain 0.321 (+13.8%), stretch p99 50.04 (+6.6%), worst-5%
  29.58% (+3.3%), tok/s 95.6 (+8.9%), n done 177 — nhiều nhất bảng.
  Đổi lại TTFT p99 141.7 s (−8.4%) tệ hơn lpm — giá điển hình của ưu
  tiên response-ratio: giant bị hoãn, đuôi TTFT của chính giant dài
  ra. hrrn đổi starvation (chờ trong hàng) lấy đuôi TTFT — đúng trục
  production đang muốn giải.
- **boost (γ = 0.09 tĩnh) — trade TTFT đuôi lấy queue đuôi:** TTFT p99
  117.7 s thấp nhất bảng (+9.9%), nhưng queue p99 173.2 s (−13.1%),
  Jain 0.261 (−7.4%), worst-5% 35.2% (−15.1%) — cứu TTFT của vài
  request bằng cách giữ số còn lại trong hàng lâu hơn. Câu hỏi
  tĩnh-vs-adaptive chưa đóng được: arm adaptive phải re-run cùng
  config 10-09 mới so formally được.
- **fcfs — trung bình, nghiêng xấu:** TTFT p99 147.3 s (−12.7%),
  queue p99 chỉ +8.2%, Jain 0.303 (+7.5%) — "công bằng theo thứ tự
  tới" nhưng không giải được đuôi.
- **dfs-weight — tệ nhất gần mọi trục:** queue p99 366.0 s (−139.0%),
  TTFT p99 383.6 s (−193.5%), Jain 0.140 (−50.3%), worst-5% 56.9%
  (−86.1%), tok/s 81.4 (−7.2%), n done 143 — ít nhất. Weight ordering
  gạt nhầm request lớn trong workload AgentX — đảo ngược hẳn intent
  chống starvation. **Cờ đỏ:** đây là arm production chọn thay lpm
  (xem §8).

\* boost-ada Eq.9 J=50 — đại diện chính của ada-boost: gamma-Ada theo
UniBoost Eq. 9 (estimator fit E2E latency tail của request hoàn thành,
thay wait tail nội sinh) + max-jump bound b_γ(k/rate) ≤ J
(`--boost-adaptive-max-jump-s`); J=50 → floor γ pin ≈ 0.090 = static-0.09
(đo γ min 0.0900, final-50 0.106). Run riêng 2026-10-14, fresh boot, seed
42, cùng protocol. Các arm ada khác — ada-wait (2991704690), Eq.9 J=60
(13-10), J=40, và J=60 + chunk-preempt cap (bác 6/6 prediction, code đã
backtrack về 60060211ce; guard PrefillBudget giữ làm reliability fix):
`metrics_v2.md` block tương ứng.

- **boost-ada Eq.9 J=50 — điểm both-wins đầu tiên của boost-ada:** so
  arm Eq.9 (J=60), cả 4 metric cùng cải thiện: queue p99 126.4 →
  **119.6 s** (+21.9% vs lpm, chỉ sau hrrn 111.3), TTFT p99 158.3 →
  **140.8 s** (−7.7% vs lpm, ≈ hrrn 141.7 — tốt hơn mọi arm boost trước
  đó), Jain 0.299 → **0.365** (+29.4%, cao nhất bảng, > hrrn 0.321),
  worst-5% 29.27 → **26.12%** (+14.6%, thấp nhất bảng). Floor pin γ =
  static-0.09 đúng như thiết kế; J=40 (floor ~0.107) nghiêng FCFS hơn,
  đánh mất queue p99 (132.8) — J=50 là mép giữa của dãy. Giá: stretch
  p99 157.7× (outlier 1 request — stretch p95 34.9 tốt hơn lpm 38.05),
  TTFT mean 30.1 s, tok/s 82.8 (−5.7%). 1 run — trong band nhiễu §7.

- **boost-ada Eq.9 — adaptive đúng paper + jump bound:** γ tự hội tụ
  ~0.085, đúng vùng static đo tốt (0.075–0.09) — hết feedback loop của
  estimator wait (ada cũ tụt 0.042 → stretch p99 148.5×, TTFT p99 180 s;
  Eq.9: 79.7×, 158 s). Queue p99 126.4 s (+17.5%) và worst-5% 29.27%
  (+4.3%) tốt nhất bảng, Jain 0.299 (+6.0%), ITL p99 +7.4%, n done 171,
  tok/s 85.6. Giá: TTFT p99 −21.2% và stretch p99 −48.7% so lpm — vẫn
  tệ hơn boost tĩnh (117.7 s / 54.3×): jump ~54 s được phép hiện ra ở
  đuôi TTFT của request lạnh. 1 run — trong band nhiễu §7.

### 4b) Concurrency 4 - Dưới tải - Không có starvation

| Policy | n done | n wait | cache hit (%) | queue p99 (s) | TTFT p99 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 220 | 222 | 91.76 | 2.936 | 13643 | 83.00 | 92.9 | 0.015 | 2.85 | 98.55 |
| boost | 212 | 212 | 91.30 | 2.246 (+23.5%) | 13542 (+0.7%) | 80.15 (+3.4%) | 88.3 (-5.0%) | 0.012 (-15.3%) | 0.24 (+91.7%) | 97.64 (+0.9%) |
| hrrn | 222 | 222 | 91.76 | 1.936 (+34.1%) | 11586 (+15.1%) | 79.97 (+3.7%) | 99.6 (+7.2%) | 0.013 (-11.2%) | 1.95 (+31.8%) | 97.34 (+1.2%) |
| fcfs | 223 | 223 | 91.79 | 3.596 (-22.5%) | 11719 (+14.1%) | 86.97 (-4.8%) | 99.3 (+6.9%) | 0.029 (+95.8%) | 4.86 (-70.3%) | 99.00 (-0.5%) |
| dfs-weight | 216 | 217 | 91.32 | 5.220 (-77.8%) | 14056 (-3.0%) | 85.84 (-3.4%) | 88.6 (-4.6%) | 0.021 (+40.7%) | 1.35 (+52.7%) | 98.38 (+0.2%) |

- queue p99 1.9–5.2 s, Jain 0.012–0.029, worst-5% 97.3–99.0%: gần như
  toàn bộ wait mass nằm trong burst bootstrap (~12 request đầu) —
  p90 chỉ 2 ms, hầu hết request vào thẳng prefill: **dưới tải**.
- ITL p99 80.0–87.0 ms như nhau mọi arm; tok/s 88.3–99.6 không theo
  trục queue nào — nhiễu 1 run của closed-loop.
- Kết luận c4: **không có starvation để giải** — mọi khác biệt giữa
  arm (kể cả các mức + lớn như fcfs Jain +95.8%) là nhiễu có cơ sở
  (§7); policy ordering không có việc để làm.

### 4c) Concurrency 8 - Dưới tải - Không có starvation

| Policy | n done | n wait | cache hit (%) | queue p99 (s) | TTFT p99 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 143 | 143 | 94.21 | 3.890 | 16322 | 74.85 | 99.6 | 0.021 | 1.71 | 99.10 |
| boost | 142 | 142 | 94.10 | 4.935 (-26.9%) | 14503 (+11.1%) | 71.91 (+3.9%) | 99.0 (-0.6%) | 0.026 (+20.2%) | 2.37 (-38.4%) | 98.85 (+0.3%) |
| hrrn | 143 | 143 | 94.04 | 1.683 (+56.7%) | 14054 (+13.9%) | 75.69 (-1.1%) | 99.4 (-0.2%) | 0.015 (-29.5%) | 0.60 (+65.0%) | 96.51 (+2.6%) |
| fcfs | 137 | 138 | 94.00 | 1.828 (+53.0%) | 14439 (+11.5%) | 84.51 (-12.9%) | 96.8 (-2.9%) | 0.016 (-27.1%) | 0.64 (+62.4%) | 96.63 (+2.5%) |
| dfs-weight | 142 | 142 | 94.02 | 2.573 (+33.9%) | 14067 (+13.8%) | 84.47 (-12.8%) | 98.9 (-0.7%) | 0.022 (+3.0%) | 0.88 (+48.4%) | 96.84 (+2.3%) |

- Cùng hình c4: queue p99 1.7–4.9 s, worst-5% 96.5–99.1% — wait mass
  vẫn nằm gần hết trong burst bootstrap (~8 request đầu), p90 chỉ
  3 ms.
- TTFT p99 14.1–16.3 s là prefill lạnh lớn (§7), không phải chờ xếp
  hàng — queue p99 chỉ 1.7–4.9 s.
- Kết luận c8: **dưới tải như c4 — chọn policy nào cũng vậy.**

## 5. Vì sao c4/c8 không phân biệt được policy? Head-of-line blocking, và mức + lớn là nhiễu

**Head-of-line blocking (1 request cold-cache chặn đứng batch).**
Fresh boot mỗi cell nên radix cache trống; trace AgentX có context
rất dài. Việc chờ sinh ra từ
1 request lạnh chặn đứng batch, không phải từ thứ tự chọn request:
khi giant chạy, hàng chỉ có 2–3 request, và gốc vấn đề là thiếu
throughput prefill (1 GPU). Muốn hạ khối chờ này phải tăng khả năng
prefill — nhiều GPU.

**Cách đọc Jain.** Jain đo sự đồng đều của chờ, không đo độ tốt.
c4/c8 Jain 0.01–0.03 trông "tệ nhất bảng" nhưng p90 = 2–3 ms: hầu
như ai cũng chờ ≈0, chỉ vài request kẹt sau giant. c16 Jain
0.14–0.32 "cao hơn" = ai cũng chờ nhiều. Jain chỉ có nghĩa khi đọc
cùng queue p99: (0.01, ~2–5 s) = burst bootstrap; (0.14–0.32,
111–366 s) = contention thật.

**Các mức + lớn ở c4/c8 là nhiễu — bốn cơ sở kiểm tra được:**

1. **Số tuyệt đối nhỏ**: boost c4 stretch p99 +91.7% thực ra là
   2.85 → 0.24 — phần trăm phóng đại từ giá trị gốc gần 0.
2. **Mẫu chỉ là một burst ~7–12 request** (worst-5% 97–99%): mọi
   metric queue ở c4/c8 là thống kê trên khoảng một chục request của
   đúng một burst — một giant xong sớm 2 s cũng lật p99 hàng giây và
   Jain hàng chục điểm phần trăm.
3. **Dấu lật tùy metric, không có cơ chế giải thích**: boost c8 có
   TTFT p99 +11.1% nhưng queue p99 −26.9% — khi hàng rỗng không tồn
   tại cơ chế nào tạo trade-off có hệ thống.
4. **Không có sắp xếp request**: p90 = 2–3 ms, policy không có gì
   để sắp — không có gì sinh ra +95.8%.

Các mức + lớn trong bảng c4/c8 là bằng chứng thang đo này chưa đủ tải để phân biệt policy 

**=> Do đó, phải thử nghiệm tiếp ở scale
lớn hơn, nhiều GPU hơn.**

## 6. Kết luận

1. **c4/c8: không có tín hiệu để chọn policy.** Mọi arm dưới tải —
   queue ≈ 0 trừ burst bootstrap (worst-5% 97–99%), mọi khác biệt
   (kể cả mức + đến +95.8%) là nhiễu có cơ sở (§5). **Bottleneck thật
   là head-of-line blocking: 1 request cold-cache chặn đứng batch —
   vấn đề throughput prefill, không phải ordering**.
2. **c16: contention thật, policy phân biệt được.** hrrn tốt nhất
   nhóm xếp hàng + throughput; boost tĩnh đổi TTFT đuôi lấy queue
   đuôi; dfs-weight tệ nhất. boost-ada Eq.9 (γ tự hội tụ ~0.085,
   jump bound 60 s) giữ queue p99 / worst-5% tốt nhất bảng và hết
   regression starvation của estimator wait — hướng adaptive đúng,
   nhưng TTFT đuôi vẫn là giá phải trả ở kiến trúc prefill-queue-only
   (unified queue + preemption của paper còn stage-2).
3. **Cần thử nghiệm ở scale lớn hơn, nhiều GPU hơn.** Production
   chạy nhiều agent đồng thời trên cụm nhiều GPU (GLM-5.3 8×H200,
   TP8 DPA8, router cache_aware/dp-aware) — regime contention thật
   nằm ở đó. Stage-2: 2 thí nghiệm trên 8×H200 — Qwen3.8-27B-FP8
   multi-GPU và GLM-5.3 production-exact — chạy đủ mọi arm qua
   production stack.
