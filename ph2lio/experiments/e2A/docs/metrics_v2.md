# Bộ metric v2 không phụ thuộc ngưỡng (analysis-only)

Ngày 2026-10-08. Thực thi spec `metrics_v2_prompt.md`: kiểm chứng metric
starvation theo ngưỡng cố định (0.2 s ở c4/c8, 10 s ở c16 — hai report
hai ngưỡng), thay bằng bộ metric v2 ngưỡng-độc lập; Jain chuyển từ TTFT
sang queue wait. **Analysis-only**: không chạy GPU, không sửa server /
run script / serving config; chỉ `analysis/` + tests + docs.

Audit metric hiện tại làm TRƯỚC, viết metric mới SAU:
`docs/metrics_v2_audit.md` (mọi bằng chứng dẫn chiếu dưới nằm trong đó).

Files mới:

| File | Nội dung |
|---|---|
| `analysis/metrics_v2.py` | tính `e2_metrics_v2.json` cho từng run dir |
| `analysis/test_metrics_v2.py` | 28 unit test, toàn bộ fixture tính tay |
| `analysis/print_metrics_v2_table.py` | in bảng v2 (hai block canonical), không đụng `print_results_table.py` |
| `results/.../<run>/e2_metrics_v2.json` | metric v2 từng cell (37 cell landed) |
| `docs/metrics_v2_audit.md` | audit metric v1 + đối chiếu tính tay |

## 0. Bảo toàn dữ liệu cũ

- `e2_metrics.json` cũ **không file nào đổi**: md5 37/37 file giống hệt
  before/after khi chạy `metrics_v2.py --all` (lệnh đối chiếu:
  `find results/1x-rtx6000pro/agentx -name e2_metrics.json | sort | xargs md5sum`).
  Cần thiết vì 12 cell c4/c8 đã mất `server_metrics_export.jsonl` (policy
  gzip): tính lại v1 hôm nay mất nguyên khối `queue` và đổi population
  warmup (audit §đầu) — ghi đè sẽ phá giá trị histogram đã đóng băng.
- Mọi tính toán mới chỉ ghi file mới `e2_metrics_v2.json`; test
  `test_write_does_not_touch_v1` chặn hồi quy này.
- Unit test 28/28 xanh TRƯỚC khi tính lại dữ liệu thật.

## 1. Population (khai báo tường minh trong từng `e2_metrics_v2.json`)

Hai nguồn không join được nhau: `rid` của eval log là id nội bộ server,
`x_request_id` của profile export là UUID client — giao tập hợp = 0
(audit §5e). Mỗi metric ghi trường `population` riêng:

- **`P_done`** — record `profile_export.jsonl` pha profiling, không
  error, đủ 4 tag lõi. Dùng cho: n done, TTFT, ITL, tok/s.
- **`P_wait`** — record `schedule_eval.jsonl` sau `offset_lines`, bỏ
  `HEALTH_CHECK_*`, bỏ arrival trước warmup cut, còn `queue_wait_s` hợp
  lệ. Dùng cho: n wait, cache hit, queue, Jain, stretch, worst-5%.
- Khác biệt n_done < n_wait = request còn in flight khi đóng cửa sổ
  (admission đã ghi, service chưa xong); mỗi cell ghi đủ số đếm
  (`n_offset_sliced`, `n_health_check`, `n_warmup_cut`, `n_no_queue_wait`,
  `n_error`, `n_phase_excluded`).
- **Warmup cut**: nguồn gốc = scrape đầu tiên pha profiling khi còn
  `server_metrics_export.jsonl`; cell đã mất export dùng fallback =
  `request_start_ns` đầu tiên của P_done. Hai nguồn bỏ đúng cùng record
  trên 6/6 cell c16 và phục hồi đúng population frozen của cell c4
  (audit §4); nguồn ghi trong `warmup_cut_source`.

## 2. Provenance từng metric (trường nguồn → công thức → test → ví dụ số thật)

| # | Metric | Trường nguồn | Công thức | Test | Ví dụ thật (boost/c16 wall, 2026-10-08) |
|---|---|---|---|---|---|
| 1 | n done / n wait | đếm P_done / P_wait | — | `test_populations` | 160 (174 − 14 warmup) / 164 (180 − 2 offset − 14 warmup) |
| 2 | cache hit (%) | eval: `matched_prefix_tokens_first_admission`, `input_len` | `sum(matched)/sum(input_len)` token-weighted | `TestCacheHit` | 13359405/16242867 = 0.82248 → 82.25% |
| 3 | queue mean/p50/p90/p95/p99 (s) | eval: `queue_wait_s` | percentile nội suy tuyến tính `k=(n−1)·p/100` | `TestPercentile` | n=164: mean 21.916, p90 74.110, p99 112.050 s |
| 4 | TTFT mean/p50/p95/p99 (ms) | profile: `time_to_first_token` (unit ms) | ms→s→ms khứ hồi | fixture trong `test_metric_values` | mean 27742.32, p95 100988.28, p99 133672.72 ms |
| 5 | ITL mean/p50/p90/p95/p99 (ms) | profile: `inter_chunk_latency` (mảng gap giữa 2 chunk liên tiếp) | pool toàn bộ chunk của P_done; fallback TPOT `(lat−ttft)/(osl−1)` ghi nhãn `tpot_based` | `TestITL` (cả hai nhánh) | 50265 chunk: mean 90.93, p90 52.56, p99 227.37 ms, source `per_chunk` |
| 6 | tok/s | profile: `output_sequence_length`, `request_start_ns`/`request_end_ns` | `sum(osl) / (end_max − start_min)`, cửa sổ ghi trong output | `TestTokPerS` | 148348 token / 1818.18 s = 81.59 tok/s |
| 7 | Jain | eval: `queue_wait_s` | `(Σx)²/(n·Σx²)` trên wait | `TestJainOnWait` | 0.339 |
| 8 | stretch p95/p99 | eval: `queue_wait_s`, `ttft_s`, `input_len − matched_first` | `wait/(ttft−wait)`; population uncached ≥ 256 (floor = k của boost bin); mẫu số ≤ 0 loại + đếm; thiếu `ttft_s` (còn xếp hàng khi đóng cửa sổ) loại + đếm riêng | `TestStretch` (6 case) | n=159: p95 41.01, p99 105.87; dropped_denominator 0, dropped_no_ttft 1 |
| 9 | worst-5% wait share (%) | eval: `queue_wait_s` | `sum(ceil(0.05n) wait lớn nhất)/sum(tất cả) × 100` | `TestWorst5Share` (n=20, n=3, n=1) | 26.22% |

Ngoài danh sách (starvation mọi ngưỡng, cold/warm, gap99) vẫn tính và
lưu trong khối `cross_check` của `e2_metrics_v2.json` (kèm percentile
histogram frozen v1 để đối chiếu), không vào bảng.

## 3. Bảng v2 — ba block canonical, đếm per-full-run (theo `sglang_commit`)

Output của `python3 analysis/print_metrics_v2_table.py`:


## Full run 2026-10-09 — commit 101ea1dcf3 (15 cells)

Sweep mentor-stage-1: c4/c8/c16 × {fcfs-match, lpm, hrrn, dfs-weight,
boost} × 1 run × 30', seed 42, fresh boot mỗi cell (FORCE_RESTART=1),
grace 30s (drain in-flight sau duration, đếm vào metrics). Boost chạy
gamma tĩnh γ̂ offline = 0.09 1/s (wall clock), không adaptive.

### c4 — concurrency 4

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 220 | 222 | 91.76% | 0.092 | 0.002 | 2.936 | 1779.20 | 7225.36 | 13642.73 | 42.66 | 44.28 | 83.00 | 92.9 | 0.015 | 0.01 | 2.85 | 98.55% |
| boost | 212 | 212 | 91.30% | 0.058 | 0.002 | 2.246 | 1768.20 | 7071.56 | 13541.61 | 42.77 | 44.05 | 80.15 | 88.3 | 0.012 | 0.01 | 0.24 | 97.64% |
| hrrn | 222 | 222 | 91.76% | 0.050 | 0.002 | 1.936 | 1682.95 | 7080.40 | 11586.28 | 43.30 | 44.22 | 79.97 | 99.6 | 0.013 | 0.01 | 1.95 | 97.34% |
| fcfs | 223 | 223 | 91.79% | 0.133 | 0.002 | 3.596 | 1839.42 | 7881.43 | 11718.51 | 42.26 | 44.57 | 86.97 | 99.3 | 0.029 | 0.01 | 4.86 | 99.00% |
| dfs-weight | 216 | 217 | 91.32% | 0.104 | 0.002 | 5.220 | 1748.93 | 7631.11 | 14055.84 | 43.79 | 44.25 | 85.84 | 88.6 | 0.021 | 0.02 | 1.35 | 98.38% |

### c8 — concurrency 8

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 143 | 143 | 94.21% | 0.183 | 0.003 | 3.890 | 2323.19 | 11117.82 | 16321.75 | 41.33 | 42.29 | 74.85 | 99.6 | 0.021 | 0.01 | 1.71 | 99.10% |
| boost | 142 | 142 | 94.10% | 0.136 | 0.003 | 4.935 | 2251.08 | 10410.30 | 14503.18 | 40.68 | 41.96 | 71.91 | 99.0 | 0.026 | 0.01 | 2.37 | 98.85% |
| hrrn | 143 | 143 | 94.04% | 0.044 | 0.003 | 1.683 | 2089.60 | 9905.51 | 14053.79 | 42.09 | 42.30 | 75.69 | 99.4 | 0.015 | 0.01 | 0.60 | 96.51% |
| fcfs | 137 | 138 | 94.00% | 0.046 | 0.003 | 1.828 | 2113.72 | 10044.72 | 14439.41 | 42.71 | 42.59 | 84.51 | 96.8 | 0.016 | 0.01 | 0.64 | 96.63% |
| dfs-weight | 142 | 142 | 94.02% | 0.059 | 0.003 | 2.573 | 2120.06 | 9953.82 | 14067.36 | 43.61 | 44.39 | 84.47 | 98.9 | 0.022 | 0.01 | 0.88 | 96.84% |

### c16 — concurrency 16

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 169 | 173 | 84.81% | 19.892 | 57.810 | 153.131 | 23958.37 | 74285.69 | 130670.55 | 77.16 | 52.70 | 211.51 | 87.8 | 0.282 | 38.05 | 53.58 | 30.57% |
| boost | 164 | 168 | 83.03% | 19.536 | 53.201 | 173.198 | 23028.70 | 84795.82 | 117749.90 | 92.71 | 53.32 | 245.14 | 85.5 | 0.261 | 31.07 | 54.27 | 35.18% |
| hrrn | 177 | 180 | 84.84% | 17.409 | 51.919 | 111.336 | 23593.50 | 82065.00 | 141705.79 | 81.41 | 53.10 | 226.71 | 95.6 | 0.321 | 25.11 | 50.04 | 29.58% |
| fcfs | 163 | 168 | 81.86% | 21.100 | 65.599 | 140.530 | 26134.03 | 93752.34 | 147291.40 | 83.28 | 52.73 | 225.47 | 83.6 | 0.303 | 47.25 | 79.39 | 32.64% |
| dfs-weight | 143 | 146 | 81.58% | 27.500 | 51.702 | 365.985 | 35114.19 | 171258.53 | 383571.18 | 95.34 | 53.81 | 191.56 | 81.4 | 0.140 | 31.60 | 65.52 | 56.89% |


## Full run 2026-10-13 — commit 2991704690 (5 cells)

Static-gamma sweep ở c16 (γ = 0.05 / 0.075 / 0.12 / 0.2, wall, k = 256,
adaptive **off** — tách trục static γ thuần) + boost-ada (γ0 = 0.09,
adaptive bật). Cùng protocol: fresh boot (FORCE_RESTART=1), seed 42,
1800 s; GRACE_PERIOD=600 (có cận, khác 10-09 dùng inf) để né drain hang.
γ = 0.02 fail hạ tầng (server không lên sau đợt dọn double-launch) —
chưa có cell. Baseline so sánh: lpm/c16 block 10-09.

γ adaptive hội tụ 0.09 → ~0.042 (final-50 mean; range 0.035–0.096) —
fixed point do workload tự sinh, γ0 bị quên sau ~100 admission đầu.

Caveat harness cell ada: aiperf treo ở drain sau khi profiling xong
(sinh token cuối 00:16, không còn in-flight; bị kill 02:51) — exports
profiling đầy đủ, run_meta + e2_metrics_v2.json dựng tay sau; không có
`profile_export_aiperf.json` (v1).

### c16 — concurrency 16 (đường cong static γ)

| arm | gamma | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lpm (10-09) | — | 169 | 173 | 84.81% | 19.892 | 57.810 | 153.131 | 23958.37 | 74285.69 | 130670.55 | 77.16 | 52.70 | 211.51 | 87.8 | 0.282 | 38.05 | 53.58 | 30.57% |
| boost static | 0.05 | 172 | 172 | 82.36% | 22.354 | 65.546 | 158.679 | 30241.97 | 119295.97 | 198668.34 | 92.82 | 53.39 | 217.03 | 81.6 | 0.297 | 38.80 | 126.24 | 31.35% |
| boost static | 0.075 | 172 | 172 | 82.18% | 23.037 | 96.996 | 141.697 | 31001.43 | 142436.69 | 177753.08 | 93.83 | 53.31 | 223.17 | 82.0 | 0.267 | 34.61 | 91.63 | 31.20% |
| boost static (10-09) | 0.09 | 164 | 168 | 83.03% | 19.536 | 53.201 | 173.198 | 23028.70 | 84795.82 | 117749.90 | 92.71 | 53.32 | 245.14 | 85.5 | 0.261 | 31.07 | 54.27 | 35.18% |
| boost static | 0.12 | 166 | 166 | 80.91% | 25.802 | 97.049 | 159.991 | 33552.21 | 143737.99 | 186807.42 | 89.55 | 53.11 | 211.54 | 82.9 | 0.296 | 34.81 | 98.99 | 30.50% |
| boost static | 0.2 | 173 | 173 | 82.95% | 21.027 | 78.365 | 130.136 | 28600.72 | 134871.47 | 156883.69 | 90.06 | 52.85 | 230.24 | 84.6 | 0.278 | 33.50 | 59.48 | 31.37% |
| boost-ada (hội tụ ~0.042) | 0.09→ | 166 | 166 | 81.66% | 22.325 | 63.775 | 135.108 | 30026.49 | 134694.66 | 180039.78 | 90.87 | 52.85 | 193.38 | 83.2 | 0.309 | 21.60 | 148.51 | 31.05% |

**Đọc sweep (1 run/cell, mọi khác biệt nằm trong band nhiễu — không
đơn điệu theo γ trên bất kỳ metric nào):** TTFT mean 23.0–33.6 s và
queue mean 19.5–25.8 s lật qua lật lại giữa các γ không theo thứ bậc;
"vùng ngọt γ=0.09" (TTFT mean 23.0 s thấp nhất) không replicate thành
xu hướng — 0.05 (30.2) tệ hơn 0.075 (31.0)? rồi 0.09 (23.0) tốt, 0.12
(33.6) tệ, 0.2 (28.6) giữa. Ở thống kê vững (mean trên ~170 request),
**lpm ≥ mọi arm boost ở mọi γ** (queue mean 19.9, TTFT mean 24.0, ITL
mean 77.2, tok/s 87.8). Kết luận stage-1: ở c16/1 GPU, chọn γ không có
tác dụng phân biệt được ngoài nhiễu; boost phải chứng minh ở CCU cao
hơn / nhiều GPU / có repeat.

## Full run 2026-10-14 — commit 60060211ce + cap patch (working tree) (3 cells + 1 void crash)

Quét max-jump bound J của boost-Ada Eq. 9: J ∈ {50, 40} so với J = 60
(cell 2026-10-13). J thấp hơn → floor γ cao hơn (giải bisection
b_γ(k/rate) ≤ J, rate ~2000 tok/s: J = 50 → floor ≈ 0.090 = static-0.09;
J = 40 → floor ≈ 0.107). Cùng protocol: fresh boot (FORCE_RESTART=1),
seed 42, 1800 s, GRACE_PERIOD=600; γ0 = 0.09, wall, k = 256.

Trạng thái code: cell J = 40 chạy đúng 60060211ce; cell J = 50 chạy với
patch chunk-preempt (Φ-keyed cap, xem block dưới khi có) có mặt trong
working tree nhưng flag OFF — behaviorally inert (server_args không có
`boost_chunk_preempt`).

### c16 — concurrency 16

| arm | J | gamma | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| boost-ada Eq.9 (13-10, tham chiếu) | 60 | 0.09→~0.085 | 171 | 171 | 82.49% | 22.522 | 73.364 | 126.404 | 30045.91 | 131392.25 | 158347.13 | 85.95 | 52.79 | 195.92 | 85.6 | 0.299 | 57.61 | 79.68 | 29.27% |
| boost-ada Eq.9 | 50 | 0.09→0.106 | 167 | 167 | 81.68% | 22.189 | 70.970 | 119.619 | 30099.72 | 105729.70 | 140758.66 | 91.94 | 52.97 | 218.80 | 82.8 | 0.365 | 34.90 | 157.74 | 26.12% |
| boost-ada Eq.9 | 40 | 0.09→0.120 | 166 | 166 | 82.00% | 23.710 | 70.544 | 132.833 | 31223.94 | 125026.49 | 146188.66 | 87.10 | 52.91 | 183.33 | 84.3 | 0.335 | 35.43 | 69.26 | 28.27% |
| boost-ada Eq.9 + cap ON (rerun, guard) | 60 | 0.09→0.088 | 166 | 166 | 81.41% | 21.277 | 71.497 | 157.544 | 29539.16 | 132596.88 | 191496.17 | 99.43 | 53.20 | 243.18 | 81.3 | 0.272 | 25.27 | 77.47 | 33.58% |

**Đọc (J-sweep):** floor hoạt động đúng thiết kế — J = 50: γ min
0.0900 bị pin đúng floor (= static-0.09, như dự đoán), final-50 mean
0.1056 (adaptive nhích lên khi tail mỏng đi); J = 40: γ min 0.0900 =
γ0 đầu chạy (chưa chạm floor ~0.107), final-50 mean 0.1200; J = 60: min
0.0804 > floor 0.077 (tự do). **J = 50 là điểm both-wins đầu tiên của
boost-ada:** so J = 60, cả 4 metric cùng cải thiện — queue p99 126.4 →
**119.6 s** (−22% vs lpm 153.1, chỉ sau hrrn 111.3), TTFT p99 158.3 →
**140.8 s** (≈ hrrn 141.7; tệ hơn lpm 7.8% nhưng tốt hơn mọi arm boost
trước đó), Jain 0.299 → **0.365** (cao nhất mọi arm đã đo, > hrrn
0.321), worst-5% 29.27 → **26.12%** (thấp nhất bảng). J = 40 (floor
~0.107, nghiêng FCFS hơn) đánh mất queue p99 (132.8) — J = 50 là mép
giữa của dãy. Giá của J = 50: TTFT mean 30.1 s (lpm 24.0), stretch p99
157.7× (outlier 1 request — stretch p95 34.9 lại tốt hơn lpm 38.05 và
J = 60 57.6), ITL p99 218.8 (tệ nhất trong 3 arm J), tok/s 82.8 (−5.7%
vs lpm). Non-monotonic theo J ở 1 run (stretch p99 79.7/157.7/69.3,
ITL p99 195.9/218.8/183.3, tok/s 85.6/82.8/84.3) — cùng kết luận như γ
sweep: tín hiệu đơn run < nhiễu. Cái phân biệt được: floor pin đúng
nơi dự đoán + J = 50 cải thiện đồng thời 4 metric. Cold/warm queue
p99: J = 50 117.8/117.0 (đều nhau); J = 40 117.8/133.0. Retractions 0
cả hai cell.

### c16 — crash VOID: cap ON lần đầu, server chết OOM prefill ở phút 24

Cell `c16_gada_wall_k256_j60_pre_20261014_104817` (10:48 boot, flag
`boost_chunk_preempt` ON — cell đầu tiên chạy cap thật): boot witness
`boost chunk preempt: enabled` xuất hiện 10:50:26 (factory
`from_schedule` nối flag tới policy — đúng thiết kế). Cap fire đúng
một lần trong 257 lần gọi: 11:04:45, bước 2 của chunk giant ~184k,
cắt 8192 → 7242 token và admit 1 waiting req giữa chunk — đúng cơ chế
mid-chunk admission.

11:14:01 scheduler chết: `alloc_token_slots` raise `RuntimeError: Out
of memory. Try to lower your batch size. Try to allocate 1343 tokens.
Available full tokens: 559 (full_evictable_size_=0)` trong
`prepare_for_extend` của prefill batch kế tiếp. Cơ chế: một giant ~149k
uncached (khác giant 184k ở trên) được adder admit vào chunked prefill
lúc usage 0.72 (3 req decode giữ ~393k / pool ~546k; budget =
available + evictable vừa đủ ở mép); chunk của nó đẩy usage 0.75 →
1.00 trong 45 s, evictable cạn về 0; alloc bước chunk kế thiếu ~784
token. Build này coi prefill alloc failure là fatal (không có đường
retract cho prefill như decode) → kill cả server: 52 success / 639
errors, cell void (không có run_meta.json/e2_metrics.json —
`write_run_meta` chết ở curl `/server_info` khi server đã chết, `set
-e` cắt runner).

**Attribution: không phải do cap.** (1) Batch crash là chính bước
continuation của giant — cap chỉ đổi kích thước alloc 8192 → 1343,
cả hai đều > 559 token còn lại, fail y hệt khi không có cap; (2)
giant được admit lúc `chunked_req=None` — code cap không nằm trong
path admission; (3) cell no-cap J = 60 (13-10) cũng từng đẩy pool lên
1.00 nhiều lần và sống — vì alloc kế tiếp của nó (1644 token, 98k
cached) vừa với ~2.7k token dư. Pool chạm 1.00 là trạng thái thường
xuyên của regime c16 này (16 × ~100k input ≈ 1.6M token cầu vs pool
546k) — sống/chết là coin-flip theo mép dư, không phụ thuộc policy.
Falsification: rerun cùng config — sống → timing luck; crash lại →
quỹ đạo cap/policy (premium fresh-giant của Φ đẩy admit giant ở mép
budget) chạm edge một cách hệ thống → guard degrade-thay-vì-crash
thành điều kiện tiên quyết production. Ghi chú production (8×H200):
prefill alloc failure fatal là lỗ hổng reliability của build này,
độc lập với policy.

### c16 — cap rerun (guard ON): sống hết run, 6/6 prediction bị bác

Rerun `c16_gada_wall_k256_j60_pre_20261014_132712` (13:27 boot, RUN
DONE 14:20:47, 180 records) sau khi guard PrefillBudget (clamp
continuation step vào live capacity + gate load-back) vào working
tree. **Guard hoạt động:** pool chạm `full token usage: 1.00` **12
lần** và sống hết (cell void trước chết ngay lần chạm đầu có alloc
fail; 0 "Scheduler hit an exception"); 5 dòng retract decode —
degrade đúng thiết kế thay vì crash. Cap fire **23/513 lần gọi** (vs
1/257 của cell void).

**6/6 prediction của giả thuyết giant-budget-hog bị bác** (điều kiện
falsify đã định trước: "queue p99 không giảm hoặc TTFT p99 tệ hơn"):
queue p99 dự đoán 100-115 s → thực tế **157.5 s** (tệ hơn cả lpm 153.1,
tệ hơn no-cap J = 60 là 126.4); TTFT p99 dự đoán 120-140 s → **191.5 s**
(tệ nhất mọi arm đã đo); worst-5% dự đoán ≤29.3% → **33.58%**; Jain
dự đoán ≥0.30 → **0.272**; tok/s dự đoán −1-2% → **−5.0%** (81.3 vs
85.6); γ dự đoán rising >0.085 → cuối 0.0884 (danh nghĩa đạt, nhưng
quỹ đạo là decay từ đỉnh 0.326 xuống band 0.084-0.088, min 0.0838 —
không phải rise; band cuối ~ giống no-cap 0.0845). Chỉ 2 metric phụ
không tệ hơn: stretch p99 77.5 (vs 79.7) và TTFT mean 29.5 (vs 30.0).

**Kết luận:** cơ chế dominant của queue-tail ở regime này không phải
"giant chiếm budget, small đói queue" — cắt giant giữa chunk để
admit small làm tail **tệ hơn** (giant chậm đi: mỗi lần fire thêm 1
chu kỳ pause/resume; small được admit giữa chunk giữ KV + decode,
pool thêm áp lực → ITL p99 196→243 ms, 12 lần pool 1.00, 5 retract).
→ đóng hướng chunk-preempt cap, chuyển **stage-2: unified
prefill+decode queue + cross-phase yielding** đúng Phase-2 của paper.

Confound phải ghi: guard là mới trong run này — clamp/skip chunk step
và retract ở pool 1.00 tự thêm queue/TTFT latency; cell no-cap
tham chiếu không bao giờ sống qua các thời điểm đó nên không thể tách
hoàn toàn chi phí cap khỏi chi phí sống-sót áp lực. Nhưng hướng xấu
đồng nhất trên 6/7 metric chính là tín hiệu suy biến nhất quán, không
phải lật nhiễu. Guard giữ nguyên như điều kiện tiên quyết production
(8×H200) — nó là reliability fix, không phải policy.


## Full run 2026-10-13 — commit 60060211ce (1 cell)

boost-Ada **Eq. 9** (code commit 4751c6068d): estimator fit **E2E latency
của request hoàn thành** (UniBoost Eq. 9 — thay cho wait tail nội sinh
của arm ada 2991704690), window 128 / min 32, feed từ
process_batch_result; low clip = **max-jump bound** b_γ(k/rate) ≤ 60 s
(`--boost-adaptive-max-jump-s`, giải bisection từ rate EMA) thay γ_min
thô. γ0 = 0.09, wall, k = 256. Cùng protocol: fresh boot (FORCE_RESTART=1),
seed 42, 1800 s, GRACE_PERIOD=600.

γ trajectory: 0.09 → **~0.085** (final-50 mean 0.0845; min 0.0804 > floor
~0.077, max 0.284 sớm-run) — hết feedback loop ada-wait (0.09 → 0.042);
jump hiệu dụng ~54 s ≈ static 0.09.

### c16 — concurrency 16

| arm | gamma | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| boost-ada Eq.9 (hội tụ ~0.085) | 0.09→ | 171 | 171 | 82.49% | 22.522 | 73.364 | 126.404 | 30045.91 | 131392.25 | 158347.13 | 85.95 | 52.79 | 195.92 | 85.6 | 0.299 | 57.61 | 79.68 | 29.27% |

**Đọc (so các arm c16 cùng seed/protocol):** so ada-wait cùng γ0
(2991704690): stretch p99 148.5 → **79.7** (−46%), TTFT p99 180.0 →
**158.3 s** (−12%), queue p99 135.1 → **126.4 s** — hết regression
starvation của estimator wait. So static 0.09 (10-09): queue p99 **−27%**
(126.4 vs 173.2 — thấp nhất mọi arm kể cả lpm 153.1), worst-5% **29.27%**
thấp nhất bảng, Jain 0.299, n done 171, ITL p99 195.9 (thứ 2 sau
ada-wait); đổi lại TTFT p99 +34% (158.3 vs 117.7 s), stretch p99 +47%
(79.7 vs 54.3). Eq. 9 tự chọn γ ≈ 0.085 (giữa static 0.075–0.09), floor
gần không phải chặn (min 0.0804 > floor 0.077) — giá trị của floor là
bảo hiểm khi workload đổi (tail béo hơn → γ̂ nhỏ hơn → trần jump vẫn
60 s), không phải chốt ngưỡng hôm nay.


## Full run 2026-10-08 — commit 0083303206 (5 cells)

### c16 — concurrency 16

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 178 | 181 | 85.81% | 18.969 | 53.215 | 132.019 | 22799.14 | 93455.81 | 127212.96 | 85.87 | 54.07 | 240.73 | 93.9 | 0.235 | 32.15 | 107.67 | 36.76% |
| hrrn | 175 | 179 | 84.07% | 20.059 | 64.587 | 151.221 | 25198.41 | 98255.25 | 153065.19 | 87.18 | 53.19 | 239.00 | 85.6 | 0.290 | 31.87 | 59.71 | 31.42% |
| boost | 160 | 164 | 82.25% | 21.916 | 74.110 | 112.050 | 27742.32 | 100988.28 | 133672.72 | 90.93 | 52.56 | 227.37 | 81.6 | 0.339 | 41.01 | 105.87 | 26.22% |
| fcfs | 150 | 153 | 80.12% | 22.543 | 54.464 | 124.324 | 28223.79 | 96827.58 | 138500.39 | 91.33 | 53.12 | 200.77 | 87.0 | 0.375 | 36.70 | 76.87 | 26.98% |
| dfs-weight | 157 | 160 | 82.99% | 22.638 | 54.633 | 281.173 | 29119.13 | 98855.83 | 318203.79 | 86.44 | 53.21 | 203.25 | 87.1 | 0.137 | 24.14 | 41.91 | 50.95% |


## Full run 2026-10-07 — commit c634f2c859 (10 cells)

### c4 — concurrency 4

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 224 | 224 | 91.79% | 0.066 | 0.002 | 2.442 | 1718.11 | 7404.46 | 11148.36 | 42.91 | 44.30 | 81.27 | 101.2 | 0.015 | 0.01 | 2.05 | 98.14% |
| hrrn | 216 | 216 | 91.37% | 0.092 | 0.002 | 3.552 | 1769.50 | 7490.25 | 13333.17 | 43.36 | 44.02 | 85.12 | 89.9 | 0.021 | 0.01 | 1.24 | 98.63% |
| boost | 216 | 217 | 91.42% | 0.098 | 0.002 | 2.920 | 1766.01 | 7230.41 | 13249.25 | 43.55 | 44.19 | 78.91 | 89.1 | 0.028 | 0.01 | 1.13 | 98.71% |
| fcfs | 223 | 223 | 91.59% | 0.084 | 0.002 | 2.919 | 1786.45 | 7310.23 | 13434.07 | 43.29 | 44.46 | 89.06 | 100.4 | 0.015 | 0.01 | 4.56 | 98.56% |
| dfs-weight | 221 | 222 | 91.67% | 0.158 | 0.002 | 3.736 | 1815.00 | 7363.00 | 21689.30 | 43.26 | 44.00 | 84.93 | 92.4 | 0.012 | 0.01 | 1.91 | 99.02% |

### c8 — concurrency 8

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 138 | 139 | 94.14% | 0.128 | 0.002 | 3.077 | 2154.11 | 10596.20 | 13881.37 | 41.19 | 42.19 | 74.74 | 96.8 | 0.015 | 0.01 | 1.73 | 98.87% |
| hrrn | 142 | 142 | 94.00% | 0.044 | 0.002 | 1.720 | 2058.37 | 9546.40 | 13898.56 | 42.63 | 43.24 | 77.43 | 98.9 | 0.015 | 0.01 | 0.61 | 96.86% |
| boost | 138 | 138 | 93.93% | 0.107 | 0.003 | 3.077 | 2233.40 | 10433.97 | 14358.45 | 42.54 | 43.07 | 77.37 | 98.5 | 0.018 | 0.01 | 1.75 | 98.60% |
| fcfs | 143 | 143 | 94.13% | 0.086 | 0.002 | 3.067 | 2154.01 | 9985.59 | 16461.77 | 42.62 | 42.76 | 84.57 | 99.7 | 0.019 | 0.01 | 0.85 | 98.37% |
| dfs-weight | 138 | 139 | 94.01% | 0.061 | 0.003 | 2.610 | 2114.71 | 10166.08 | 13937.78 | 40.94 | 42.17 | 76.81 | 96.8 | 0.022 | 0.01 | 0.88 | 97.10% |


## Cross-check: queue p99, eval log (v2) vs frozen histogram (v1)

| Block | Cell | n wait | p99 eval (s) | p99 hist (s) | Δp99 |
|---|---|---|---|---|---|
| 101ea1dcf3 | lpm/c4 | 222 | 2.936 | 3.780 | -22.3% |
| 101ea1dcf3 | boost/c4 | 212 | 2.246 | 2.440 | -7.9% |
| 101ea1dcf3 | hrrn/c4 | 222 | 1.936 | 2.780 | -30.4% |
| 101ea1dcf3 | fcfs/c4 | 223 | 3.596 | 3.885 | -7.4% |
| 101ea1dcf3 | dfs-weight/c4 | 217 | 5.220 | 6.383 | -18.2% |
| 101ea1dcf3 | lpm/c8 | 143 | 3.890 | 4.570 | -14.9% |
| 101ea1dcf3 | boost/c8 | 142 | 4.935 | 6.450 | -23.5% |
| 101ea1dcf3 | hrrn/c8 | 143 | 1.683 | 2.570 | -34.5% |
| 101ea1dcf3 | fcfs/c8 | 138 | 1.828 | 2.620 | -30.2% |
| 101ea1dcf3 | dfs-weight/c8 | 142 | 2.573 | 2.790 | -7.8% |
| 101ea1dcf3 | lpm/c16 | 173 | 153.131 | 213.500 | -28.3% |
| 101ea1dcf3 | boost/c16 | 168 | 173.198 | 172.000 | +0.7% |
| 101ea1dcf3 | hrrn/c16 | 180 | 111.336 | 140.000 | -20.5% |
| 101ea1dcf3 | fcfs/c16 | 168 | 140.530 | 181.333 | -22.5% |
| 101ea1dcf3 | dfs-weight/c16 | 146 | 365.985 | 377.000 | -2.9% |
| 0083303206 | lpm/c16 | 181 | 132.019 | 183.800 | -28.2% |
| 0083303206 | hrrn/c16 | 179 | 151.221 | 174.429 | -13.3% |
| 0083303206 | boost/c16 | 164 | 112.050 | 167.200 | -33.0% |
| 0083303206 | fcfs/c16 | 153 | 124.324 | 174.500 | -28.8% |
| 0083303206 | dfs-weight/c16 | 160 | 281.173 | 340.000 | -17.3% |
| c634f2c859 | lpm/c4 | 224 | 2.442 | 2.760 | -11.5% |
| c634f2c859 | hrrn/c4 | 216 | 3.552 | 3.840 | -7.5% |
| c634f2c859 | boost/c4 | 217 | 2.920 | 2.943 | -0.8% |
| c634f2c859 | fcfs/c4 | 223 | 2.919 | 3.770 | -22.6% |
| c634f2c859 | dfs-weight/c4 | 222 | 3.736 | 4.780 | -21.8% |
| c634f2c859 | lpm/c8 | 139 | 3.077 | 3.610 | -14.8% |
| c634f2c859 | hrrn/c8 | 142 | 1.720 | 2.570 | -33.1% |
| c634f2c859 | boost/c8 | 138 | 3.077 | 3.620 | -15.0% |
| c634f2c859 | fcfs/c8 | 143 | 3.067 | 3.570 | -14.1% |
| c634f2c859 | dfs-weight/c8 | 139 | 2.610 | 2.800 | -6.8% |

## 4. Đọc bảng v2

- **Block 2026-10-09 (sweep stage-1, 15 cell)**: c4/c8 mọi arm gần
  trùng nhau (queue mean 0.04–0.18 s; worst-5% ~97–99% = burst bootstrap
  chiếm gần hết wait mass). Ở c16 khác biệt thật: hrrn tốt nhất mọi trục
  (queue mean 17.41 s, p99 111.3 s, Jain 0.321, tok/s 95.6); dfs-weight
  tệ nhất mọi trục (queue p99 366 s, TTFT p99 383.6 s, Jain 0.140,
  worst-5% 56.9%). **Boost γ=0.09 tĩnh thua chính block 10-08 gamma-Ada**
  (queue p99 173.2 vs 112.1 s; Jain 0.261 vs 0.339; TTFT p99 của boost
  117.7 s là thấp nhất trong cả hai block) — bằng chứng trực tiếp rằng
  adaptation mua được hơn chọn đúng hằng số tĩnh; caveat 1 run/cell, đuôi c16 nhiễu
  lớn, chờ repeat ở stage starvation. Cross-check c16: boost hist khớp
  eval (+0.7%) — block trước vẹch -33% vì bucket v1 giữa phân phối đổi
  hẳn khi p99 rơi vùng khác.

- **"fcfs" trong bảng v2 là fcfs-match** (quyết định 2026-10-08: fcfs-match
  là arm fcfs — ordering giống hệt fcfs, chỉ thêm sort-time match mà sau fix
  bookkeeping `bdc67a5720` không còn cần ép). Cell plain fcfs không vào bảng:
  block 2026-10-07 là artifact pre-fix (hit 0.00%), cell c16 2026-10-08
  (78.49%) giữ trong `e2_metrics_v2.json` làm cross-check với fcfs-match
  (80.12%).
- **ITL per-chunk đọc được ở cả ba band**: mean ~41–91 ms (phân phối lệch
  nặng: p90 ~52–54 ms, đuôi dài kéo mean); đây là
  gap giữa hai chunk stream (chunk gom ~2.3 token), khác TPOT per-request
  mà bảng v1 dùng (không so sánh trực tiếp). 37/37 cell có mảng
  `inter_chunk_latency` → nhánh fallback TPOT-based không cell nào phải
  dùng (test vẫn phủ hai nhánh).
- **Jain trên queue wait thấp hơn Jain trên TTFT** (lpm c16: 0.235 vs
  0.378 ở v1): tách được thời gian prefill hợp lệ ra khỏi "unfairness",
  phần bất công còn lại thuần về xếp hàng. Thứ hạng arm đổi (c16, fcfs =
  fcfs-match): fcfs 0.375 > boost 0.339 > hrrn 0.290 > lpm 0.235 >
  dfs-weight 0.137.
- **stretch**: c16 p95 24–41× nghĩa là request lạnh chờ gấp vài chục lần
  prefill của chính mình; boost p99 105.87× — gam cao nhất, nhất quán
  với cold-starvation trade-off đã biết; c4/c8 p95 ~0.01–4.6 (gần như
  không chờ quá công việc của mình, trừ fcfs c4 4.56×).
- **worst-5% wait share ở c4/c8 ~97–99%**: gần như toàn bộ wait mass nằm
  trong 4–5 request đầu mỗi cell (burst bootstrap, mọi arm như nhau) —
  nén thông tin starvation vào một con số không cần ngưỡng; c16 rơi về
  25–37% (wait trải đều hơn).
- **tok/s** (output token/giây trên cửa sổ profiling): c16 boost 81.6
  thấp nhất; lpm 93.9 cao nhất.
- **queue percentile c16**: eval-log khớp histogram ở phần giữa phân phối
  (p95 từng đo ±1.3% — bucket giữa còn mịn); p99 mới là chỗ histogram vẹch
  (−13..−33%), nên bảng cross-check in p99. Ở c4/c8 thì histogram vẹch từ
  rất sớm (−40..−68% ngay tại p95).

## 5. Chạy lại

```bash
python3 analysis/test_metrics_v2.py            # 28 test, phải xanh trước
python3 analysis/metrics_v2.py --all           # ghi e2_metrics_v2.json cho mọi cell landed
python3 analysis/print_metrics_v2_table.py     # hai block canonical
python3 analysis/print_metrics_v2_table.py --all-blocks   # mọi block landed
```

## 6. Acceptance (khớp spec §6)

- Doc audit hoàn chỉnh, từng metric cũ có công thức + 3 request đối
  chiếu tính tay: `docs/metrics_v2_audit.md` (§1 bản đồ, §2 tính tay,
  §3 cross-check 18 cell).
- `e2_metrics_v2.json` cho mọi cell landed (37/37); bảng v2 in đủ
  c4/c8/c16 đúng cột danh sách spec (§3 trên).
- 28/28 unit test xanh trước khi tính lại dữ liệu thật; provenance doc
  (metric → trường nguồn → công thức → test → ví dụ số thật) = §2 trên.
- Checksum md5 37/37 `e2_metrics.json` trước/sau giống hệt; không đổi
  server code / run scripts / serving config.
- Không thêm metric ngoài danh sách; không chạy thí nghiệm mới.
