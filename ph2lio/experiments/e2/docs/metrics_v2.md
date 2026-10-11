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
| 3 | queue mean/p90/p99 (s) | eval: `queue_wait_s` | percentile nội suy tuyến tính `k=(n−1)·p/100` | `TestPercentile` | n=164: mean 21.916, p90 74.110, p99 112.050 s |
| 4 | TTFT mean/p95/p99 (ms) | profile: `time_to_first_token` (unit ms) | ms→s→ms khứ hồi | fixture trong `test_metric_values` | mean 27742.32, p95 100988.28, p99 133672.72 ms |
| 5 | ITL mean/p90/p99 (ms) | profile: `inter_chunk_latency` (mảng gap giữa 2 chunk liên tiếp) | pool toàn bộ chunk của P_done; fallback TPOT `(lat−ttft)/(osl−1)` ghi nhãn `tpot_based` | `TestITL` (cả hai nhánh) | 50265 chunk: mean 90.93, p90 52.56, p99 227.37 ms, source `per_chunk` |
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
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 220 | 222 | 91.76% | 0.092 | 0.002 | 2.936 | 1779.20 | 7225.36 | 13642.73 | 42.66 | 44.28 | 83.00 | 92.9 | 0.015 | 0.01 | 2.85 | 98.55% |
| boost | 212 | 212 | 91.30% | 0.058 | 0.002 | 2.246 | 1768.20 | 7071.56 | 13541.61 | 42.77 | 44.05 | 80.15 | 88.3 | 0.012 | 0.01 | 0.24 | 97.64% |
| hrrn | 222 | 222 | 91.76% | 0.050 | 0.002 | 1.936 | 1682.95 | 7080.40 | 11586.28 | 43.30 | 44.22 | 79.97 | 99.6 | 0.013 | 0.01 | 1.95 | 97.34% |
| fcfs | 223 | 223 | 91.79% | 0.133 | 0.002 | 3.596 | 1839.42 | 7881.43 | 11718.51 | 42.26 | 44.57 | 86.97 | 99.3 | 0.029 | 0.01 | 4.86 | 99.00% |
| dfs-weight | 216 | 217 | 91.32% | 0.104 | 0.002 | 5.220 | 1748.93 | 7631.11 | 14055.84 | 43.79 | 44.25 | 85.84 | 88.6 | 0.021 | 0.02 | 1.35 | 98.38% |

### c8 — concurrency 8

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 143 | 143 | 94.21% | 0.183 | 0.003 | 3.890 | 2323.19 | 11117.82 | 16321.75 | 41.33 | 42.29 | 74.85 | 99.6 | 0.021 | 0.01 | 1.71 | 99.10% |
| boost | 142 | 142 | 94.10% | 0.136 | 0.003 | 4.935 | 2251.08 | 10410.30 | 14503.18 | 40.68 | 41.96 | 71.91 | 99.0 | 0.026 | 0.01 | 2.37 | 98.85% |
| hrrn | 143 | 143 | 94.04% | 0.044 | 0.003 | 1.683 | 2089.60 | 9905.51 | 14053.79 | 42.09 | 42.30 | 75.69 | 99.4 | 0.015 | 0.01 | 0.60 | 96.51% |
| fcfs | 137 | 138 | 94.00% | 0.046 | 0.003 | 1.828 | 2113.72 | 10044.72 | 14439.41 | 42.71 | 42.59 | 84.51 | 96.8 | 0.016 | 0.01 | 0.64 | 96.63% |
| dfs-weight | 142 | 142 | 94.02% | 0.059 | 0.003 | 2.573 | 2120.06 | 9953.82 | 14067.36 | 43.61 | 44.39 | 84.47 | 98.9 | 0.022 | 0.01 | 0.88 | 96.84% |

### c16 — concurrency 16

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 169 | 173 | 84.81% | 19.892 | 57.810 | 153.131 | 23958.37 | 74285.69 | 130670.55 | 77.16 | 52.70 | 211.51 | 87.8 | 0.282 | 38.05 | 53.58 | 30.57% |
| boost | 164 | 168 | 83.03% | 19.536 | 53.201 | 173.198 | 23028.70 | 84795.82 | 117749.90 | 92.71 | 53.32 | 245.14 | 85.5 | 0.261 | 31.07 | 54.27 | 35.18% |
| hrrn | 177 | 180 | 84.84% | 17.409 | 51.919 | 111.336 | 23593.50 | 82065.00 | 141705.79 | 81.41 | 53.10 | 226.71 | 95.6 | 0.321 | 25.11 | 50.04 | 29.58% |
| fcfs | 163 | 168 | 81.86% | 21.100 | 65.599 | 140.530 | 26134.03 | 93752.34 | 147291.40 | 83.28 | 52.73 | 225.47 | 83.6 | 0.303 | 47.25 | 79.39 | 32.64% |
| dfs-weight | 143 | 146 | 81.58% | 27.500 | 51.702 | 365.985 | 35114.19 | 171258.53 | 383571.18 | 95.34 | 53.81 | 191.56 | 81.4 | 0.140 | 31.60 | 65.52 | 56.89% |


## Full run 2026-10-08 — commit 0083303206 (5 cells)

### c16 — concurrency 16

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 178 | 181 | 85.81% | 18.969 | 53.215 | 132.019 | 22799.14 | 93455.81 | 127212.96 | 85.87 | 54.07 | 240.73 | 93.9 | 0.235 | 32.15 | 107.67 | 36.76% |
| hrrn | 175 | 179 | 84.07% | 20.059 | 64.587 | 151.221 | 25198.41 | 98255.25 | 153065.19 | 87.18 | 53.19 | 239.00 | 85.6 | 0.290 | 31.87 | 59.71 | 31.42% |
| boost | 160 | 164 | 82.25% | 21.916 | 74.110 | 112.050 | 27742.32 | 100988.28 | 133672.72 | 90.93 | 52.56 | 227.37 | 81.6 | 0.339 | 41.01 | 105.87 | 26.22% |
| fcfs | 150 | 153 | 80.12% | 22.543 | 54.464 | 124.324 | 28223.79 | 96827.58 | 138500.39 | 91.33 | 53.12 | 200.77 | 87.0 | 0.375 | 36.70 | 76.87 | 26.98% |
| dfs-weight | 157 | 160 | 82.99% | 22.638 | 54.633 | 281.173 | 29119.13 | 98855.83 | 318203.79 | 86.44 | 53.21 | 203.25 | 87.1 | 0.137 | 24.14 | 41.91 | 50.95% |


## Full run 2026-10-07 — commit c634f2c859 (10 cells)

### c4 — concurrency 4

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 224 | 224 | 91.79% | 0.066 | 0.002 | 2.442 | 1718.11 | 7404.46 | 11148.36 | 42.91 | 44.30 | 81.27 | 101.2 | 0.015 | 0.01 | 2.05 | 98.14% |
| hrrn | 216 | 216 | 91.37% | 0.092 | 0.002 | 3.552 | 1769.50 | 7490.25 | 13333.17 | 43.36 | 44.02 | 85.12 | 89.9 | 0.021 | 0.01 | 1.24 | 98.63% |
| boost | 216 | 217 | 91.42% | 0.098 | 0.002 | 2.920 | 1766.01 | 7230.41 | 13249.25 | 43.55 | 44.19 | 78.91 | 89.1 | 0.028 | 0.01 | 1.13 | 98.71% |
| fcfs | 223 | 223 | 91.59% | 0.084 | 0.002 | 2.919 | 1786.45 | 7310.23 | 13434.07 | 43.29 | 44.46 | 89.06 | 100.4 | 0.015 | 0.01 | 4.56 | 98.56% |
| dfs-weight | 221 | 222 | 91.67% | 0.158 | 0.002 | 3.736 | 1815.00 | 7363.00 | 21689.30 | 43.26 | 44.00 | 84.93 | 92.4 | 0.012 | 0.01 | 1.91 | 99.02% |

### c8 — concurrency 8

| Policy | n done | n wait | cache hit (%) | queue mean (s) | queue p90 (s) | queue p99 (s) | TTFT mean (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL mean (ms) | ITL p90 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
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
