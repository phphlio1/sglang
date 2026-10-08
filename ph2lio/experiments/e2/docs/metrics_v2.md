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
| 3 | queue avg/p95/p99 (s) | eval: `queue_wait_s` | percentile nội suy tuyến tính `k=(n−1)·p/100` | `TestPercentile` | n=164: mean 21.916, p95 94.109, p99 112.050 s |
| 4 | TTFT avg/p95/p99 (ms) | profile: `time_to_first_token` (unit ms) | ms→s→ms khứ hồi | fixture trong `test_metric_values` | mean 27742.32, p95 100988.28, p99 133672.72 ms |
| 5 | ITL p50/p95/p99 (ms) | profile: `inter_chunk_latency` (mảng gap giữa 2 chunk liên tiếp) | pool toàn bộ chunk của P_done; fallback TPOT `(lat−ttft)/(osl−1)` ghi nhãn `tpot_based` | `TestITL` (cả hai nhánh) | 50265 chunk: p50 49.66, p95 54.48, p99 227.37 ms, source `per_chunk` |
| 6 | tok/s | profile: `output_sequence_length`, `request_start_ns`/`request_end_ns` | `sum(osl) / (end_max − start_min)`, cửa sổ ghi trong output | `TestTokPerS` | 148348 token / 1818.18 s = 81.59 tok/s |
| 7 | Jain | eval: `queue_wait_s` | `(Σx)²/(n·Σx²)` trên wait | `TestJainOnWait` | 0.339 |
| 8 | stretch p95/p99 | eval: `queue_wait_s`, `ttft_s`, `input_len − matched_first` | `wait/(ttft−wait)`; population uncached ≥ 256 (floor = k của boost bin); mẫu số ≤ 0 loại + đếm; thiếu `ttft_s` (còn xếp hàng khi đóng cửa sổ) loại + đếm riêng | `TestStretch` (6 case) | n=159: p95 41.01, p99 105.87; dropped_denominator 0, dropped_no_ttft 1 |
| 9 | worst-5% wait share (%) | eval: `queue_wait_s` | `sum(ceil(0.05n) wait lớn nhất)/sum(tất cả) × 100` | `TestWorst5Share` (n=20, n=3, n=1) | 26.22% |

Ngoài danh sách (starvation mọi ngưỡng, cold/warm, gap99) vẫn tính và
lưu trong khối `cross_check` của `e2_metrics_v2.json` (kèm percentile
histogram frozen v1 để đối chiếu), không vào bảng.

## 3. Bảng v2 — hai block canonical, đếm per-full-run (theo `sglang_commit`)

Output của `python3 analysis/print_metrics_v2_table.py`:

## Full run 2026-10-08 — commit 0083303206 (6 cells)

### c16 — concurrency 16

| Policy | n done | n wait | cache hit (%) | queue avg (s) | queue p95 (s) | queue p99 (s) | TTFT avg (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p95 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
**lpm (baseline)** | 178 | 181 | 85.81% | 18.969 | 78.946 | 132.019 | 22799.14 | 93455.81 | 127212.96 | 49.40 | 95.20 | 240.73 | 93.9 | 0.235 | 32.15 | 107.67 | 36.76%
hrrn | 175 | 179 | 84.07% | 20.059 | 89.665 | 151.221 | 25198.41 | 98255.25 | 153065.19 | 49.27 | 58.68 | 239.00 | 85.6 | 0.290 | 31.87 | 59.71 | 31.42%
boost | 160 | 164 | 82.25% | 21.916 | 94.109 | 112.050 | 27742.32 | 100988.28 | 133672.72 | 49.66 | 54.48 | 227.37 | 81.6 | 0.339 | 41.01 | 105.87 | 26.22%
fcfs-match | 150 | 153 | 80.12% | 22.543 | 91.745 | 124.324 | 28223.79 | 96827.58 | 138500.39 | 50.61 | 54.20 | 200.77 | 87.0 | 0.375 | 36.70 | 76.87 | 26.98%
fcfs | 149 | 152 | 78.49% | 22.764 | 90.103 | 118.128 | 29614.01 | 100711.10 | 122862.94 | 49.62 | 54.21 | 176.04 | 64.5 | 0.361 | 30.57 | 60.57 | 25.51%
dfs-weight | 157 | 160 | 82.99% | 22.638 | 94.292 | 281.173 | 29119.13 | 98855.83 | 318203.79 | 49.94 | 55.74 | 203.25 | 87.1 | 0.137 | 24.14 | 41.91 | 50.95%

## Full run 2026-10-07 — commit c634f2c859 (12 cells)

### c4 — concurrency 4

| Policy | n done | n wait | cache hit (%) | queue avg (s) | queue p95 (s) | queue p99 (s) | TTFT avg (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p95 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
**lpm (baseline)** | 224 | 224 | 91.79% | 0.066 | 0.002 | 2.442 | 1718.11 | 7404.46 | 11148.36 | 36.54 | 45.29 | 81.27 | 101.2 | 0.015 | 0.01 | 2.05 | 98.14%
hrrn | 216 | 216 | 91.37% | 0.092 | 0.003 | 3.552 | 1769.50 | 7490.25 | 13333.17 | 35.50 | 44.99 | 85.12 | 89.9 | 0.021 | 0.01 | 1.24 | 98.63%
boost | 216 | 217 | 91.42% | 0.098 | 0.003 | 2.920 | 1766.01 | 7230.41 | 13249.25 | 35.19 | 45.99 | 78.91 | 89.1 | 0.028 | 0.01 | 1.13 | 98.71%
fcfs-match | 223 | 223 | 91.59% | 0.084 | 0.002 | 2.919 | 1786.45 | 7310.23 | 13434.07 | 36.26 | 45.27 | 89.06 | 100.4 | 0.015 | 0.01 | 4.56 | 98.56%
fcfs | 216 | 217 | 0.00% | 0.039 | 0.001 | 0.003 | 1680.61 | 7149.91 | 13982.07 | 36.26 | 44.85 | 88.19 | 89.1 | 0.008 | 0.01 | 0.01 | 97.91%
dfs-weight | 221 | 222 | 91.67% | 0.158 | 0.003 | 3.736 | 1815.00 | 7363.00 | 21689.30 | 36.81 | 44.96 | 84.93 | 92.4 | 0.012 | 0.01 | 1.91 | 99.02%

### c8 — concurrency 8

| Policy | n done | n wait | cache hit (%) | queue avg (s) | queue p95 (s) | queue p99 (s) | TTFT avg (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p95 (ms) | ITL p99 (ms) | tok/s | Jain | stretch p95 | stretch p99 | worst-5% wait (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
**lpm (baseline)** | 138 | 139 | 94.14% | 0.128 | 0.003 | 3.077 | 2154.11 | 10596.20 | 13881.37 | 38.32 | 44.70 | 74.74 | 96.8 | 0.015 | 0.01 | 1.73 | 98.87%
hrrn | 142 | 142 | 94.00% | 0.044 | 0.003 | 1.720 | 2058.37 | 9546.40 | 13898.56 | 38.37 | 44.92 | 77.43 | 98.9 | 0.015 | 0.01 | 0.61 | 96.86%
boost | 138 | 138 | 93.93% | 0.107 | 0.003 | 3.077 | 2233.40 | 10433.97 | 14358.45 | 38.41 | 45.02 | 77.37 | 98.5 | 0.018 | 0.01 | 1.75 | 98.60%
fcfs-match | 143 | 143 | 94.13% | 0.086 | 0.003 | 3.067 | 2154.01 | 9985.59 | 16461.77 | 38.42 | 45.11 | 84.57 | 99.7 | 0.019 | 0.01 | 0.85 | 98.37%
fcfs | 146 | 146 | 0.00% | 0.084 | 0.002 | 3.059 | 2166.02 | 9948.82 | 16379.90 | 38.32 | 44.92 | 83.29 | 100.1 | 0.019 | 0.01 | 0.82 | 98.89%
dfs-weight | 138 | 139 | 94.01% | 0.061 | 0.003 | 2.610 | 2114.71 | 10166.08 | 13937.78 | 38.32 | 43.76 | 76.81 | 96.8 | 0.022 | 0.01 | 0.88 | 97.10%

Bảng phụ cross-check queue p95/p99 (eval log vs histogram frozen v1,
từng cell của hai block — bằng chứng di chuyển bảng v2 sang eval log):

| Block | Cell | n wait | p95 eval (s) | p95 hist (s) | Δp95 | p99 eval (s) | p99 hist (s) | Δp99 |
|---|---|---|---|---|---|---|---|
0083303206 | lpm/c16 | 181 | 78.946 | 79.750 | -1.0% | 132.019 | 183.800 | -28.2% |
0083303206 | hrrn/c16 | 179 | 89.665 | 90.250 | -0.6% | 151.221 | 174.429 | -13.3% |
0083303206 | boost/c16 | 164 | 94.109 | 93.600 | +0.5% | 112.050 | 167.200 | -33.0% |
0083303206 | fcfs-match/c16 | 153 | 91.745 | 91.750 | -0.0% | 124.324 | 174.500 | -28.8% |
0083303206 | fcfs/c16 | 152 | 90.103 | 91.333 | -1.3% | 118.128 | 169.600 | -30.3% |
0083303206 | dfs-weight/c16 | 160 | 94.292 | 95.000 | -0.7% | 281.173 | 340.000 | -17.3% |
c634f2c859 | lpm/c4 | 224 | 0.002 | 0.005 | -54.3% | 2.442 | 2.760 | -11.5% |
c634f2c859 | hrrn/c4 | 216 | 0.003 | 0.005 | -48.4% | 3.552 | 3.840 | -7.5% |
c634f2c859 | boost/c4 | 217 | 0.003 | 0.005 | -45.4% | 2.920 | 2.943 | -0.8% |
c634f2c859 | fcfs-match/c4 | 223 | 0.002 | 0.005 | -55.8% | 2.919 | 3.770 | -22.6% |
c634f2c859 | fcfs/c4 | 217 | 0.001 | 0.004 | -67.5% | 0.003 | 0.005 | -38.3% |
c634f2c859 | dfs-weight/c4 | 222 | 0.003 | 0.005 | -47.9% | 3.736 | 4.780 | -21.8% |
c634f2c859 | lpm/c8 | 139 | 0.003 | 0.005 | -43.8% | 3.077 | 3.610 | -14.8% |
c634f2c859 | hrrn/c8 | 142 | 0.003 | 0.005 | -44.8% | 1.720 | 2.570 | -33.1% |
c634f2c859 | boost/c8 | 138 | 0.003 | 0.005 | -40.1% | 3.077 | 3.620 | -15.0% |
c634f2c859 | fcfs-match/c8 | 143 | 0.003 | 0.005 | -44.1% | 3.067 | 3.570 | -14.1% |
c634f2c859 | fcfs/c8 | 146 | 0.002 | 0.005 | -64.6% | 3.059 | 3.540 | -13.6% |
c634f2c859 | dfs-weight/c8 | 139 | 0.003 | 0.005 | -40.2% | 2.610 | 2.800 | -6.8% |

## 4. Đọc bảng v2

- **fcfs c4/c8 hit 0.00% là artifact pre-fix**: block 2026-10-07 chạy
  trước fix fcfs admission bookkeeping (commit `bdc67a5720`); cell c16
  2026-10-08 (post-fix) đọc 82.25%/78.49% thật. Bảng v1 dùng marker ‡ cho
  các dòng này; bảng v2 giữ đúng danh sách cột của spec nên ghi chú tại
  đây thay vì thêm marker.
- **ITL per-chunk đọc được ở cả ba band**: p50 ~35–50 ms ổn định; đây là
  gap giữa hai chunk stream (chunk gom ~2.3 token), khác TPOT per-request
  mà bảng v1 dùng (không so sánh trực tiếp). 37/37 cell có mảng
  `inter_chunk_latency` → nhánh fallback TPOT-based không cell nào phải
  dùng (test vẫn phủ hai nhánh).
- **Jain trên queue wait thấp hơn Jain trên TTFT** (lpm c16: 0.235 vs
  0.378 ở v1): tách được thời gian prefill hợp lệ ra khỏi "unfairness",
  phần bất công còn lại thuần về xếp hàng. Thứ hạng arm đổi: fcfs-match
  0.375 > fcfs 0.361 > boost 0.339 > hrrn 0.290 > lpm 0.235 >
  dfs-weight 0.137 (c16).
- **stretch**: c16 p95 24–41× nghĩa là request lạnh chờ gấp vài chục lần
  prefill của chính mình; boost p99 105.87× — gam cao nhất, nhất quán
  với cold-starvation trade-off đã biết; c4/c8 p95 ~0.01–4.6 (gần như
  không chờ quá công việc của mình, trừ fcfs-match c4 4.56×).
- **worst-5% wait share ở c4/c8 ~97–99%**: gần như toàn bộ wait mass nằm
  trong 4–5 request đầu mỗi cell (burst bootstrap, mọi arm như nhau) —
  nén thông tin starvation vào một con số không cần ngưỡng; c16 rơi về
  25–37% (wait trải đều hơn).
- **tok/s** (output token/giây trên cửa sổ profiling): c16 fcfs 64.5
  thấp nhất — nhất quán với hit thấp nhất; lpm 93.9 cao nhất.
- **queue p95 c16**: eval-log (78.9–94.3 s) khớp histogram ±1.3% — bucket
  giữa còn mịn; p99 mới là chỗ histogram vẹch (−13..−33%). Ở c4/c8 thì
  ngay p95 đã vô dụng (−40..−68%).

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
