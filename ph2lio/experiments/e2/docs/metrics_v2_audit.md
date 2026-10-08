# Audit metric hiện tại (trước khi xây bộ metric v2)

Ngày 2026-10-08. Phạm vi: hai block đã landed dưới
`results/1x-rtx6000pro/agentx/` — full run c4/c8 (2026-10-07, commit
`c634f2c859`, 12 cell) và c16 sweep (2026-10-08, commit `0083303206`,
6 cell). Analysis-only: không chạy GPU, không sửa server / run script;
mọi tính toán mới ghi vào `e2_metrics_v2.json` riêng, `e2_metrics.json`
cũ chỉ đọc (checksum before/after trong `docs/metrics_v2.md`).

Kết luận tóm tắt (chi tiết từng mục dưới):

1. Tính lại `agentx_metrics.compute_metrics` trên cell c16 (còn
   `server_metrics_export.jsonl`) reproducer **đúng tuyệt đối** giá trị
   frozen trong `e2_metrics.json` (1 diff duy nhất: chuỗi `run_dir` tuyệt
   đối vs tương đối). Code hiện tại deterministic, giá trị frozen tin được.
2. Tính lại trên cell c4 (export đã dọn theo policy gzip) **mất nguyên
   khối `queue`** (histogram) và warmup cut eval đổi population (cold 10
   vs frozen 7) → bằng chứng trực tiếp vì sao **không được ghi đè
   `e2_metrics.json`** (§0.1 của spec v2).
3. Percentile queue từ histogram lệch nặng so với eval log per-request:
   p95 ở c4/c8 lệch **−40%..−68%** (bucket thô bậc 1–5 ms), p99 ở c16
   histogram **cao hơn +13%..+33%** (bucket đuôi rộng) → bảng v2 chuyển
   sang eval log (§3 dưới).
4. `rid` của eval log **không join được** với `x_request_id` của profile
   export (test giao tập hợp = 0 trên 2 cell; `rid` là id nội bộ server,
   32 hex, khác UUID client) → v2 dùng hai population `P_done` / `P_wait`
   khai báo tường minh.
5. `inter_chunk_latency` trong profile export là **mảng per-chunk**
   (gap giữa hai chunk liên tiếp) → ITL thật theo chunk khả thi; ngược lại
   `inter_token_latency` aggregate của AIPerf thực chất là phân phối
   TPOT per-request (p95 168.140039675861 == TPOT p95 của code đến 12 chữ
   số) — phải ghi nhãn trung thực nếu dùng.

## 1. Bản đồ metric hiện tại

Nguồn: `analysis/agentx_metrics.py` + `analysis/starvation.py`. Population
ký hiệu: **P-profile** = record `profile_export.jsonl` với
`metadata.benchmark_phase == "profiling"`, không `error`, đủ 4 tag
(`time_to_first_token`, `request_latency`, `input_sequence_length`,
`output_sequence_length`) — đây là `n_complete`. **P-eval** = record
`schedule_eval.jsonl` sau khi cắt `run_meta.schedule_eval.offset_lines`,
bỏ `rid` bắt đầu `HEALTH_CHECK_`, còn lại warmup cut (mục 1.3).

| Metric | Trường nguồn | Công thức | Đơn vị | Population |
|---|---|---|---|---|
| TTFT client | `profile_export.jsonl` → `metrics.time_to_first_token` (unit `ms`) | `_metric_seconds`: ms / 1000 → s; `percentile_stats`: s × 1000 → ms, mean, p95, p99 | ms (đổi ms→s→ms) | P-profile |
| E2E latency | `metrics.request_latency` | như TTFT | ms | P-profile |
| TPOT | dẫn xuất | `(latency − ttft) / (osl − 1)` với `osl ≥ 2` | s/token → ms | P-profile ∧ osl≥2 |
| RPM / oTPM / TPM | `profile_export_aiperf.json` → `request_throughput` / `output_token_throughput` / `total_token_throughput` (trường `avg`) | × 60 | req/min, tok/min | toàn run (AIPerf aggregate) |
| Queue histogram | `server_metrics_export.jsonl` (family `sglang:queue_time_seconds`) — **đã mất ở cell c4/c8** | delta first→last scrape pha profiling; `prometheus_quantile` nội suy tuyến tính trong bucket; `starvation_rate` = share bucket trên ngưỡng | s | mọi request qua queue trong cửa sổ profiling (không lọc warmup/error) |
| Queue wait per-request | `schedule_eval.jsonl` → `queue_wait_s` | lấy trực tiếp; percentile `_percentile` nội suy tuyến tính `k = (n−1)·p/100` | s | P-eval (chia cold/warm theo `matched_prefix_tokens_first_admission`) |
| Cache hit | `schedule_eval.jsonl` → `matched_prefix_tokens_first_admission`, `input_len` | `sum(matched) / sum(input_len)` (token-weighted) | fraction | P-eval |
| Jain TTFT | dẫn xuất | `(Σx)² / (n·Σx²)` trên `ttft_ms` | 0–1 | P-profile |
| Warmup cut (client) | `metadata.benchmark_phase != "profiling"` | loại record | — | — |
| Warmup cut (eval) | `server_metrics_export.jsonl` → scrape đầu tiên pha profiling (`timestamp_ns`/1e9) | bỏ record `arrival_wall_s < cut` | s epoch | P-eval |

Percentile dùng một công thức duy nhất (`starvation._percentile`): với mảng
đã sort, `k = (n−1)·p/100`, nội suy tuyến tính giữa `floor(k)` và `ceil(k)`;
`n = 0` → NaN, `n = 1` → chính giá trị đó.

## 2. Đối chiếu tính tay — cell `boost/c16_gada_wall_k256_20261008_045917`

### 2.1 Ba record eval log (nguyên liệu in từ `schedule_eval.jsonl`)

Record A (bị warmup cut):

```json
{"rid": "9b3a2a60eeab45f6981a65a5df4ee5f2", "arrival_wall_s": 1791411087.6237664,
 "queue_wait_s": 16.188697147008497, "ttft_s": 49.65718958998332,
 "input_len": 110405, "matched_prefix_tokens_first_admission": 0}
```

Record B (cold, giữ lại):

```json
{"rid": "e2bf405cc0f04c029c88678340071090", "arrival_wall_s": 1791411841.8735368,
 "queue_wait_s": 3.732870854990324, "ttft_s": 4.3163149709871504,
 "input_len": 1155, "matched_prefix_tokens_first_admission": 0}
```

Record C (warm, match cao nhất cell):

```json
{"rid": "a2ca7c1241b440bf98366786d7409367", "arrival_wall_s": 1791411455.0053496,
 "queue_wait_s": 0.002533299004426226, "ttft_s": 1.7969286580046173,
 "input_len": 213334, "matched_prefix_tokens_first_admission": 210816}
```

Tính tay từng bước:

- **Warmup cut**: cut_s = scrape đầu tiên pha profiling =
  1791411137.400277. Record A: 1791411087.6238 < 1791411137.4003 → **bị
  cắt** (đúng: A là record warmup-phase, `queue_wait_s` 16.19 s của nó
  không được lọt vào P-eval). Record B (1791411841.87) và C
  (1791411455.01) ≥ cut → giữ lại.
- **Queue wait**: B = 3.732870854990324 s, C = 0.002533299004426226 s —
  lấy thẳng trường, không biến đổi. Hai giá trị này nằm trong mảng
  `queue_wait_s` của 164 record giữ lại (cell này: 180 dòng − offset 2 −
  14 warmup = 164).
- **Phân lớp cold/warm**: B có `matched = 0` → cold; C có
  `matched = 210816 > 0` → warm. Cell này: 18 cold / 146 warm.
- **Cache hit (đóng góp của C)**: 210816/213334 = 0.98819 (98.8% input
  của C là hit). Tổng cell: sum(matched) = 13 359 405, sum(input_len) =
  16 242 867 → 13359405/16242867 = 0.82248 → **82.25%**, khớp giá trị
  bảng c16 (82.25%) và `first_attempt_cache_hit_rate` trong
  `e2_metrics.json`.

### 2.2 Ba record profile (nguyên liệu in từ `profile_export.jsonl`, pha profiling)

| x_request_id | isl | osl | ttft (ms) | latency (ms) |
|---|---|---|---|---|
| `1d508bff-…` | 514 | 20 | 178.828142 | 520.355617 |
| `d2231e5a-…` | 100104 | 1195 | 6514.229395 | 31804.634556 |
| `073a6910-…` | 214305 | 26571 | 66231.625317 | 669394.103665 |

Tính tay từng bước:

- **Đổi đơn vị ms→s**: 178.828142 ms → 0.178828142 s (chia 1000). Code
  lưu mảng `ttft_s` rồi `percentile_stats` nhân ngược 1000 khi in ms —
  khứ hồi không đổi giá trị: 0.178828142 × 1000 = 178.828142 ms, bằng
  đúng `min` của aggregate AIPerf (`time_to_first_token.min` =
  178.82814199999999).
- **TPOT** = (latency − ttft)/(osl − 1):
  - `d2231e5a`: (31804.634556 − 6514.229395)/(1195 − 1) =
    25290.405161/1194 = **21.18124 ms**. Code: `tpot_s` = 0.021181244 →
    21.18124 ms. Khớp.
  - `073a6910`: (669394.103665 − 66231.625317)/(26571 − 1) =
    603162.478348/26570 = **22.70088 ms**. Code: 0.022700884 →
    22.70088 ms. Khớp.
  - `1d508bff`: (520.355617 − 178.828142)/(20 − 1) = 341.527475/19 =
    **17.97513 ms**. Code: 0.017975130 → 17.97513 ms. Khớp.
- **Population**: cell này 174 record profile = 160 profiling + 14 warmup
  (bỏ) + 0 error → `n_complete` = 160; cả 160 record có osl ≥ 2 nên mảng
  TPOT cũng 160. P-eval giữ 164 > 160 = P-profile: 4 admission còn in
  flight khi đóng cửa sổ (khác biệt population hai nguồn, ghi nhận cho
  v2).

### 2.3 Phát hiện ghi nhãn: `inter_token_latency` aggregate = TPOT

`profile_export_aiperf.json` của cell này ghi `inter_token_latency`
p95 = 168.140039675861, p99 = 279.2323400879 — bằng đúng p95/p99 của
mảng TPOT per-request code tính ra (168.14003967586137 /
279.2323400879, khớp đến 12 chữ số). Tức aggregate "inter token latency"
của AIPerf thực chất là phân phối `(latency−ttft)/(osl−1)` per-request,
không phải gap token thật. Ghi nhãn trung thực khi đối chiếu.

### 2.4 ITL per-chunk: schema `inter_chunk_latency`

`metrics.inter_chunk_latency.value` là **mảng** gap giữa hai chunk liên
tiếp (unit ms), vd record osl=213 nhận 91 chunk có 90 gap, gap đầu
32.16995 ms == `time_to_second_token`. Trên 160 record profiling của
cell này: tổng 50 265 gap; phân bố `len(mảng) − chunks_received` =
−1 ở 134 record, 0 ở 19, +1 ở 6, −2 ở 1 (cách đếm chunk của export không
hoàn toàn nhất quán ở mép) — v2 dùng nguyên mảng gap, count ghi vào
`e2_metrics_v2.json`. Ở cell c4 (`lpm/c4_20261007_022137`): 224/227
record có mảng (3 record còn lại không có gap), schema đồng nhất giữa hai
block → nhánh ITL thật (per-chunk) dùng được cho cả 18 cell.

## 3. Cross-check queue percentile: eval log (chính xác) vs histogram (frozen)

Cùng population-cut, p50/p95/p99 tính từ `queue_wait_s` per-request của
eval log so với giá trị histogram đã đóng băng trong `e2_metrics.json`
cũ (cell c4/c8: histogram computed lúc export còn, nay chỉ còn giá trị
frozen):

| cell | n | eval p50 | hist p50 | eval p95 | hist p95 | eval p99 | hist p99 | Δp95 | Δp99 |
|---|---|---|---|---|---|---|---|---|---|
| boost/c16 | 164 | 8.702 | 8.636 | 94.109 | 93.600 | 112.050 | 167.200 | +0.5% | −33.0% |
| dfs-weight/c16 | 160 | 4.999 | 5.000 | 94.292 | 95.000 | 281.173 | 340.000 | −0.7% | −17.3% |
| fcfs/c16 | 152 | 11.833 | 11.957 | 90.103 | 91.333 | 118.128 | 169.600 | −1.3% | −30.3% |
| fcfs-match/c16 | 153 | 14.114 | 13.553 | 91.745 | 91.750 | 124.324 | 174.500 | −0.0% | −28.8% |
| hrrn/c16 | 179 | 7.556 | 7.946 | 89.665 | 90.250 | 151.221 | 174.429 | −0.6% | −13.3% |
| lpm/c16 | 181 | 6.586 | 6.071 | 78.946 | 79.750 | 132.019 | 183.800 | −1.0% | −28.2% |
| boost/c4 | 217 | 0.001 | 0.003 | 0.003 | 0.005 | 2.920 | 2.943 | −45.4% | −0.8% |
| dfs-weight/c4 | 222 | 0.002 | 0.003 | 0.003 | 0.005 | 3.736 | 4.780 | −47.9% | −21.8% |
| fcfs/c4 | 217 | 0.001 | 0.001 | 0.001 | 0.004 | 0.003 | 0.005 | −67.5% | −38.3% |
| fcfs-match/c4 | 223 | 0.001 | 0.002 | 0.002 | 0.005 | 2.919 | 3.770 | −55.8% | −22.6% |
| hrrn/c4 | 216 | 0.001 | 0.002 | 0.003 | 0.005 | 3.552 | 3.840 | −48.4% | −7.5% |
| lpm/c4 | 224 | 0.001 | 0.002 | 0.002 | 0.005 | 2.442 | 2.760 | −54.3% | −11.5% |
| boost/c8 | 138 | 0.001 | 0.003 | 0.003 | 0.005 | 3.077 | 3.620 | −40.1% | −15.0% |
| dfs-weight/c8 | 139 | 0.002 | 0.003 | 0.003 | 0.005 | 2.610 | 2.800 | −40.2% | −6.8% |
| fcfs/c8 | 146 | 0.001 | 0.001 | 0.002 | 0.005 | 3.059 | 3.540 | −64.6% | −13.6% |
| fcfs-match/c8 | 143 | 0.001 | 0.003 | 0.003 | 0.005 | 3.067 | 3.570 | −44.1% | −14.1% |
| hrrn/c8 | 142 | 0.001 | 0.003 | 0.003 | 0.005 | 1.720 | 2.570 | −44.8% | −33.1% |
| lpm/c8 | 139 | 0.002 | 0.003 | 0.003 | 0.005 | 3.077 | 3.610 | −43.8% | −14.8% |

Đọc bảng:

- **c4/c8**: wait thật bậc 1–3 ms; bucket histogram bậc 0.001/0.002/
  0.005 s → p95 histogram luôn đọc 0.004–0.005 s, lệch −40%..−68% so với
  eval log. Histogram không phân giải được vùng này.
- **c16**: p95 hai nguồn khớp ±1.3% (bucket giữa vẫn đủ mịn), nhưng p99
  histogram cao hơn +13%..+33% — bucket đuôi rộng 50–100 s, mọi giá trị
  110–170 s bị nội suy về biên bucket trên. Đây là bằng chứng đúng như
  kỳ vọng của spec và là lý do bảng v2 không dùng histogram.
- n eval của c16 (164–181) ≠ n histogram (152–181): histogram đếm mọi
  request qua queue trong cửa sổ scrape, eval log đếm admission sau
  offset + cut — cùng bậc, chênh ≤ 3.

## 4. Kiểm chứng warmup cut fallback (cho cell đã mất export)

`first_profiling_scrape_s` cần `server_metrics_export.jsonl` — đã mất ở
12 cell c4/c8. Fallback đề xuất cho v2: cut =
`min(request_start_ns)` của các record profile pha profiling / 1e9.
Kiểm chứng trên 6 cell c16 (có cả hai nguồn):

| cell | offset | post-offset | health | cut theo scrape (bỏ) | cut theo profile-start (bỏ) | khớp |
|---|---|---|---|---|---|---|
| lpm/c16 | 2 | 195 | 0 | 14 | 14 | YES |
| dfs-weight/c16 | 2 | 174 | 0 | 14 | 14 | YES |
| fcfs/c16 | 2 | 166 | 0 | 14 | 14 | YES |
| fcfs-match/c16 | 2 | 167 | 0 | 14 | 14 | YES |
| hrrn/c16 | 2 | 193 | 0 | 14 | 14 | YES |
| boost/c16 | 2 | 178 | 0 | 14 | 14 | YES |

Hai cut bỏ đúng cùng 14 record warmup trên cả 6 cell (health check nằm
trong phần offset cắt trước). Fallback dùng cho 12 cell c4/c8; cell c16
dùng scrape (nguồn gốc) — cùng kết quả.

Lưu ý population frozen: cell c4 `lpm/c4_20261007_022137` khi chạy thật
có cut theo scrape (bỏ 3), hôm nay tính lại không export → không cut
(giữ 227, cold 10 vs frozen 7). Fallback cut theo profile-start bỏ lại
đúng 3 → giữ 224, cold 7 — **phục hồi đúng population frozen**. Giá trị
frozen giữ nguyên; v2 dùng fallback cut cho cell đã mất export, nguồn
cut khai báo trong `e2_metrics_v2.json` (`warmup_cut_source`).

## 5. Kết luận audit → quyết định cho bộ v2

- **(a)** Bảng v2 dùng percentile queue từ eval log per-request; histogram
  chỉ lưu trong `e2_metrics_v2.json` làm cross-check (bằng chứng §3).
- **(b)** Không ghi đè `e2_metrics.json` — tính lại cell c4 hôm nay mất
  khối queue và đổi population warmup (bằng chứng đầu trang).
- **(c)** Jain v2 tính trên `queue_wait_s` (P_wait), không trên TTFT —
  TTFT trộn thời gian prefill biến thiên hợp lệ vào "unfairness".
- **(d)** Starvation theo ngưỡng cố định (0.2 s ở c4/c8, 10 s ở c16) và
  cold/warm split không vào bảng v2; chỉ tính và lưu trong
  `e2_metrics_v2.json` để đối chiếu.
- **(e)** Không join được rid ↔ x_request_id → hai population tường minh:
  `P_done` (profile, profiling phase, bỏ warmup/error) cho n/TTFT/ITL/
  tok/s; `P_wait` (eval, offset + cut + bỏ health) cho cache hit/queue/
  Jain/stretch/worst-5%. Khác biệt n của hai population là request còn
  in flight khi đóng cửa sổ (+ record lỗi đơn vị) — khai báo trong output.
- **(f)** ITL v2 = phân phối per-chunk từ mảng `inter_chunk_latency`
  pooled toàn bộ chunk của P_done (mục 2.4); nhánh TPOT-based chỉ là
  fallback khi mảng vắng, ghi nhãn `itl_source`.
- **(g)** tok/s v2 = tổng output tokens của P_done / cửa sổ đo
  (profiling window: `request_start_ns` đầu → `request_end_ns` cuối của
  P_done) — khai báo cửa sổ trong output.
