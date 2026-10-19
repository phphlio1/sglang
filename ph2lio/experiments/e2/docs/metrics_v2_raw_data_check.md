# Nguồn dữ liệu thô có chính xác không? — findings 2026-10-08

Câu hỏi sau khi bảng v2 (`docs/metrics_v2.md`) cho ra hai con số trông "bất
thường": c4/c8 queue p95 ~0.002 s nhưng p99 ~2.4–3.7 s (chênh ~1000 lần),
c16 queue p99 112–281 s. File này ghi lại các kiểm tra trả lời câu hỏi
**gốc**: các file thô mà metric đọc từ đó có tin được không — và hai "bất
thường" kia là lỗi dữ liệu hay hiện tượng thật.

## 1. Bốn nguồn thô, ba đường đo độc lập

| Nguồn | Ai ghi | Dùng cho |
|---|---|---|
| `schedule_eval.jsonl` | server (schedule-eval logger, ghi lúc admission) | queue_wait_s, ttft_s, input_len, matched prefix |
| `profile_export.jsonl` | client benchmark (AIPerf, ghi lúc request xong) | TTFT, ITL per-chunk, osl, start/end ns |
| `server_metrics_export.jsonl[.gz]` | scrape Prometheus `/metrics` của server | histogram queue_time, gauge `sglang:num_queue_reqs` |
| `e2_metrics.json` (frozen v1) | pipeline v1 chạy lúc sweep | cross-check |

Ba đường đầu đo cùng một hệ thống bằng code hoàn toàn khác nhau. Nếu một
nguồn sai, các kiểm tra chéo dưới đây sẽ vỡ — chúng không vỡ (trừ đúng một
defect đã biết, §4.1).

## 2. Kiểm tra độc lập: các nguồn nhất quán với nhau

### 2.1 Little's law khớp gauge queue-depth của server — 18/18 cell

Nếu `queue_wait_s` của eval log bị phóng đại/méo do bug, đồng nhất thức
L = W·n/T (mean wait × n request / cửa sổ profiling) sẽ lệch khỏi độ sâu
hàng đợi trung bình mà **server tự đo bằng gauge riêng** (`sglang:num_queue_reqs`,
trung bình trên ~5.5k scrape pha profiling, mỗi cell):

| Cell | n wait | mean wait (s) | W·n/T | gauge |
|---|---|---|---|---|
| lpm/c4 | 224 | 0.066 | 0.008 | 0.006 |
| hrrn/c4 | 216 | 0.092 | 0.011 | 0.011 |
| boost/c4 | 217 | 0.098 | 0.012 | 0.011 |
| fcfs-match/c4 | 223 | 0.084 | 0.010 | 0.008 |
| fcfs/c4 | 217 | 0.039 | 0.005 | 0.004 |
| dfs-weight/c4 | 222 | 0.158 | 0.019 | 0.019 |
| lpm/c8 | 139 | 0.128 | 0.010 | 0.009 |
| hrrn/c8 | 142 | 0.044 | 0.003 | 0.003 |
| boost/c8 | 138 | 0.107 | 0.008 | 0.007 |
| fcfs-match/c8 | 143 | 0.086 | 0.007 | 0.006 |
| fcfs/c8 | 146 | 0.084 | 0.007 | 0.005 |
| dfs-weight/c8 | 139 | 0.061 | 0.005 | 0.005 |
| lpm/c16 | 181 | 18.969 | 1.905 | 2.097 |
| hrrn/c16 | 179 | 20.059 | 1.979 | 2.114 |
| boost/c16 | 164 | 21.916 | 1.977 | 2.110 |
| fcfs-match/c16 | 153 | 22.543 | 1.951 | 2.106 |
| fcfs/c16 | 152 | 22.764 | 1.904 | 2.119 |
| dfs-weight/c16 | 160 | 22.638 | 2.021 | 2.276 |

Hai đường đo độc lập (client-ghi wait tổng hợp lại vs server gauge) khớp
ở cả hai bậc độ lớn: c4/c8 ~0.005–0.019, c16 1.9–2.0 vs 2.1–2.3. Chênh ~10%
ở c16 là kì vọng của Little's law rời rạc (gauge scrape mỗi ~0.33 s, cửa
sổ đóng còn request in flight).

### 2.2 Percentile eval log vs histogram server — sai lệch chỉ ở độ phân giải bucket

Bảng cross-check đầy đủ trong `docs/metrics_v2.md` §3. Tóm lại: c16 p95
khớp ±1.3% (bucket giữa còn mịn); lệch lớn chỉ xuất hiện ở đúng chỗ bucket
quá rộng (5 ms ở c4/c8 p95; 50–100 s ở c16 p99) — đặc điểm của resolution,
không phải của dữ liệu sai.

### 2.3 Ràng buộc vật lý ttft >= queue_wait — 0 vi phạm / 18 cell

Service không thể bắt đầu trước khi được admission. Trên mọi P_wait record
có đủ cả hai trường của 18 cell chuẩn: **0 record** có `ttft_s < queue_wait_s`.

### 2.4 AIPerf aggregate == phân phối per-request (audit §2.3)

Percentile `inter_token_latency` tổng hợp của AIPerf trùng phân phối TPOT
per-request tính từ `profile_export.jsonl` đến **12 chữ số** (p95
168.140039675861) — hai đường code khác nhau của cùng client.

### 2.5 Determinism

Tính lại pipeline v1 in-memory trên cell c16 còn đủ nguồn → tái lập
`e2_metrics.json` frozen **tuyệt đối**. Nguồn thô không đổi giữa hai lần đọc.

### 2.6 Warmup cut fallback (audit §4)

Cell mất `server_metrics_export.jsonl` (gzip policy) dùng cut =
`request_start_ns` đầu tiên của P_done: bỏ đúng cùng 14 record trên 6/6
cell c16 so với cut gốc (scrape đầu pha profiling) và phục hồi đúng
population frozen của cell c4.

## 3. Hai "bất thường" là hiện tượng thật, đọc trực tiếp từ raw

### 3.1 c4/c8 p95 ~ms nhưng p99 ~giây: phân phối bimodal thật

Raw wait đã sort của lpm/c4 (n=224), 12 giá trị lớn nhất (s):
`6.293, 3.725, 2.594, 1.933, 0.005, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002`.
Chỉ **4/224 request** chờ > 1 s; median của phần còn lại **1.35 ms**.
p95 (rank ~12 từ trên) rơi vào cụm ms; p99 (rank ~3) rơi vào cụm giây —
vách đá giữa hai rank liền nhau là thật. Nguyên nhân: head-of-line
blocking — đối chiếu arrival của 4 request chờ lâu nhất, mỗi request đến
1.5–5 s sau khi một request khác có 23–34k token chưa cache được admission
(vd wait 6.29 s đến 1.7 s sau prefill 33,596 uncached token); bản thân
chúng gần như warm (matched 65k/65.7k, chỉ cần 424–2,749 token riêng)
nhưng phải xếp hàng sau prefill của request khác. Histogram của server
(đường đo thứ ba) cũng đọc p99 2.76–4.78 s cùng bậc — không phải bug của
eval log.

### 3.2 c16 wait tới 306 s: band bão hòa thật

Mỗi cell c16 nạp 14.3–18.7 triệu input token trong 30 phút (lpm/c16:
18,670,125 token trên 181 admission, ~10.4k input token/giây; fcfs/c16 thấp
nhất 14.3M, ~7.9k/giây) — GPU chạy kín suốt cửa sổ. Các wait lớn nhất
đứng sau các prefill khổng lồ chưa cache, đọc thẳng từ record eval:

- lpm/c16: wait 306.3 s — input 59,528 token, matched 0 (toàn lạnh)
- lpm/c16: wait 152.6 s — input 233,976, matched 65,536 → 168,440 uncached
- lpm/c16: wait 126.9 s — input 130,238, matched 24,710 → 105,528 uncached
- boost/c16: wait 121.9 s — input 94,237, matched 84,277

Little's law (§2.1) xác nhận tổng khối lượng chờ tương ứng với queue-depth
gauge ~2.1 của server suốt cửa sổ. c16 là band load-binding đã được ghi
nhận từ thiết kế sweep.

## 4. Khiếm khuyết thực sự tìm thấy trong nguồn thô

Đây là danh sách đầy đủ những gì KHÔNG tin được — đã biết, đã xử lý:

1. **fcfs pre-fix: matched prefix toàn 0 (defect thật, đã fix)**. Cell
   fcfs 2026-10-07 (vd `fcfs/c4_20261007_033335`): `sum(matched)=0` trên
   12,908,229 input token — cache hit 0.00% là **lỗi bookkeeping admission
   của fcfs**, không phải thực tế. Fix ở commit `bdc67a5720`; cell c16
   2026-10-08 (post-fix) đọc 78.49% thật. Bảng v2 ghi chú tại
   `docs/metrics_v2.md` §4.
2. **Histogram bucket thô**: 5 ms ở c4/c8 (p95 vô dụng, −40..−68%),
   50–100 s ở đuôi c16 (p99 +13..+33%). Lý do bảng v2 chuyển sang eval log
   per-request; histogram chỉ còn vai trò cross-check.
3. **12 cell c4/c8 mất `server_metrics_export.jsonl`** vì policy gzip —
   file vẫn đọc được qua `.gz` (chỉ đọc, không giải nén); warmup cut dùng
   fallback đã kiểm chứng (§2.6).
4. **`rid` (eval log) và `x_request_id` (profile) không join được** —
   giao tập hợp 0. Hai population khai báo tường minh, không metrics nào
   trộn hai nguồn trong một công thức.
5. **17 record in-flight-at-close ở c16** (có `queue_wait_s`, `ttft_s=None`,
   wait 72–156 s): request còn xếp hàng khi cửa sổ đóng — không phải dữ
   liệu sai; stretch đếm riêng `n_dropped_no_ttft`.

## 5. Kết luận

Nguồn thô chính xác ở mọi metric trong bảng v2, được chứng minh bằng 6
kiểm tra độc lập (§2) với ba đường đo không chung code. Đúng một defect
thật đã được tìm thấy trong nguồn thô (fcfs matched bookkeeping pre-fix,
§4.1) — đã fix và đã ghi chú. Hai "bất thường" là hiện tượng load thật:
bimodal head-of-line blocking ở c4/c8 và bão hòa GPU ở c16.

## 6. Chạy lại các kiểm tra

```bash
# Little's law + ràng buộc ttft>=wait, 18 cell chuẩn (đọc .gz chỉ-đọc)
python3 analysis/raw_data_check.py
# percentile eval vs histogram + warmup cut: docs/metrics_v2_audit.md §3–4
python3 analysis/metrics_v2.py --all
python3 analysis/print_metrics_v2_table.py
```
