# E3 Plan - KV/host-tier-aware placement cho dynamo kv-router (DualMap)

*(Đính chính 23/10: đổi tên từ "Size-aware placement" — dynamo 1.4.2 **đã có** term
size/uncached trong cost, xem §1. Điểm thật sự thiếu là **chi phí reload tầng
host + độ chật KV**. Bản cũ của plan này dựng trên kết luận "size-blind
placement" — kết luận đó đã bị đính chính 20/10 ở
`ph2lio/experiments/e2/docs/profile_bottleneck_5runs.md` §4.)*

Giữ dynamo-router, contribution hướng dynamo repo. Phát triển từ profile
bottleneck: `ph2lio/experiments/e2/docs/profile_bottleneck_5runs.md` §4 (đính
chính 20/10) + §7 (xác nhận 10 run & 5 cell NSys, 22/10) + §8 (delayer
outcomes, 23/10).

## 1. Problem (đã đính chính — bản cũ "argmin state, không xét chi phí request đang đặt" là SAI)

Dynamo 1.4.2 kv-router **có** tính chi phí request đang đặt. Cost tái dựng được
từ dữ liệu 5/5 cell NSys:

```
score = state + 16 × (input_len − device_matched − 0.75 × host_matched) / 64
```

(đơn vị block, page 64). Kiểm chứng hành vi: corr(logit, công thức này) =
**+0,877–0,897** (state-only chỉ +0,519–0,611) — khớp source tag v1.4.2
`lib/kv-router/src/scheduling/selector.rs`. Số liệu per-cell
`/shared/loipp/logs/retrace_<stamp>.txt` (phần A3/B); reproduce
`scripts/focus2_retrace.py --part formula <run_dir>`.

Vấn đề thật (nhất quán 10/10 run, 5/5 cell NSys):

- **Ghim cấu trúc**: workload AgentX lặp history (matched ≈ 0.9×input; cached
  94,5–98,3k token/request) → credit cached ≈ 16×96k/64 ≈ 24k block, lớn hơn
  hẳn chênh lệch state giữa các rank → argmin gần như luôn trả về rank giữ
  prefix: **89–91% request bị ghim** → router gần hết đòn bẩy cân bằng cho
  ~90% traffic.
- **Mù chuỗi KV/host-tier** — điểm sửa chính của E3:
  - host-hit được **credit 0,75** (`lib/kv-router/src/scheduling/config.rs`
    default) — coi reload gần như miễn phí; thực đo: request trúng host chờ
    **21,2–25,7 s** (thu hồi chỗ + nạp lại toàn bộ span) vs trúng device
    **1,80–2,42 s** (8–14×); càng chờ lâu càng theo kích thước span (host
    ~132k → 34,8 s; ~5k → 11,2 s);
  - không có term độ chật KV → rank nóng vẫn bị dồn tiếp tới KV 1.00 (chạm ở
    6/10 run; retract 9 lần/5 run) → write_back đẩy prefix xuống host tier
    (cuối window rank nóng dùng 37–100% tầng host 128 GB/rank);
  - weight 16 có vẻ underestimate chi phí prefill (nghi vấn §2.5 focus report
    từng run).
- Hệ quả định lượng: active prefill token lệch **3,9–10,8×**/8 rank; KV spread
  p50 0,50–0,57; queue-wait per-rank max **8,4–23,2 s** vs min 2,1–4,9 s; nhóm
  trúng host chỉ 14–18% request nhưng chiếm **65,7–70,9% tổng giây chờ** →
  nguồn chính TTFT p99 59,9–99,4 s.
- **Vòng phản hồi (2)→(1) đo được trực tiếp** (§8, 23/10): mất cân bằng rank
  (focus 2) khiến prefill-delayer giữ prefill ở **65–67% số pass** (100% từ
  nhánh mixed; mọi lần chờ đủ mũ 30 pass ≈ 2,1–2,4 s) → bottleneck scheduling
  được nuôi bằng bottleneck routing.

Sửa **flow tại nguồn** thì stock (KV) tự đều theo sau — giữ định hướng này.

## 2. Direction

| Nguồn | Đóng góp cho cơ chế |
|---|---|
| **DualMap** (ICLR'26) | `TTFT(r,i) = T_q(r,i) + T_c(r,i)` - `T_c` = thời gian compute **của request r** trên candidate i, tính từ số token reusable query được; switch affinity→load khi TTFT dự báo vượt SLO; 2 home bằng dual-hash (PoTC) |
| **Preble** (OSDI'24) | cost = tải gần đây + eviction + **missed-token cost của request mới** |
| **PoTC** (Mitzenmacher) | chọn giữa 2 candidate giảm max-load ~loglog - cơ sở cân bằng của dual-candidate |
| **sgl-router reorg** (upstream 10/2026) | rank theo `uncached_tokens = input − matched` per candidate (`experimental/sgl-router/src/policies_reorg/cache_aware.rs:136`) - thiết kế đã được chấp nhận ở production khác — **cùng họ với term uncached mà dynamo 1.4.2 đã có** (§1), củng cố việc giữ term này |
| Forensics | chứng minh mode thất bại **trên workload AgentX**, các paper không có |

**Note (đính chính 23/10)**: term chi phí uncached **đã có sẵn** trong công thức
dynamo (weight 16×block) — đóng góp của E3 không phải "bổ sung T_c còn thiếu"
như bản cũ ghi, mà là **đưa chi phí thật vào cost**: (a) thay credit 0,75 của
host-hit bằng chi phí reload T_reload ≈ 15–28 s (đo được, §1), (b) thêm term độ
chật/headroom KV. Machinery tính cả 2 term **đã tồn tại** trong stack: kv-events
index mỗi request mỗi worker (thứ sinh `cache_credits`/`host_pinned_blocks`,
`_core.pyi:1926`) — không cần hạ tầng mới.

## 3. Implementation (trên dynamo-router)

Công thức đề xuất (đơn vị block, đồng nhất logit hiện tại; so với bản cũ:
**không thêm term uncached — nó đã có weight 16** — sửa 2 chỗ sai của cost):

```
score' = state + w_u × (input − device_matched)/64    # giữ term uncached, BỎ credit 0.75×host
       + w_r × host_matched/64                        # host-hit: PHẠT theo chi phí reload (15–28 s) thay vì credit
       + w_k × kv_occupancy(rank)                     # né rank gần đầy KV (headroom)
```

Baseline hiện tại tương ứng: `w_u = 16`, host credit −0,75×16, `w_k = 0`.

Phân tầng (mỗi tầng benchmark riêng):

- **T1 - cost thật**: (a) hiệu chỉnh `w_u` (replay offline chọn giá trị);
  (b) đảo credit host thành phạt `w_r` theo T_reload đo được; (c) thêm `w_k`
  (dữ liệu độ chật KV đã có trong state router đẩy lên mỗi scrape).
- **T2 - candidate giới hạn theo prefix-holder + SLO-switch**: sinh candidate
  từ **index kv-events sẵn có** (prefix-holder set kiểu sgl-router
  `cache_candidate_min/max_workers`), giữ affinity home tới khi pending-tokens
  vượt ngưỡng SLO (DualMap: `ttft_slo_threshold` = max pending prefill tokens
  xử lý được trong SLO). T2 giờ còn là cân đối trực tiếp với ghim cấu trúc (89–91%
  bị ghim, §1): khi holder đang nóng, candidate thứ 2 nhận tải.
- **Dual-hash của DualMap bị loại khỏi scope E3**: nó là giải pháp *index-free*
  để sinh 2 candidate - stack đã có index nên candidate từ index chính xác hơn
  (biết residency thật + eviction). Hash hiện có của dynamo toàn bộ là
  **identity hash** (đối chiếu prefix), không phải placement hash: engine tính
  block-hash chuỗi SHA256 theo nội dung (`srt/mem_cache/utils.py:166-175`,
  parent-hash + tokens - cùng prefix ⇒ cùng hash trên mọi worker), router hash
  prefix request bằng XXH3 (`public-xxh3-v1`, hoặc `keyed-xxh3-v1` + key 32-byte
  cho tenant isolation - `_core.pyi:1931,1956-1961`) để tra index ra
  `cache_credits`/`host_pinned_blocks`. Nơi đặt = argmin score, không có hash
  tham gia.
- **T3 - hotspot migration** (phase sau): di request đang chờ sang home kia theo
  benefit `B = TTFT_i − TTFT_j > 0` (DualMap Eq. 6); dynamo đã có hạ tầng
  migration (tắt, limit 0 - `frontend/main.py:343`).

Chỗ cắm: **dynamo 1.5.1 custom WorkerScorer** (Python policy, 1.5.0 GA có API
này) - dựng từ wheel public, không cần registry creds team. Fallback nếu wrapper
1.5.1 không tương thích sglang branch ta: giữ 1.4.2, đo T1/T2 bằng **offline
replay** trước (dynamo pkg có sẵn module `replay/`).


## 4. Kết quả kì vọng (baseline = 10 run đã xác nhận 22/10)

| Chỉ số | Hiện tại (10 run) | Kì vọng |
|---|---|---|
| corr(logit, TTFT) | +0,26–0,52 | → ~0 (cost phản ánh chờ thật) |
| Request bị ghim (pinned) | 89–91% | giảm khi holder nóng (T2) |
| qw request trúng host | 21,2–25,7 s | tiệm về 2–4 s (gần mức trúng device) |
| TTFT p99 | 59,9–99,4 s | **kỳ vọng GIẢM** (đổi kỳ vọng cũ "không kỳ vọng"): 65,7–70,9% tổng giây chờ là chuỗi host-tier — router nhìn thấy qua `host_pinned_blocks` nên sửa được ở nguồn; phần delayer giữ chỉ ~2,1–2,4 s có giới hạn (§8) |
| Prefill token spread | 3,9–10,8× | ≤ 2× |
| KV spread p50 / chạm 1.00 | 0,50–0,57 / 6 run-10 | hẹp hơn; retract (9 lần/5 run) → hiếm/0 |
| Queue-wait skew max/min rank | 8,4–23,2 s / 2,1–4,9 s | thu hẹp |
| Pass bị delayer giữ (mixed-share, §8) | 65–67% | giảm — thước đo trực tiếp vòng phản hồi (2)→(1) |
| Cache reuse | 87,0–88,0% | giữ nguyên (cost vẫn cache-aware, không đánh đổi affinity khi rank còn khỏe) |
| Throughput CCU64 | 520–546 | trung tính (GPU 77–98%) |

Tham chiếu chuyển tiếp: DualMap +2,25× capacity @5s SLO; #26611 (sglang) 5,7×
TTFT same-prefix chỉ riêng affinity.

## 5. Định hướng các bước

- **S0 - spike (0 GPU)**: dựng venv dynamo 1.5.1 public wheel trên login node
  (CPU): `pip install ai-dynamo==1.5.1 ai-dynamo-runtime==1.5.1` → import + đọc
  API WorkerScorer thực tế (release notes → code thật); smoke stack đầy đủ để
  dành cửa sổ GPU. Đồng thời xin team path/creds image `1.5.0-kvoverlap1` (đã
  verify: patch chỉ thêm metric log).
- **S1 - offline replay (0 GPU)**: mô phỏng `score'` trên request stream +
  gauges của 10 run cũ → xác nhận hướng & chọn `w_u`/`w_r`/`w_k` trước khi tốn
  GPU. Khung đã có sẵn: `scripts/focus2_retrace.py --part formula` (A3/A4/B) —
  mở rộng thành các biến thể score' rồi đo lại pinned/qw/spread như retrace.
- **S2 - A/B serving**: baseline vs T1 vs T2 trên harness e2
  (start_dynamo_stack.sh + aiperf + `compare_focus_runs.py`), CCU64, seed 42,
  cold cache, ≥3 run/mode (retract intermittent nên 1 run không đủ). Thêm 2 thước
  đo A/B riêng của E3: pinned % (join frontend log ↔ schedule_eval) và
  mixed-share delayer (`scripts/delayer_outcomes_scan.py`, §8).
- **S3 - phân tích + doc + PR dynamo**: story cập nhật theo đính chính —
  "dynamo tính credit cached từ 1.4.x **và đã có term uncached (weight 16)**;
  cái thiếu là chi phí reload host-tier + độ chật KV — bằng chứng sản xuất
  (10 run, 5 cell NSys) + sửa + A/B".

## 6. Phối hợp e2 ↔ e3 (attribution + tranh GPU)

E2 = scheduling policy (sglang engine), E3 = routing (dynamo router) - code gần
như không giao nhau, nhưng benchmark thì lẫn. Luật:

- **Freeze-and-vary**: E3 chỉ biến thiên router, engine đóng băng (mọi report
  ghi 1 dòng config-freeze: sglang commit + policy + flags + dynamo version). E2
  chỉ biến thiên policy, router giữ 1.4.2 mặc định. 2×2 hợp nhất (best policy ×
  best router) để sau khi cả hai có A/B riêng.
- **Code**: E3 không commit lên branch làm việc của e2; scorer + scripts nằm
  trong `e3/`; script start stack copy bản riêng (scripts đọc fresh từ /shared
  mỗi boot cell - sửa chung phá job e2 đang chạy). Nếu cần thêm metric engine →
  feature branch riêng + merge window thỏa thuận với e2.
- **GPU cao điểm**: S0 + S1 đều 0-GPU (login/srun CPU), làm ngay không cần cửa
  sổ; serving A/B gom 1 job (3 mode × 3 run × ~40 phút ≈ nửa ngày), phối hợp
  queue e2.

## 7. Rủi ro

- ~~Patch `kvoverlap1` của team chưa biết nội dung~~ **Đã verify với team
  (20/10/2026): patch chỉ thêm metric log, không đụng router cost/scoring** → T1
  không trùng.
- ~~"Công thức dynamo thiếu term uncached"~~ **Đã đính chính (§1)**: term có sẵn
  (weight 16), corr +0,877–0,897 — story PR phải nói đúng chỗ thiếu là reload
  cost + KV headroom, không nói "thiếu size term".
- Wheel 1.5.1 ↔ wrapper `dynamo.sglang` ↔ branch sglang ta: compat chưa verify
  (S0 giải).
- Biến thiên benchmark: triệu chứng retract intermittent (9 lần/10 run) → cần
  ≥3 run/mode để không kết luận từ nhiễu.
