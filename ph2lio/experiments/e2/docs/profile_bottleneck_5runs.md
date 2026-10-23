# [Report] - Profiling Bottleneck trên config Prod GLM5.3 (13/10/2026)

So sánh thông số Engine SGLang để profile bottleneck giữa 5 run có cùng cấu hình Production GLM5.3, workload AgentX

*(Cập nhật 20/10: §2/§3/§4 đính chính sau khi kiểm chứng lại — kết luận "size-blind placement"
trước đây sai, vấn đề đúng là chuỗi dung lượng KV/hicache host tier, xem §4.)*

`BASE_DIR = /shared/loipp/sglang/ph2lio/experiments/e2/results/glm53/agentx_pilot_preprod/agentx_dynamo/profiling/dfs-weight/`

| Run | Stamp | Run dir |
|---|---|---|
| 1 | c64_20261014_112105 | `$BASE_DIR/c64_20261014_112105` |
| 2 | c64_20261015_133031 | `$BASE_DIR/c64_20261015_133031` |
| 3 | c64_20261015_144717 | `$BASE_DIR/c64_20261015_144717` |
| 4 | c64_20261015_201358 | `$BASE_DIR/c64_20261015_201358` |
| 5 | c64_20261015_224930 | `$BASE_DIR/c64_20261015_224930` |

## 1. Bảng so sánh thông số giữa các run

| Chỉ số | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Consistent? |
|---|---|---|---|---|---|---|
| Request/output throughput (tok/s) | 522 | 532 | 531 | 520 | 528 | ✓ |
| TTFT mean (s) | 7.1 | 7.6 | 6.8 | 8.2 | 8.2 | ✓ |
| TTFT p99 (s) | 75.4 | 89.7 | 76.0 | 99.4 | 96.9 | ✓ |
| Running-req mean/rank (min–max) | 2.75–3.75 | 2.31–4.68 | 2.60–3.93 | 2.55–4.50 | 2.45–3.66 | ✓ |
| Pending backlog per-minute (min–max) | 5.970–132.265 | 5.443–99.600 | 5.886–73.559 | 5.273–78.131 | 11.324–83.241 | ✓ |
| Active prefill token spread | 5.6 | 5.1 | 4.5 | 3.9 | 5.0 | ✓ |
| KV usage mean (min–max) | 0.58–0.82 | 0.47–0.78 | 0.57–0.76 | 0.55–0.74 | 0.56–0.79 | ✓ |
| KV usage spread p50 | 0.56 | 0.56 | 0.51 | 0.57 | 0.51 | ✓ |
| #rank chạm KV 1.00 | 4 | 0 | 3 | 0 | 0 |  |
| #retract | 3 | 0 | 2 | 0 | 0 |  |
| corr(logit, TTFT) | +0.40 | +0.26 | +0.52 | +0.37 | +0.42 | ✓ cùng dấu |
| Cache reuse engine (%) | 87.5 | 88.1 | 87.9 | 87.0 | 87.5 | ✓ |
| Cached tokens/request | 96111 | 96206 | 96302 | 95644 | 96645 | ✓ |
| EAGLE accept len (per-minute min–max) | 1.97–2.31 | 2.00–2.35 | 1.91–2.45 | 1.98–2.43 | 1.99–2.55 | ✓ |
| ITL mean (ms) | 42.0 | 37.7 | 42.8 | 41.5 | 40.0 | ✓ |
| GPU util window (min–max %) | 79–97 | 77–97 | 79–96 | 81–97 | 77–97 | ✓ |

## 2. Mất cân bằng workload/KV per rank

Thống kê per rank trong benchmark window (hàng workload = min–max mean qua 8 rank, hàng join = rank cực đại vs cực tiểu)

| Chỉ số (per rank) | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|---|---|---|---|---|---|
| Active prefill tokens | 3.8k–21.6k (5.6×) | 4.3k–21.5k (5.1×) | 3.5k–15.9k (4.5×) | 5.3k–21.0k (3.9×) | 5.3k–26.5k (5.0×) |
| Active decode blocks | 5.3k–7.5k (1.4×) | 4.2k–7.4k (1.8×) | 4.8k–7.6k (1.6×) | 5.0k–6.9k (1.4×) | 5.0k–7.4k (1.5×) |
| KV token usage `full_token_usage` | 0.58–0.82 | 0.47–0.78 | 0.57–0.76 | 0.55–0.74 | 0.56–0.79 |
| Cache hit rate | 0.53–0.76 | 0.58–0.80 | 0.41–0.77 | 0.56–0.75 | 0.42–0.78 |
| Admission queue-req | 0.09–0.56 | 0.10–0.92 | 0.11–0.54 | 0.17–0.58 | 0.14–0.71 |
| Queue-wait mean/rank (rank max vs min) | r0 9.6 s / r7 2.4 s | r7 10.5 s / r6 3.3 s | r5 8.7 s / r0 2.5 s | r1 11.2 s / r2 4.9 s | r2 13.0 s / r1 2.8 s |
| TTFT mean/rank (rank max, p99 của nó vs rank min) | r0 11.2 s (p99 119) / r3 4.0 s | r7 14.7 s (p99 213) / r3 5.2 s | r5 11.6 s (p99 119) / r0 4.9 s | r1 15.7 s (p99 267) / r2 7.3 s | r2 16.5 s (p99 151) / r1 5.3 s |
| Request count per rank (min–max) | 72–221 | 70–270 | 77–279 | 79–226 | 113–217 |
| #rank chạm KV 1.00 | 4 | 0 | 3 | 0 | 0 |
| #retract | 3 | 0 | 2 | 0 | 0 |

- **Router overhead - không đáng kể**: 1172 dec/4.8 ms, 1216 dec/12.1 ms, 1221 dec/9.6 ms, 1206 dec/7.4 ms, 1188 dec/13.3 ms - quá nhỏ so TTFT giây.
- **Công thức cost - đầy đủ, đã kiểm chứng lại (20/10)**: corr(logit, TTFT) = +0.40, +0.26, +0.52, +0.37, +0.42. Logit **có** chứa chi phí request đang đặt: corr(logit, state + 16×uncached/64) = +0.88–0.90 mọi run (state-only chỉ +0.50–0.60). corr(logit, input_len) thấp (+0.26–0.32) không phải do thiếu term mà vì input_len gần như hằng số trong workload lặp history (matched ≈ 0.9×input_len) — đính chính chi tiết ở §4.

## 3. Focus 1 - scheduling policy dfs-weight dẫn tới batch mỏng và đói prefill

- Batch decode mỏng mọi run (running-req mean ≤ 4.68/rank).
- Backlog prefill duy trì mọi run (đỉnh per-minute 73,559–132,265 token).
- Throughput chênh ≤ 2%, TTFT p99 chênh ≤ 32%.

**Nguyên nhân**:

- Cặp `--schedule-policy=dfs-weight` + `--enable-prefill-delayer` bảo vệ
  request đang decode (prefill chen ngang làm ITL giật) → scheduler hoãn
  prefill mới → backlog `#pending-token` duy trì suốt benchmark  dù admission queue gần rỗng (queue-req
  0.09–0.92/rank). TTFT tail 75–99 s đến từ backlog này, không phải cạn tài nguyên - GPU
  vẫn 77–97%. (Đính chính 20/10: thành phần lớn nhất của backlog/TTFT tail là request chờ
  load-back host-tier — ~62% tổng thời gian chờ, phân tích ở §4; phần delayer giữ là phần còn lại.)
- Batch decode mỏng 2.31–4.68 req/rank
  (dù trần `--max-running-requests=256`): prefill bị hoãn nên không có việc dồn vào batch.
- **Tóm lại:** scheduling policy đánh đổi TTFT lấy ITL ổn định. Ở CCU 64 nó giữ batch mỏng kéo dài suốt window nên là bottleneck là hành vi cố ý của policy.

**Verdict:** Consistent giữa 5 run

## 4. Focus 2 - mất cân bằng per-rank: chuỗi dung lượng KV (hicache host tier), router không nhìn thấy

> **Đính chính (20/10):** kết luận cũ của mục này ("size-blind placement" — logit thiếu chi phí
> request đang đặt) là **sai**. Test cũ dùng corr(logit, input_len): trong workload lặp history,
> matched ≈ 0.9×input_len nên input_len gần như hằng số, không thể mang tín hiệu. Test đúng với
> unc = input_len − cached cho corr +0.88–0.90 (state-only +0.50–0.60) → binary dynamo 1.4.2
> **đã có đủ** term chi phí request trong logit, khớp source tag v1.4.2
> (`lib/kv-router/src/scheduling/selector.rs`). Công thức tính cost không thiếu gì.

**Vấn đề đúng (nhất quán 5/5 run):**

- **~90% request là lượt mới của hội thoại cũ**, bị ghim về rank đang giữ prefix: credit
  ≈ 16×96k/64 ≈ 22k block, lớn hơn hẳn chênh lệch trạng thái giữa các rank → argmin luôn trả về
  holder. Cache-aware routing hoạt động đúng, nhưng router gần như không còn đòn bẩy cân bằng
  cho 90% traffic.
- **Rank sở hữu nhiều hội thoại đầy KV device** → hicache write_back
  (`--enable-hierarchical-cache --hicache-size=128 --hicache-write-policy=write_back`) đẩy prefix
  cũ xuống host tier; cuối window rank nóng dùng 83–100% host tier (3/5 run chạm ~100%).
- **Hội thoại quay lại mà prefix đã nằm trên host** → engine phải reclaim + load-back toàn bộ
  span lên device **trước khi admit** (`python/sglang/srt/managers/schedule_policy.py`
  `prepare_load_back`, `python/sglang/srt/mem_cache/prefill_budget.py` `reclaim_for_prealloc`) →
  chờ queue ~15–28s, gấp ~8–14× request cùng cỡ nhưng prefix còn trên device (~2s). Confounder
  đã loại bằng: so sánh trong cùng rank/cùng window vẫn chênh lệch đó; request lớn (input_len
  ≥120k) không host-hit vẫn chỉ chờ ~2–4s.
- **Nhóm host-hit chỉ ~14% request nhưng chiếm ~62% tổng thời gian chờ queue**; đa số request
  chờ >30s thuộc nhóm này → nguồn chính của TTFT p99 75–99s. Spread "active prefill" 3.9–5.6×
  phần lớn là bóng của backlog này: gauge chỉ drain khi token đầu tiên quay về router
  (`lib/llm/src/kv_router/push_router/request_guard.rs`), nên backlog load-back hiện thành
  mất cân bằng trên metric router.
- **Router mù trước toàn bộ chuỗi**: logit không có term KV-occupancy hay chi phí reload;
  host-hit còn được credit 0.75 (`lib/kv-router/src/scheduling/config.rs` default) — coi reload
  gần như miễn phí. KV-full + retract (2/5 run, tổng 5 request) chỉ là biên hiếm của cùng áp lực
  dung lượng, không phải vấn đề chính.

**Verdict:** Consistent giữa 5 run. Hướng sửa (E3, cập nhật e3_plan.md sau): đưa chi phí thật
của host-tier hit và headroom KV vào quyết định định tuyến.

## 5. Bottleneck khác ngoài phạm vi Scheduling (maybe future work?)

Các hiện tượng dưới đây có đo được trong 5 run nhưng **không thuộc** phạm vi scheduler/routing - ghi lại để theo dõi sau:


- **EAGLE accept len** 1.91–2.55
  (per-minute mean): vài phút dưới ngưỡng khỏe ≥2.0 - spec decode chưa là
  nghẽn chính (ITL ~38–43 ms ổn định) nhưng đáng theo dõi.
- **Thước đo còn hở**: `sglang:fwd_occupancy` thỉnh thoảng bị drop mẫu NaN
  trong export; client không report `usage.prompt_tokens_details.cached_tokens`
  (cần `--enable-cache-report` phía server); `sglang:prefill_delayer_outcomes`
  đã được phân tích chi tiết ở §8 (23/10, sau khi extract giữ đủ label).
- **MoE expert imbalance**: EP=8 → mỗi bước forward, token mọi rank all-to-all tới đúng expert, nếu 1 expert nóng, rank chứa expert đó chậm → gate cả bước của cả 8 rank. Với workload AgentX (agent/tool-use) phân bố expert theo chủ đề rất dễ lệch. Trong stack hiện tại không có metric per-expert nào được register - chỉ có sglang:eplb_balancedness (metrics_collector.py:923, Summary per forward_mode) nhưng chỉ register khi bật EPLB và ở moe_ep_rank=0. **Số đo GPU §9 (23/10): deep_ep chiếm 92,7% kernel-time — vấn đề lớn nhất theo thời gian GPU nhưng out of scope mentor.** 


## 6. Nguồn dữ liệu & quy ước cite (compact)

| Nguồn | Nội dung | Dùng ở |
|---|---|---|
| `analysis/sglang_rank_ts.csv` + `analysis/dynamo_ts.csv` | trích từ `server_metrics_export.jsonl` (8081 `sglang:*` per dp_rank; 8888 `dynamo_*`), benchmark window | §1–§2 per-rank mean/spread, KV, cache hit, queue |
| `engine_*.log` / `frontend_*.log` (symlink trong run dir) | batch scheduler per-minute, pending-token, boot lines, WARN retract; dòng `Selected worker router_mode="kv"` (logit, host_pinned_blocks, dp_rank) + `Expiring stale request`; join router↔engine theo timestamp ±2 s; join `schedule_eval.jsonl` (`input_len`, `output_len`, `queue_wait_s`, `device_matched_prefix_tokens_first_admission`, `host_hit_prefix_tokens_first_admission`) theo arrival ±2 s | §1 pending/running, §2 queue-wait/TTFT per rank, corr(logit,·), §4 phân tích pinned/host-hit |
| `e2_metrics.json` | throughput/TTFT/E2E/TPOT/fairness client | §1 |
| `gpu_samples.csv` | nvidia-smi 1 s × 8 GPU | §1 GPU util |
| `profiling/pyspy_*.svg` | flame 25 Hz × 120 s × 8 scheduler (4/5 run; run 2 probes bị skip) | §3 cơ chế |
| `profile_export.jsonl` | `benchmark_phase` + `request_start_ns` → window; duration từ `e2_metrics.json` (`run.duration_s`) | mọi bảng |

Benchmark window mỗi run = record đầu phase `profiling` → +1800 s. **Lưu ý (20/10):**
`schedule_eval.jsonl` là file chung bị **append qua các run** trong cùng đợt — mọi phép
join/phân tích phải lọc record theo khoảng thời gian log của từng run, không thì gom
nhầm toàn bộ vào run đầu. Chi tiết cách tính từng số + cite file/dòng: focus report từng run - run 1
`docs/focus3320_routing_scheduling.md`, run 2–5
`docs/focus_c64_<stamp>_routing_scheduling.md`. Reproduce: `extract_metrics_timeseries.py <run_dir>`
(srun trên node) → `gen_focus_report.py <run_dir>` → `compare_focus_runs.py
<run_dir...>`; chuỗi kiểm chứng lại §4 (20/10, đã gom thành 1 script, tự lọc window
từng run): `focus2_retrace.py [--part formula|kvblind|pinned|drain|perreq|channels|hhsplit|verify|all] <run_dir>...`

## 7. Xác nhận 10 run + NSys 5-run (22/10)

Thêm 5 cell NSys (run 6–10: `c64_20261021_{141334,152339,171014,182013,193141}`,
env-gate `NSYS_PROFILE=1` commit `cec7d6ace1`, collection window +120 s) — so sánh
đầy đủ trong `docs/focus_consistency_10runs.md` (10 run) và
`docs/focus_consistency_5runs.md` (riêng 5 cell NSys). Kết luận:

- **10/10 run CONSISTENT cả 2 focus + health** (bảng §1 của doc 10-run): throughput
  520–546 tok/s, running-req 2.30–4.68/rank, backlog đỉnh 73.559–132.265 token,
  KV spread p50 0.50–0.57, corr(logit,TTFT) +0.26–+0.52, cache reuse 87.0–88.0%.
- **Chuỗi §4 lặp lại 5/5 cell NSys** (retrace từng cell, log
  `/shared/loipp/logs/retrace_<stamp>.txt`): pinned 89–91%; host-hit 14.1–17.8% req
  gom 65.7–70.9% tổng qw-giây; qw host-hit 21.2–25.7 s vs prefix-on-device 1.8–2.4 s;
  corr(logit, state+16·unc/64) +0.877–+0.897 vs state-only +0.519–+0.611 — binary
  1.4.2 có term uncached ở cả 5 cell mới.
- **NSys wrap không perturb hành vi**: nhóm NSys throughput 530–546 vs nhóm cũ
  520–532 (chênh +2.6% nhóm, nằm trong dải run-to-run của từng nhóm); engine code
  dfs-weight không đổi giữa 2 nhóm — 2 commit `python/sglang` trong khoảng 15–21/10
  (`b21cde8cce`, `0229372590`) đều chỉ đụng nhánh boost (`_sort_by_boost`,
  `AdaptiveBoostGamma`), path dfs-weight chỉ đổi value radix tree simulated
  (bool→int64, không đọc bởi dfs-weight).
- **Bằng chứng GPU window (+120 s, pha warmup — host tier 0.00M tại window, chuỗi
  load-back chưa diễn ra)**: 5/5 cell `deep_ep::legacy::intranode`
  dispatch+cached_notify_combine+notify_dispatch = **94.3–94.6% kernel-busy**
  (n≈53–59k/window, avg 18.2–19.1 ms, max 60–171 ms — spin chờ rank khác; dữ liệu
  thật `combine` chỉ ~0.7 ms) — chữ ký GPU của batch mỏng MoE EP=8 (focus 1);
  memcpy DtoD 18.1–20.4 GB/window, HtoD/DtoH ≈ 0 GB. Truy vấn:
  `scripts/nsys_window_summary.py` (sqlite từng cell, `profiling/nsys/*.sqlite`).

## 8. Xác nhận trực tiếp cơ chế prefill-delayer từ metric outcomes (23/10)

Trích `sglang:prefill_delayer_outcomes` (counter) + 2 histogram
`wait_forward_passes`/`wait_seconds` **kèm đủ label** từ
`server_metrics_export.jsonl` của 5 cell NSys (delta trong benchmark window từng
cell; trước đây `extract_metrics_timeseries.py` bỏ label nên metric này chưa
phân tích được). Script `scripts/delayer_outcomes_scan.py`; log
`/shared/loipp/logs/delayer_outcomes_retry.log`; chi tiết per-rank
`/shared/loipp/logs/delayer_outcomes_<stamp>.tsv`.

| Chỉ số (window, gom 8 rank) | 141334 | 152339 | 171014 | 182013 | 193141 |
|---|---|---|---|---|---|
| Pass bị hoãn — nhánh mixed (`delay`) | 118.704+1.328 | 120.067+1.269 | 119.712+1.338 | 116.709+1.377 | 124.319+1.209 |
| — trong đó chunk tiếp diễn vẫn chạy (exec=true) | 1.328 | 1.269 | 1.338 | 1.377 | 1.209 |
| Pass không rank nào có prefill (`none`) | 59.783 | 61.982 | 56.311 | 56.138 | 58.752 |
| Hết 30 pass buộc cho phép, KHÔNG sinh batch | 3.002 | 3.044 | 3.033 | 2.985 | 3.185 |
| Hết 30 pass buộc cho phép, CÓ sinh batch | 1.094 | 1.084 | 1.079 | 1.031 | 1.085 |
| Nhánh "all" (mọi rank cùng có prefill) | **0** | **0** | **0** | **0** | **0** |
| `wait_forward_passes` bucket ≤20 pass | 0 | 0 | 0 | 0 | 0 |
| `wait_seconds` mean mỗi lần chờ (s) | 2,38 | 2,15 | 2,32 | 2,35 | 2,06 |
| Tổng pass/rank (→ ms/pass) | 22.989 (78) | 23.431 (77) | 22.684 (79) | 22.280 (81) | 23.569 (76) |
| Tỷ lệ pass rơi vào trạng thái mixed | 65,3% | 64,7% | 66,7% | 66,3% | 66,6% |

Kết luận (5/5 cell nhất quán):

- **Nhánh "all" = 0** → slot-trigger trong nhánh "all" (`prefill_delayer.py`
  270-277) chưa bao giờ chạy — khớp suy luận code+config (256 slot ≫ running
  2,3-4,5/rank; queue-trigger tắt `queue_trigger_enabled=False`; valve watermark
  tắt `token_usage_low_watermark=None`, engine log dòng 673-694). **100% lần hoãn
  đến từ nhánh mixed** ("một số rank có prefill, số kia không" —
  `prefill_delayer.py:319-346`).
- **100% lần chờ kết thúc ở bucket 21-29 pass** (bucket ≤20 = 0) → mỗi lần chờ
  đều chạm mũ `max_delay_passes=30`, không bao giờ dừng sớm nhờ alignment (vì
  "all" không đạt được bao giờ).
- **wait_seconds mean 2,06-2,38 s ≈ queue-wait của request trúng device
  (1,80-2,42 s, §7)** → phần delayer góp vào hàng chờ = đúng 1 chu kỳ 30 pass
  (~76-81 ms/pass), đúng như thiết kế bounded.
- **~65-67% tổng số pass scheduler bị giữ prefill** vì mất đồng bộ rank — nguyên
  nhân gốc là router ghim ~90% request về rank giữ tiền tố (focus 2): bottleneck
  (1) được nuôi bằng bottleneck (2), khớp chiều đã chốt ở §3-§4.
- Hết 30 pass mà vẫn **không sinh batch: 2,7-2,9×** số lần sinh batch → phần lớn
  lần thử vào batch thất bại ngay cả khi delayer đã buông — nhất quán với request
  trúng host phải chờ thu hồi+nạp lại qua nhiều chu kỳ (qw trúng host 21,2-25,7 s
  ≈ ~9-11 chu kỳ × ~2,3 s cộng thời gian nạp; metric không có label host/device
  nên chưa tách trực tiếp từng lần).
- Kiểm tra chéo: tổng pass có prefill chạy (timeout-exec + chunk-exec) =
  2.294-2.422 ≈ số dòng `Prefill batch` trong window đếm từ engine log (cell
  141334: 2.422 vs 2.424 — chênh 2 ở biên scrape).

## 9. NSys readout đầy đủ theo quy trình mentor (23/10)

Script `scripts/nsys_readout.py` (bước 1 `nsys stats` 4 report chuẩn; bước 2
python đọc sqlite `?mode=ro`, không ghi file gốc). Output:
`results/glm53/agentx_dynamo/profiling/dfs-weight/nsys_readout/` (20 PNG + CSV
từng cell + `summary_cross.csv` + `index.html` kèm chú giải). 5/5 cell nhất
quán, số ví dụ cell 141334:

- **Kernel theo tên**: bộ ba deep_ep (`dispatch` 253,8 s +
  `cached_notify_combine` 252,8 s + `notify_dispatch` 62,3 s) = **92,7%
  kernel-time** (khớp §7 94,3-94,6%, chênh nhỏ do ranh giới window/tổng); 3
  kernel đếm đúng 1:1:1 (13.384) = mỗi pass một vòng all-to-all; `combine`
  (dữ liệu thật) 0,73 ms/lần vs `dispatch` 18,96 ms — phần còn lại là spin chờ
  rank khác. Compute thật (GEMM/attention/elementwise/NCCL) gộm ~5%.
- **CUDA API**: `cudaStreamSynchronize` 413,4 s ≈ 97% thời gian API, avg
  **30,7 ms/lần**, số lần (13.475) ≈ số pass → mỗi pass kết thúc bằng một lần
  sync chờ; launch rẻ (344.404 lần = 2,83 s) → không launch-bound.
- **Timeline 8 GPU**: busy **55,0-63,9%**/GPU, khá đều giữa 8 GPU → idle 36-45%
  nhưng **phân mảnh**: 2.211-2.353 gap ≥ 1 ms/GPU, gap max 0,82-0,97 s, avg
  ~21 ms — khe nhỏ tại ranh giới forward pass. Nhịp pass ~76 ms (1.673
  pass/GPU/window) khớp §8.
- **Memcpy**: DtoD 18,1-20,4 GB/window (~147 MB/s — không memory-bound);
  HtoD/DtoH ≈ 0 — window warmup, host tier 0,00M, **không suy ra được
  load-back steady-state** (bằng chứng load-back vẫn là metrics §4/§7/§8).
- **Chẩn đoán**: comm/sync-bound (EP=8) xác nhận mạnh; không
  compute/launch/memory-bound tại window này. Idle phân mảnh + spin trong
  kernel = bóng GPU của focus 1 (batch mỏng) — batch càng mỏng, chi phí
  all-to-all trên mỗi token càng lớn. EP-MoE là vấn đề lớn nhất theo thời gian
  GPU nhưng out of scope (§5). NVTX ~20-23k events/cell chưa khai thác.
