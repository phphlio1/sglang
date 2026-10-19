# Danh sách sửa trước khi re-run E2 — SCRATCH, không commit

Ngày 2026-10-08. Hai quyết định mới từ đợt review raw-data:

1. **Nghiêm trọng**: benchmark hiện là closed-loop, phải là open-loop →
   check hết, sửa hết, rồi chạy lại toàn bộ.
2. **Đã chốt**: report `fcfs-match` làm đại diện fcfs (bỏ plain fcfs khỏi
   bảng).

File này là todo working note; mỗi mục khi xong sẽ chuyển thành thay đổi
thật trong code/docs và mục được đánh dấu.

## A. Open-loop (sửa trước khi chạy lại)

### A1. Chẩn đoán

`run_e2.sh:243` chạy client với `--concurrency N` → closed loop: client
phát request mới ngay khi request cũ xong → **arrival phụ thuộc
completion** → policy đang được đánh giá làm nhiễm chính quá trình arrival
(feedback loop). Bằng chứng trong dữ liệu closed-loop hiện có: hai arm
ordering identical (`fcfs` vs `fcfs-match`, cùng seed 42) mà queue p99 c4
chênh 0.003 s vs ~2.9–3.2 s; số vụ head-of-line blocking khác nhau (2 vs
4 vụ) — không thể xảy ra nếu arrival schedule cố định.

### A2. Cần check trong AIPerf (đường phát request — tìm chỗ closed-loop
coupling vào)

- `workers/session_manager.py` — session multi-turn: turn kế có phải chờ
  turn trước **hoàn thành** không (replay có input pre-canned từ trace,
  không cần output — nhưng code có thể vẫn serialize)
- `workers/worker_manager.py` / `workers/worker.py` — semaphore
  concurrency: khi quá ceiling thì request bị hold → lệch schedule
- `orchestrator/strategies.py` + `config/phases.py` — phase type:
  `concurrency` vs `fixed_schedule`; scenario `inferencex-agentx-mvp` là
  `TimingMode.AGENTIC_REPLAY` + `forbid_ignore_trace_delays=True`
- `dataset/loader/semianalysis_cc_traces_weka.py` +
  `dataset/loader/_delay_cap.py` — trace chỉ có inter-turn delay (relatif)
  hay timestamp arrival tuyệt đối; `system_idle_gap_cap_seconds=10` đã
  nén gap lớn

### A3. AIPerf đã có sẵn open-loop (không cần viết client mới)

- `phases: type: fixed_schedule` — phát request đúng timestamp của trace
  (xem template `config/templates/trace_replay.yaml`: `autoOffset`,
  `gracePeriod` drain, `speedupRatio`, `maxIsl/maxOsl`). Trường
  `concurrency` ở mode này chỉ là **ceiling an toàn**, không phải driver.
- `user_centric_rate` (`timing/strategies/user_centric_rate.py`) — **chế
  độ giữ được knob "N users" mà vẫn open-loop**: num_users N, mỗi user
  phát turn theo lịch giờ (`next_send_time = max(now, scheduled +
  turn_gap)`, `turn_gap = N / request_rate`), user mới spawn trải đều —
  arrival theo lịch, chỉ trượt cục bộ khi turn trước chạy quá lâu, không
  có global completion-feedback.
- `rate_series` — QPS piecewise-linear.
- `adaptive_scale` — tìm boundary của control variable
  (concurrency/request_rate/users) theo SLA.
- format `mooncake_trace`: 1 dòng/request
  `{timestamp_ms, input_length, output_length, hash_ids}`.

### A3b. Giải đáp "open-loop thì còn concurrency N không?"

Trong open-loop thuần, concurrency là **đầu ra** (đo được), không phải
đầu vào. Nhưng yêu cầu "thử nghiệm các mức concurrency khác nhau" vẫn
giữ được, theo hai cách:

1. **Band theo num_users** (khuyến nghị): `user_centric_rate` với
   num_users = 4/8/16 — đúng tinh thần band cũ (N agent session), mỗi
   user có tối đa 1 turn in-flight nên concurrency thực tế ≈ N khi hệ
   kịp phục vụ, và tụt xuống khi bão hòa (chính là tín hiệu cần đo).
   Chọn request_rate để đưa tải vào band mong muốn.
2. Band theo arrival rate: fixed_schedule/rate_series, rồi đo
   concurrency thực tế từng band (gauge `num_queue_reqs` + running_reqs
   + Little's law đã có sẵn trong analysis).

Cả hai cách ghi rõ rate/tải vào run_meta; bảng report ghi concurrency
thực đo cạnh concurrency mục tiêu.

### A4. Hệ quả phải suy nghĩ lại

- "Band" c4/c8/c16 **mất nghĩa** (concurrency là tham số closed-loop) →
  định nghĩa lại theo arrival rate / input-token rate. Gộp với
  pressure-band recalibration đã lên kế hoạch cho H200 (mục D3).
- Mọi kết luận closed-loop cũ (bimodal c4/c8, saturation c16, queue-depth
  ~2.1, Little's law) chỉ còn giá trị làm phụ lục "closed-loop pilot";
  bảng chính của report phải từ re-run open-loop.
- Drain: dùng `gracePeriod` để tránh vụ in-flight-at-close (2 giant của
  lpm/c16 wait 306.3/152.6 s, prefill chưa xong khi đóng cửa sổ).
- Warmup cut + `offset_lines` bookkeeping: cơ chế giữ nguyên, kiểm lại
  với `autoOffset` của fixed_schedule.

### A5. Câu hỏi mở cần chốt TRƯỚC khi sửa

1. Trace có timestamp arrival tuyệt đối không, hay chỉ inter-turn delay?
   (fixed_schedule cần timestamp; nếu chỉ có delay thì tổng hợp từ delay
   — quyết định trong loader.)
2. Giữ cấu trúc multi-turn session (mỗi session serialize các turn của
   nó) hay flatten tất cả request về một schedule? Multi-turn + open-loop
   thuần có mâu thuẫn: turn kế của một session có nên phát đúng giờ trace
   kể cả khi turn trước chưa xong? (Replay pre-canned cho phép, nhưng có
   còn là "agentic" không — cần chốt semantics.)
3. Áp suất: replay đúng tốc độ trace (speedupRatio 1.0) hay scale để ép
   vào band bão hòa tương đương c16 cũ?

## B. fcfs-match (đã chốt, áp dụng cho bảng hiện tại + re-run)

- B1. Report `fcfs-match` cho cả 2 block (c4/c8 2026-10-07 và c16
  2026-10-08); bỏ dòng plain `fcfs` khỏi bảng in. Lý do chọn: valid ở cả
  hai block (plain fcfs hỏng cột cache hit ở block pre-fix), nhất quán
  một tên, và trùng `POLICIES=(fcfs-match lpm hrrn dfs-weight boost)` —
  default hiện tại của `run_e2.sh:100`.
- B2. Sửa code: `print_results_table.py` + `print_metrics_v2_table.py`
  bỏ dòng plain fcfs (chú thích rõ trong commit: this lifts the
  metrics-v2-time constraint of not touching print_results_table.py
  default mode — quyết định mới).
- B3. Sửa docs: `full_run.md`, `full_c16.md`, `metrics_v2.md` — bảng nhúng
  bỏ dòng fcfs, thêm footnote: pre-fix defect
  (`num_matched_prefix_tokens=0` với policy cache-agnostic → cache hit
  0.00% trên 12,908,229 token của fcfs/c4; ordering vẫn đúng) → fix
  `bdc67a5720` (`schedule_batch.py:1718-1727`, read-only re-match
  bookkeeping) → fcfs-match là arm đại diện; plain fcfs c16 (78.49%)
  giữ trong json làm cross-check với fcfs-match (80.12%).
- B4. Re-run open-loop: chạy 5 arm của POLICIES hiện tại, không chạy lại
  plain fcfs.

## C. Sửa nhỏ từ đợt audit raw (làm cùng re-run)

- C1. In-flight-at-close: drain bằng `gracePeriod`; counter
  `n_dropped_no_ttft` của stretch đã có — giữ.
- C2. TPOT p95 c16 boost-wall (168.1 ms, decode tail): khi re-run thêm
  arm static-wall tại gamma-hat offline ≈ 0.09 1/s.
- C3. (optional, phía server sglang) histogram `sglang:queue_time_seconds`
  bucket mịn hơn — không bắt buộc vì bảng v2 đã dùng eval log per-request;
  chỉ làm nếu muốn histogram tự đứng được.

## D. H200 8-GPU (parked; A làm thay đổi vài mục)

- D1. TP>1 cần rank-gate schedule-eval logger (hiện mỗi rank ghi → 8×
  duplicate records).
- D2. Results root per-hardware (`results/4x-h200/` stub); SBATCH
  `--gres=gpu:1` → theo layout khi tới.
- D3. Pressure-band recalibration **gộp với A4**: band giờ định nghĩa theo
  arrival rate open-loop, không theo concurrency.
- D4. Hai câu hỏi mở cũ: GPU layout (TP8 một queue / DP replicas /
  TP4×2) và SM90 config basis (cookbook + campaign knobs / carry RTX
  flags).
- D5. **J của ada-boost không mang nguyên giá trị RTX (J=50):** J là
  max displacement theo giây (req mới chỉ được vượt req đến trong J s
  gần nhất) — floor tự bù theo rate qua bisection (γ_floor giải
  b_γ(k/rate) = J), nhưng J tối ưu phụ thuộc timescale chờ của
  workload (RTX c16: queue p99 ~120-150 s, giant prefill ~30-60 s →
  J=50 ≈ 1 giant prefill). Ở H200 rate TP8 cao hơn nhiều lần + pool KV
  lớn → timescale khác. Cách đặt: **derive từ đo đạc trên rig** (J ≈
  p95 thời lượng prefill uncached của giant đo ở phút đầu, hoặc phân
  số SLO TTFT nếu prod có SLO), rồi mini-sweep 2-3 arm {nửa, một, đôi} ×
  đại lượng đó — không lặp sweep 5 điểm kiểu RTX. Giảm rủi ro: J chỉ là
  floor, chỉ bind khi adaptive muốn tụt γ (thin tail); nếu γ̂ đứng cao
  hơn floor thì J hầu như không ảnh hưởng (RTX J=60: floor không bao
  giờ bind). Mọi kết luận "J nào tốt hơn" cần repeat ≥2 run/cell.

## E. Đóng góp upstream sglang về fcfs (câu hỏi "RFC")

- Thuật ngữ đúng cho sglang: **GitHub issue + PR** (sglang không có quy
  trình RFC chính thức như PEP/Rust RFC; thứ gần nhất là issue discussion
  cho design lớn — cái này không cần).
- Đáng đóng góp: bug `num_matched_prefix_tokens` = 0 với policy
  cache-agnostic (fcfs/lof/random/routing-key). Symptom upstream-visible:
  `SchedulerLoadInquirer.get_num_waiting_uncached_tokens`
  (load_snapshot) thấy 0 reuse → ước lượng pending prefill tokens sai.
  Fix: 7 dòng trong `init_next_round_input` (tách từ `bdc67a5720`,
  `schedule_batch.py:1718-1727`).
- Không đóng góp: `ScheduleEvalLogger` (fork-specific), boost wall-clock
  (research). Tách riêng, đừng gửi nguyên commit trộn.
- Trước khi gửi: đã verify `origin/main` (b1bbd74f287) còn đúng bug —
  `schedule_batch.py` init `num_matched_prefix_tokens = 0` (dòng 1198),
  reset 0 khi retraction (dòng 1986), consumer `load_inquirer.py:87`
  giống hệt fork. Remote của repo này chính là upstream
  (`sgl-project/sglang`) → PR thẳng từ branch sạch. Viết test nhỏ theo
  chuẩn upstream, issue nêu evidence (0 matched trên 12.9M token).
  Lưu ý thêm: retraction reset field về 0 — fix phải ghi lại ở admission
  re-match như bdc67a5720 thì mới đúng qua chu kì retract.
