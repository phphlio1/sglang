# Bottleneck Profile — GLM-5.3 Prod Config (13-10-2026)

Trạng thái: **spec + kế hoạch cho bản đầy đủ (frontend dynamo + engine)**.
Cell engine-only vừa rồi (3065) **bỏ không tính** — thiếu frontend dynamo;
sẽ chạy lại cell mới trên config đầy đủ (§5). Artifacts của cell cũ giữ lại
chỉ để audit (§6).

## 1. Workload: AgentX

| Hạng mục | Cấu hình |
|---|---|
| Benchmark | AIPerf `aiperf profile`, scenario `inferencex-agentx-mvp`, streaming + per-chunk usage |
| Dataset | `semianalysis_cc_traces_weka_062126_256k` (trace agent, context dài, 256K-capped) |
| Model | `zai-org/GLM-5.3` (checkpoint GLM-5.3-W4AFP8, quant w4afp8) |
| Hardware | 8×H200 (node hgx-h200-01, 2×NUMA 96c, ~2TB RAM) |
| Scheduling Policy | `dfs-weight` (policy của prod) |
| Concurrency | 64 (mức cao nhất của sweep 16/32/64) |
| Duration | 30 phút (1800 s) + grace 360 s |
| Seed | 42 |
| Boot | fresh boot mỗi cell (FORCE_RESTART=1) — radix cache lạnh mỗi run |

## 2. Serving Configuration (prod verbatim)

**Frontend / Router — glm53-w4afp8-dynamo-frontend**

Image: `vcr.vngcloud.vn/60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1`

```
python3 -m dynamo.frontend \
  --http-host=0.0.0.0 \
  --http-port=8888 \
  --router-mode=kv \
  --router-prefill-load-scale=16 \
  --router-min-initial-workers=1 \
  --dyn-chat-processor=sglang \
  --dyn-preprocess-workers=8 \
  --tool-call-parser=glm47 \
  --reasoning-parser=glm45
```

**Engine — glm53-w4afp8-dynamo-engine**

Image: `vcr.vngcloud.vn/108942-public69/dynamo-stack/sglang-runtime:1.5.0`

```
python3 -m dynamo.sglang \
  --model-path=/models/PhalaCloud/GLM-5.3-W4AFP8 \
  --served-model-name=zai-org/GLM-5.3 \
  --quantization=w4afp8 \
  --tp=8 --dp=8 --ep=8 \
  --enable-dp-attention \
  --enable-dp-attention-local-control-broadcast \
  --enable-dp-lm-head \
  --dist-init-addr=127.0.0.1:10888 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --moe-a2a-backend=deepep \
  --chunked-prefill-size=32768 \
  --tool-call-parser=glm47 \
  --reasoning-parser=glm45 \
  --mem-fraction-static=0.75 \
  --max-running-requests=256 \
  --context-length=300000 \
  --kv-cache-dtype=fp8_e4m3 \
  --dsa-prefill-backend=flashmla_sparse_q8 \
  --allow-auto-truncate \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --enable-cache-report \
  --enable-hierarchical-cache \
  --hicache-size=128 \
  --hicache-io-backend=direct \
  --hicache-write-policy=write_back \
  --speculative-algorithm=EAGLE \
  --speculative-num-steps=3 \
  --speculative-eagle-topk=1 \
  --speculative-num-draft-tokens=4 \
  --schedule-policy=dfs-weight \
  --enable-prefill-delayer \
  --kv-events-config={"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:8900"}
```

**ENV (prod k8s, mang nguyên văn):** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`SGLANG_DP_USE_GATHERV=1`, `SGLANG_ENABLE_METRICS_DP_ATTENTION=1`,
`SGLANG_ENABLE_METRICS_DEVICE_TIMER=1`, `NCCL_P2P_LEVEL=NVL`, `HF_HUB_OFFLINE=1`.

**Map sang branch này** (khác chính tả, cùng config sau khi resolve — server_args
dump trong boot log xác nhận): `--tp=8 --dp=8 --enable-dp-attention --ep=8` →
`--tp-size 8 --attn-dp-size 8 --ep-size 8` (dp_size là deprecated alias;
attn-dp-size>1 tự bật dp-attention); `python3 -m dynamo.sglang` →
`python3 -m sglang.launch_server` (cho tới khi gắn dynamo, §5); port 8888/8889 →
30000 (harness); `--dsa-decode-backend flashmla_kv` viết rõ (prod để auto, auto
chọn đúng flashmla_kv trên Hopper+fp8_e4m3).

**Divergence còn lại (3):**
1. `--model-path` = bản /mnt của cùng weights (tài khoản không được download
   model; HF_HUB_OFFLINE=1 ép local-only).
2. ENV deployment bắt buộc mà prod không liệt kê: CUDA_HOME/nvcc 13.4,
   NVSHMEM_HCA_LIST, `SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=600`.
3. `--page-size=64` tường minh (prod để auto): cùng giá trị resolve —
   FlashMLA ép đúng 64 — nhưng dynamo publisher đọc `server_args.page_size`
   ngay trên record input chưa resolve của branch (resolution nằm trong
   overlay, field input giữ None) → auto làm `KvEventPublisher` crash
   (smoke 3190). Set tường minh để dynamo đọc được 64.

## 3. Thông số profiling bottleneck — toàn stack (frontend global → engine local)

### 3.1 Frontend / Router (toàn cục — quyết định request đi rank nào)

| Thông số | Giá trị | Vai trò trong bottleneck profile |
|---|---|---|
| HTTP ingress | `0.0.0.0:8888` | điểm vào client (aiperf); port harness 30000 |
| `--router-mode=kv` | KV-aware routing | route theo radix-tree KV mỗi worker thay vì round-robin — biến chính triệt tiêu bottleneck rank-imbalance |
| `--router-prefill-load-scale=16` | 16 | trọng số cache-affinity trong cost function router: `cost = 16×prefill_blocks + potential_decode + decode_active×active_reqs`, chọn argmin |
| `--router-min-initial-workers=1` | 1 | số worker tối thiểu trước khi router bắt đầu quyết định |
| `--dyn-chat-processor=sglang` | sglang | processor chat phía dynamo |
| `--dyn-preprocess-workers=8` | 8 | 8 preprocess worker (tokenize/parse song song trước dispatch) |
| `--tool-call-parser / --reasoning-parser` | glm47 / glm45 | parse tool-call/reasoning stream |
| Nguồn dữ liệu router | kv-events ZMQ `tcp://*:8900` | engine publish KV events mỗi attn-tp rank 0 (8 endpoint 8900+dp_rank) — đã verify chạy đủ 8 rank ở boot |
| Service discovery | dynamo component graph (NATS) | frontend tìm worker `dynamo.sglang`; lý do engine phải chạy dưới wrapper dynamo, không phải `sglang.launch_server` trần |

### 3.2 Engine — topology / song song

| Thông số | Giá trị | Ghi chú profiling |
|---|---|---|
| `--tp / --dp(attn) / --ep` | 8 / 8 / 8 | 8 attention-DP rank, mỗi rank 1 GPU; MoE expert song song 8 |
| `--enable-dp-attention` (+`local-control-broadcast`, `dp-lm-head`) | on | DP-attention: prefill/decode batch độc lập mỗi rank — nơi đo imbalance per-rank |
| `--dist-init-addr` | 127.0.0.1:10888 | khởi tạo process group nội node |
| `--numa-node` | 0 0 0 0 1 1 1 1 | 4 rank đầu NUMA 0, 4 rank sau NUMA 1 (mỗi NUMA 96c) |
| `--moe-a2a-backend` | deepep | all-to-all MoE qua DeepEP (NVSHMEM, HCA rocep41s0f0) |

### 3.3 Engine — bộ nhớ / KV cache

| Thông số | Giá trị | Ghi chú profiling |
|---|---|---|
| `--mem-fraction-static` | 0.75 | ~121 GiB/GPU weights+KV pools (đo được 121.3/124 GiB) |
| `--kv-cache-dtype` | fp8_e4m3 | KV nửa byte — ảnh hưởng lựa chọn DSA backend |
| `--enable-hierarchical-cache` + `--hicache-size 128` + `direct` + `write_back` | 128 GB host/rank | đo cache hit: device/host/storage hit rate |
| `--max-running-requests` | 256 | trần admission mỗi rank |
| `--context-length` | 300000 | trần context (dataset cap 256K) |
| `--allow-auto-truncate` | on | tự cắt input vượt context |

### 3.4 Engine — compute / kernel

| Thông số | Giá trị | Ghi chú profiling |
|---|---|---|
| `--quantization` | w4afp8 | weights 65.3 GB/rank (đo từ boot log) |
| `--dsa-prefill-backend` | flashmla_sparse_q8 | DSA prefill sparse FP8 |
| `--dsa-decode-backend` | flashmla_kv (auto→) | cặp decode bắt buộc trên Hopper+fp8 |
| `--chunked-prefill-size` | 32768 | = 4096 token/rank sau chia attn-dp |
| `--speculative-algorithm` (EAGLE) + steps/topk/draft | 3 / 1 / 4 | spec decode; đo accept len (ngưỡng khỏe ≥2.0) |
| `--enable-prefill-delayer` | on | cân bằng prefill/decode tại admission |

### 3.5 Engine — scheduler local (biến thí nghiệm)

| Thông số | Giá trị | Ghi chú profiling |
|---|---|---|
| `--schedule-policy` | dfs-weight (cell profile) | 5 policy sweep: fcfs-match lpm hrrn dfs-weight boost |
| `--in-batch-prefix-caching-max-pin-s` | 60 | trần follower-pin in-batch dedup |
| Boost (chỉ policy `boost`) | adaptive gamma, bin 256, clock wall | chờ code Adaptive Gamma Boost của teammate |

### 3.6 ENV runtime (ảnh hưởng hành vi đo)

| ENV | Giá trị | Tác động |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:True | phân bổ GPU memory — ảnh hưởng hicache pool |
| `SGLANG_DP_USE_GATHERV` | 1 | gatherv cho DP-attention sync |
| `NCCL_P2P_LEVEL` | NVL | P2P qua NVLink đúng fabric đơn node |
| `SGLANG_ENABLE_METRICS_DP_ATTENTION` / `..._DEVICE_TIMER` | 1 / 1 | metrics per-DP-rank + device timer (mối quan hệ stage seconds) |
| `HF_HUB_OFFLINE` | 1 | ép local weights (không chạm mạng) |

### 3.7 Khung đo (instrumentation phía thí nghiệm)

| Công cụ | Thông số | Đo gì |
|---|---|---|
| aiperf client | streaming, per-chunk usage, `--use-server-token-count` | TTFT/TPOT/ITL per chunk, cold/warm class |
| `/metrics` scrape | 3 Hz, theo `benchmark_phase` (warmup/profiling) | scheduler stage seconds, queue/running per dp_rank, cache hit, retraction |
| GPU sampler | nvidia-smi 1 s | util/memory/power mỗi GPU |
| schedule-eval log | per-request | queue_wait, matched_prefix_tokens, host_hit, in_batch_dedup |
| torch profiler | — | không khả dụng trên stack dynamo (frontend không proxy /start_profile); thay bằng py-spy + gpu_samples |
| py-spy | 25 Hz × 120 s mỗi scheduler (8/8) | flame Python scheduler — attach được nhờ `SGLANG_ALLOW_PTRACE=1` → scheduler gọi prctl(PR_SET_PTRACER_ANY) trên node ptrace_scope=1 |
| bottleneck_report.py | Q1–Q7 decision tree | verdict tổng hợp: GPU busy, client/server gap, TTFT front path, rank balance, KV, spec, hicache |

## 4. Cell full-stack (3320) — CONFIG ĐẦY ĐỦ (frontend + engine)

Cell `agentx_dynamo/profiling/dfs-weight/c64_20261014_112105`, job 3320
(2026-10-14, ~68 phút). **Cấu hình đúng deliverable**: dynamo frontend
`--router-mode=kv` (8888) + engine worker `python -m dynamo.sglang` với toàn
bộ engine config prod (§2) — chạy qua `scripts/profile_cell.sh dynamo 64
dfs-weight` (seed 42, 30 min, c64, drain 360 s, fresh boot).

### 4.1 Kết quả tổng quan (client-side, aiperf)

| Chỉ số | Giá trị |
|---|---|
| Complete records | 1169 (warmup 65 loại; 19 error) |
| Request throughput | 32.3 req/min (0.54 rps) |
| Output throughput | 522 tok/s (31.3k tok/min); total 51.2k tok/s |
| TTFT mean / p95 / p99 | 7.12 s / 20.5 s / 75.4 s |
| Queue-wait (server, per-request) | mean ~4.8 s warm-class / 3.9 s cold-class; p99 ~247 s (queue metric) |
| Starvation rate | ~10% (threshold 10 s) |

Tham chiếu audit: cell engine-only 3065 (cùng điểm dfs-weight × c64) chỉ thu
277 records, TTFT mean 108 s, 8 req/min — cell đó bị lỗi đo (boot lạnh NFS +
mất in-flight lúc close), không dùng so sánh; 3320 là số chuẩn.

### 4.2 Flame py-spy 8/8 scheduler (25 Hz × 120 s, steady-state)

Lần đầu cell nào có flame đầy đủ (3065 mất sạch vì ptrace). Top frames
(inclusive, trung bình trên 8 scheduler):

| % samples/scheduler | Frame |
|---|---|
| ~52% | `forward_impl` MoE (`moe/ep_moe/layer.py:267`) |
| ~51% | `run_moe_core` → `forward_cutlass_w4afp8` → `apply_deepep_normal` |
| ~25% | DeepEP dispatch: `dispatch_b` (`token_dispatcher/deepep.py`) qua TBO (`two_batch_overlap.py`) |
| ~11.5% | model forward tổng (`deepseek_v2.py`) |

Đọc: hơn nửa thời gian CPU scheduler nằm trong path MoE, trong đó a2a
DeepEP (dispatch_b normal mode) chiếm ~1/4 — phần còn lại là grouped GEMM
w4afp8 (`cutlass_w4a8_moe_deepep_normal`). Tương thích observation từ
engine log: fwd occupancy prefill chỉ 13-15% (idle chờ a2a/gather), tức MoE
a2a + weight-loading là ứng viên bottleneck chính.

### 4.3 Ghi chú / giới hạn cell full-stack

- **torch-profiler không có**: frontend dynamo không proxy `/start_profile`,
  engine dưới dynamo không có mặt HTTP riêng → không có trace kernel-level;
  flame py-spy + gpu_samples 1 s thay thế.
- **`prompt_tokens_details.cached_tokens` vắng trong usage** dù engine bật
  `--enable-cache-report` — frontend dynamo không thông qua field này →
  prompt-cache metrics phía client trống; dùng `matched_prefix_tokens`
  trong `schedule_eval.jsonl` (1237 records) thay thế.
- **Counter reset sglang:\***: aiperf detect counter reset lặp lại trong run
  (worker_id không đổi → không phải restart; khả năng dynamo publisher
  re-register counter) → thống kê counter server-side (decode/prefill
  counts) có thể undercount; không ảnh hưởng latency records phía client.
- GPU telemetry DCGM (9400/9401) không có trên node → aiperf drop; gpu_samples
  (nvidia-smi 1 s) vẫn đầy đủ.

### 4.4 File dữ liệu cell 3320

| File | Nội dung |
|---|---|
| `profile_export.jsonl` (926 KB) | per-request client records (1169) |
| `server_metrics_export.jsonl` (2.3 GB) | /metrics 3 Hz — chỉ phân tích qua slurm |
| `gpu_samples.csv` (1.1 MB) | util/memory/power 1 s mỗi GPU |
| `schedule_eval.jsonl` (1237) | queue_wait, matched_prefix_tokens, host_hit per request |
| `profiling/pyspy_*.svg` ×8 | flame 8 scheduler |
| `e2_metrics.json` / `run_meta.json` | tổng quan + config verbatim |
| `engine_*.log` / `frontend_*.log` (symlink) | boot + serving log |


## 5. Kế hoạch chạy lại trên config đầy đủ

1. **Lấy package dynamo**: ~~prod dùng image nội bộ `sglang-runtime:1.5.0-kvoverlap1`
   (chưa có trên node; cần credential registry VNG)~~ — **mentor chấp nhận
   dynamo 1.4.2** (quyết định 2026-10-14): không cần swap 1.5.0. Đường chạy
   hiện tại dùng dynamo **1.4.2** trích từ image
   `nvcr.io_nvidia_ai-dynamo_vllm-runtime_1.4.2.sqsh` team đặt sẵn trong
   /mnt/containers (đủ `dynamo.frontend` router-mode=kv + `dynamo.sglang` +
   nats-server tĩnh; mọi flag prod đều có) — đây là config chính thức của
   campaign.
2. **Ghép engine**: `dynamo.sglang` chạy engine in-process qua `sgl.Engine` →
   venv-dynamo (py3.12) mang cả dynamo 1.4.2 lẫn sglang branch mình (shim 1
   file cho `server_args_config_parser` — dynamo chỉ import, không gọi khi
   chạy thuần CLI). **Đã build xong** (BUILD VENV DYNAMO OK: imports dynamo
   frontend/sglang/runtime + torch 2.14.1+cu130 + shim). Engine worker expose
   metrics `sglang:*` multiprocess qua `DYN_SYSTEM_PORT=8081` (harness đổi URL
   scrape); frontend 8888 là mặt HTTP (aiperf bắn vào đây). NATS 4222 làm
   discovery/event plane. Stack start = `scripts/start_dynamo_stack.sh`
   (nats → frontend → engine, prod flags nguyên văn).
3. **Smoke full stack** (`scripts/smoke_dynamo_stack.sbatch`): **PASSED**
   (smoke 3307, 2026-10-14). Chain xanh end-to-end: hicache_hash_cpp
   pre-compile (openssl headers từ mirror) → frontend :8888 listen → model
   register sau 281 s → GET /v1/models → chat completion non-stream (64
   tokens, prompt 26) → `sglang:*` metrics đầy đủ 8 dp_rank trên
   `http://127.0.0.1:8081/metrics` (DYN_SYSTEM_PORT). Dòng "FAIL 5a" trong
   log 3307 là artifact `set -o pipefail` của chính smoke (`curl | grep -m`
   → SIGPIPE), không phải lỗi stack — đã sửa (curl ghi file rồi grep).
   9 vòng smoke 3167→3307, mỗi vòng đúng 1 lỗi: etcd discovery → blake3 →
   `get_model_config` (API upstream vs branch) → 2 bug scheduler branch
   (init sau early-return) → `page_size` (dynamo đọc field chưa resolve) →
   openssl JIT trên node. Vòng smoke kế tiếp verify thêm: chat **streaming
   SSE** (path aiperf `--streaming`, chưa từng test) + py-spy attach vào
   `sglang::scheduler` (xem mục 4).
4. Chạy cell profiling mới: dfs-weight × c64, seed 42, 30 min, full stack
   (frontend + engine) qua `scripts/profile_cell.sh dynamo 64 dfs-weight`
   (arm mới). Wiring harness đã xong trong `run_e2.sh`: `READY_ON_MODEL=1`
   (/health của dynamo trả 200 cả khi 0 worker — readiness thật = model trên
   /v1/models), `SERVER_PROC_PATTERN='dynamo\.sglang'` (fail-fast pgrep),
   `SERVER_METRICS_URLS=127.0.0.1:8081` (aiperf scrape engine registry;
   endpoint unreachable chỉ bị skip, không giết run), /server_info fallback
   (frontend dynamo không có endpoint này). **Flame py-spy**: node bật yama
   `ptrace_scope=1`, không sudo → attach sibling bị deny (mất toàn bộ flame
   cell 3065). Fix trên branch: `run_scheduler_process` gọi
   `prctl(PR_SET_PTRACER, ANY)` khi `SGLANG_ALLOW_PTRACE=1` (env-gated, đã
   validate trên node: py-spy dump sibling OK sau prctl; scheduler.py đã
   push lên mirror). torch-profiler qua HTTP không còn (frontend không proxy
   /start_profile; engine dưới dynamo không có mặt HTTP riêng) → flame là
   py-spy, ghi nhận là giới hạn của cell full-stack.

## 6. File dữ liệu (cell engine-only 3065 — chỉ để audit)

| File | Nội dung |
|---|---|
| `server_server_20261013_200032.log` (3.0 MB) | boot log: server_args dump, timeline boot, decode/prefill batch, accept len, retract |
| `bottleneck_report.json` | 7 verdict Q1–Q7 |
| `schedule_eval.jsonl` (345 records) | per-request: queue_wait, matched_prefix_tokens, host_hit |
| `server_metrics_export.jsonl` (2.35 GB) | /metrics 3 Hz theo benchmark_phase (chỉ phân tích qua slurm) |
| `gpu_samples.csv` (30,096 mẫu) | util/memory/power 1 s mỗi GPU |
| `e2_metrics.json` / `run_meta.json` | metrics tổng quan + config/seed/command verbatim |

Tất cả nằm trong
`results/8x-h200/glm53/agentx/profiling/dfs-weight/c64_20261013_200031/`.
