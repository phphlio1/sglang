# Bottleneck Profile — GLM-5.3 Prod Config

RunID: **3320** (14-10-2026) — full stack dynamo 1.4.2 (frontend router-mode=kv +
engine)

## 1. Workload: AgentX

| Hạng mục | Cấu hình |
|---|---|
| Benchmark | AIPerf `inferencex-agentx-mvp`, streaming + per-chunk usage, seed 42 |
| Dataset | `semianalysis_cc_traces_weka_062126_256k` (trace agent, 256K-capped) |
| Model / Hardware | `zai-org/GLM-5.3` (W4AFP8) / 8×H200 hgx-h200-01 (2×NUMA 96c) |
| Policy / Conc / Duration | `dfs-weight` (policy prod) / 64 / 30 min + drain 360 s |
| Engine | SGLang v0.5.20 |
| Boot | fresh mỗi cell (FORCE_RESTART=1, radix cache lạnh) |

## 2. Serving Configuration (prod verbatim)

**Frontend — glm53-w4afp8-dynamo-frontend**:

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

(Router kv chọn worker theo cost = `16×prefill_blocks + potential_decode +
decode_active×active_reqs`, argmin; nguồn cost = kv-events ZMQ 8900+dp_rank.)

**Engine — glm53-w4afp8-dynamo-engine**:

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

**ENV (prod k8s):** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`SGLANG_DP_USE_GATHERV=1`, `SGLANG_ENABLE_METRICS_DP_ATTENTION=1`,
`SGLANG_ENABLE_METRICS_DEVICE_TIMER=1`, `NCCL_P2P_LEVEL=NVL`, `HF_HUB_OFFLINE=1`.

**Divergence vs prod CLI:**
1. `HF_HUB_OFFLINE=1`
2. ENV deployment thêm: `CUDA_HOME/nvcc 13.4, NVSHMEM_HCA_LIST=rocep41s0f0`,
   memlock unlimited (deepep QP), openssl-dev headers từ mirror (JIT
   hicache_hash_cpp trên node).
3. `--page-size=64` (prod để auto): cùng giá trị resolve — FlashMLA
   ép 64.

## 3. Thông số profiling (instrumentation)

| Công cụ | Thông số đo | Bắt được gì → file |
|---|---|---|
| aiperf client | streaming, per-chunk usage, server token count, seed 42 | TTFT / ITL / TPOT per chunk, req + output throughput, starvation rate, cold/warm class → `profile_export.jsonl`, `e2_metrics.json` |
| Engine log (stdout scheduler, 1 dòng/batch) | trạng thái mỗi forward batch | Decode: `#running-req`, `#token`, **token usage**, `accept len/rate`, **gen throughput**, `fwd occupancy`; Prefill: `#new-seq`, `#new-token`, `#cached-token` (cache reuse), `#pending-token`, input throughput → `engine_*.log` |
| Frontend log (dynamo.frontend stdout) | per request | kv-router worker selection: `logit` (cost), `dp_rank`, `overlap_blocks`; `ttft_ms`/`elapsed_ms` per request; WARN stale-request → `frontend_*.log` |
| /metrics scrape 3 Hz (theo benchmark_phase) | `sglang:*` gauges/counters theo dp_rank | #queue-req, #running-req, cache hit, retraction, stage seconds — endpoint DYN_SYSTEM_PORT **8081** (frontend 8888 không có) → `server_metrics_export.jsonl` |
| GPU sampler | nvidia-smi 1 s (benchmark + drain window) | util/mem/power/clock mỗi GPU → `gpu_samples.csv` |
| schedule-eval | per request | queue_wait, matched_prefix_tokens, host_hit, in-batch dedup → `schedule_eval.jsonl` |
| py-spy | 25 Hz × 120 s × 8 scheduler, attach ở steady-state +120 s | flame CPU scheduler; attach được nhờ prctl(PR_SET_PTRACER_ANY) qua env `SGLANG_ALLOW_PTRACE=1` (node yama ptrace_scope=1) → `profiling/pyspy_*.svg` |
| torch profiler | — | không khả dụng trên stack dynamo (frontend không proxy `/start_profile`) |
| bottleneck_report.py | cây quyết định Q1–Q7 | verdict tổng hợp: GPU busy, client/server gap, TTFT front path, rank balance, KV, spec, hicache |


## 4. Kết quả bottleneck

Cell `agentx_dynamo/profiling/dfs-weight/c64_20261014_112105`.

### 4.1 Tổng quan client (aiperf, 1169 records)

| Chỉ số | Giá trị |
|---|---|
| Request / output throughput | 32.3 req/min / 522 tok/s (total 51.2k tok/s) |
| TTFT mean / p95 / p99 | 7.12 s / 20.5 s / **75.4 s** |
| Queue-wait (server) | mean 4.8 s (warm) / 3.9 s (cold); **p99 ≈ 247 s** |
| Starvation rate | ~10% (ngưỡng 10 s) |

### 4.2 Bottleneck

**#1 — Batch decode chỉ 1–4 request/rank + chi phí cố định MoE all-to-all là
nút thắt throughput.**

*Hiện tượng:* mỗi bước decode, scheduler mỗi rank gom toàn bộ request đang
chạy của rank đó thành một batch rồi forward một lượt. Log cho thấy
`#running-req: 1–4` — mỗi bước chỉ 1–4 request (~4–16 token với EAGLE draft 4)
dù trần admission là 256/rank.

*Vì sao batch nhỏ:* concurrency 64 của workload chia đều qua 8 attn-DP rank
≈ 8 request/rank tại một thời điểm; trừ lúc request đang chờ/kẹt prefill thì
còn ít hơn.

*Cơ chế MoE a2a:* GLM-5.3 là MoE chạy EP=8 — expert của token nằm rải trên
8 rank, mỗi bước decode rank phải trao đổi all-to-all: `dispatch_b` gửi
token tới rank sở hữu expert → expert tính → `combine_b` gom kết quả về
(DeepEP). Chi phí một lần trao đổi gần như cố định: 4 token hay 400 token
vẫn phải exchange đủ 8 rank + sync + round-trip, cộng kernel launch và
EAGLE verify 3 bước. Tok/s/rank ≈ token_mỗi_bước / (chi_phí_cố_định +
chi_phí_theo_token) — batch mỏng làm tử số nhỏ, mẫu số gần nguyên cố định
→ a2a chạy **latency-bound** thay vì bandwidth-bound, throughput tụt
27–317 tok/s/rank.

Flame py-spy 8/8 scheduler (`profiling/pyspy_*.svg`, inclusive, TB 8 rank)
xác nhận cùng chuyện:

| % samples/scheduler | Frame |
|---|---|
| ~52% | `forward_impl` MoE (`moe/ep_moe/layer.py:267`) |
| ~25% | DeepEP `dispatch_b` (`token_dispatcher/deepep.py`) qua TBO |
| ~11.5% | model forward (`deepseek_v2.py`) |

Engine log (`engine_*.log`, dòng `Decode batch` steady-state):

> `Decode batch, #running-req: 4, #token: 312576, token usage: 0.54, accept len: 2.16, ..., gen throughput (token/s): 35.88, fwd occupancy: 58.90%`

> `Decode batch, #running-req: 1, ... gen throughput (token/s): 27.04, ...`

(11:51:37 / 11:51:58)

**#2 — Đuôi TTFT nằm ở admission/prefill, không phải engine chậm.**
TTFT p99 75 s + queue-wait p99 247 s đối lập `#queue-req: 0` lúc decode —
request chờ ở prefill backlog:

> `Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, ..., token usage: 0.83, #pending-token: 44029, input throughput (token/s): 7172.36, fwd occupancy: 96.15%`

(engine_*.log, 12:00:11, benchmark window) — chunk 4096/rank, backlog ~44k
token chờ cắt; pending backlog 5–112k token **suốt** benchmark (per-minute
tổng hợp). *Cơ chế:* không phải "ưu tiên request nặng decode" —
prefill-delayer + dfs-weight **bảo vệ request đang decode** (prefill chen
vào làm ITL của request đang chạy giật) nên hoãn prefill mới; request input
dài/ít cache hit nằm backlog (hàng admission rỗng — `#queue-req` ~0–0.8 —
mới có mâu thuẫn với queue-wait p99 247 s). Lưu ý: fwd occupancy cao
(88–95%) — GPU "bận" nhưng bận với batch decode nhỏ (2.6–4.6 running-req,
latency-bound như #1): bận ≠ hiệu quả, và prefill chờ là chọn lối policy
bảo vệ ITL, không phải cạn tài nguyên.

**#3 — EAGLE spec decode khỏe, không phải nguồn bottleneck.**

> `accept len: 2.58, accept rate: 0.53` (engine_*.log, 11:52:37) — ≥ ngưỡng 2.0 mọi mẫu.

**#4 — Memory áp lực muộn run nhưng không phải nút thắt chính.**
Hicache host pool đúng 128 GB/rank — boot, engine_*.log, 8 rank:

> `11:25:06 INFO base.__init__: Allocating kv hierarchical KV host pool: 2469952 tokens, 128.00 GB host memory`

Token usage (field trong mọi dòng batch của engine_*.log, 9.518 dòng) chạm
1.00 muộn run: 486 dòng ≥ 0.95 (5%), 115 dòng ≥ 0.99, tập trung 12:00–12:21;
kèm 3 retract đơn lẻ:

> `12:10:11 WARN scheduler.update_running_batch: KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 192`

(engine_*.log; thêm 12:18:16 gained 1408, 12:20:06 gained 256) — workload
256K-context nặng memory thật; hicache offload giữ ổn định, retract rải rác
không sập throughput.

**#5 — Routing/frontend path không phải bottleneck.**
Frontend HTTP rẻ — `frontend_*.log`, dòng `http response sent`:
`latency_ms` 6436×0 ms / 4240×1 / 188×2 / 36×3 / 7×4. TTFT đo ở frontend
− TTFT engine = 0.26 s median (mean 0.42 s): join theo timestamp 1.151
request giữa `ttft_ms` trong dòng `request completed` (frontend_*.log) và
`ttft_s` trong `schedule_eval.jsonl` (request_id frontend ≠ rid engine nên
không join trực tiếp được). KV-router phân bổ lệch về số request
(84–221/rank sau warmup — đếm `dp_rank` trong dòng `Selected worker ... logit=`,
frontend_*.log) nhưng theo cost, không phải hot-rank: rank nhận ít request
mang input lớn nhất (rank 0: 107K token trung bình → queue-wait 9.6 s), rank
nhiều request nhỏ input nhất (rank 3/7: 79–85K → TTFT ~4 s) — độ lệch rank
là hệ quả của #2, không phải router thất thường. WARN (frontend_*.log):
25 `Expiring stale request` (ví dụ 11:56:37), 1 `Failed to mark prefill
completed` (12:21:04).

### 4.3 Giới hạn đo (ghi nhận)

- Không torch-profiler: frontend dynamo không proxy `/start_profile`, engine
  không có mặt HTTP riêng → flame py-spy + gpu_samples 1 s thay thế.
- `cached_tokens` vắng trong usage (frontend không thông qua field dù engine
  bật --enable-cache-report) → prompt-cache client metrics trống; thay bằng
  `matched_prefix_tokens` trong schedule_eval.jsonl.
- Counter sglang:* bị reset lặp trong run (publisher re-register, không phải
  restart) → counter server-side có thể undercount; latency client không ảnh hưởng.
- Frontend 8888/metrics **đã được scrape trong cell này** (aiperf tự lấy
  `/metrics` của `--url` 8888 cộng `--server-metrics` 8081 — 9.588 record
  mỗi endpoint trong `server_metrics_export.jsonl`): đủ `dynamo_frontend_*`
  (TTFT/ITL, `kv_hit_rate`, gauge per-worker) lẫn `sglang:*` per dp_rank.

### 4.4 Artifacts (run dir `c64_20261014_112105`)

Phân tích sâu theo 2 hướng định hướng mentor (scheduling policy + routing
imbalance, kèm health-check GPU/token/cache/KV): `docs/focus3320_routing_scheduling.md`.

`profile_export.jsonl` (1169 records) · `server_metrics_export.jsonl` (4.4 GB,
hai endpoint 8081+8888, 9.588 scrape mỗi cái, phân tích qua slurm) ·
`gpu_samples.csv` (1 s) · `schedule_eval.jsonl` (1237) ·
`profiling/pyspy_*.svg` ×8 · `e2_metrics.json`/`run_meta.json` ·
symlink `engine_*.log`/`frontend_*.log` · `analysis/` (time series per rank
trích từ metrics export).
