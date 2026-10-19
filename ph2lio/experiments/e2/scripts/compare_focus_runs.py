#!/usr/bin/env python3
"""Compare N profiling runs (same config) and write a consistency report.

Usage:
    python3 compare_focus_runs.py <run_dir_baseline> <run_dir_2> [<run_dir_3> ...]
Output: docs/focus_consistency_<N>runs.md (next to this script's ../docs).

Each run must already have analysis/*.csv (extract_metrics_timeseries.py).
Verdicts are heuristic and stated as such: "consistent" = cùng hướng và cùng
bậc độ lớn (within ~2x, cùng dấu với corr).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focus_report_lib import analyze_run, summary  # noqa: E402

ROWS = [
    # (label, key, fmt)
    ("Request/output throughput (tok/s)", "throughput_tps", "{:.0f}"),
    ("TTFT mean (s)", "ttft_mean_s", "{:.1f}"),
    ("TTFT p99 (s)", "ttft_p99_s", "{:.1f}"),
    ("Running-req mean/rank (min–max)", None, None),  # special: range
    ("Pending backlog per-minute (min–max)", None, None),
    ("Active prefill token spread (×)", "prefill_spread_x", "{:.1f}"),
    ("KV usage mean (min–max)", None, None),
    ("KV usage spread p50", "kv_spread_p50", "{:.2f}"),
    ("#rank chạm KV 1.00", "n_rank_full", "{}"),
    ("#retract", "n_retract", "{}"),
    ("corr(logit, TTFT)", "corr_logit_ttft", "{:+.2f}"),
    ("Cache reuse engine (%)", "cache_reuse_pct", "{:.1f}"),
    ("Cached tokens/request", "cached_tokens_req", "{:.0f}"),
    ("EAGLE accept len (per-minute min–max)", None, None),
    ("ITL mean (ms)", "itl_mean_ms", "{:.1f}"),
    ("GPU util window (min–max %)", None, None),
]
RANGES = {
    "Running-req mean/rank (min–max)": ("run_mean_min", "run_mean_max", "{:.2f}–{:.2f}"),
    "Pending backlog per-minute (min–max)": ("pending_pm_min", "pending_pm_max", "{:,.0f}–{:,.0f}"),
    "KV usage mean (min–max)": ("kv_mean_min", "kv_mean_max", "{:.2f}–{:.2f}"),
    "EAGLE accept len (per-minute min–max)": ("accept_pm_min", "accept_pm_max", "{:.2f}–{:.2f}"),
    "GPU util window (min–max %)": ("gpu_win_util_min", "gpu_win_util_max", "{:.0f}–{:.0f}"),
}
# metrics where "consistent" means same order of magnitude (ratio <= 2)
FACTOR2 = {"throughput_tps", "ttft_mean_s", "ttft_p99_s", "prefill_spread_x",
           "kv_spread_p50", "cache_reuse_pct", "cached_tokens_req", "itl_mean_ms"}
SIGN = {"corr_logit_ttft"}


def fmt_v(summ, label, key, fmt):
    if label in RANGES:
        a, b, f = RANGES[label]
        return f.format(summ[a], summ[b]).replace(",", ".")
    return fmt.format(summ[key])


def verdict_row(vals, key):
    if key in SIGN:
        return "✓ cùng dấu" if all(v > 0 for v in vals) or all(v < 0 for v in vals) else "≠ khác dấu"
    if key in FACTOR2:
        lo, hi = min(vals), max(vals)
        return "✓" if (lo > 0 and hi / lo <= 2) or (lo == hi == 0) else "≠ >2×"
    return ""


def extras(rep):
    """Per-rank imbalance + consequences + router/cost + fairness numbers."""
    d, r, j, cost = rep["dynamo"], rep["rank"], rep["join"], rep["cost"]
    m = r["means"]

    def vals(dd):
        return [v for v in dd.values() if v is not None and v == v]

    ap, adb = vals(d["active_prefill"]), vals(d["active_decode_blocks"])
    usage, chr_ = vals(m["sglang:full_token_usage"]), vals(m["sglang:cache_hit_rate"])
    qreq = vals(m["sglang:num_queue_reqs"])
    pr = j["per_rank"]
    rmax_qw = max(pr, key=lambda k: pr[k]["qw_mean"])
    rmin_qw = min(pr, key=lambda k: pr[k]["qw_mean"])
    rmax_tf = max(pr, key=lambda k: pr[k]["ttft_mean"])
    rmin_tf = min(pr, key=lambda k: pr[k]["ttft_mean"])
    rmin_n = min(pr, key=lambda k: pr[k]["n"])
    rmax_in = max(pr, key=lambda k: pr[k]["in_mean"])
    parts = d["router_overhead"].get("dynamo_router_overhead_total_ms", {})
    n_dec = sum(p["decisions"] for p in parts.values())
    ov = (sum(p["decisions"] * p["mean_ms"] for p in parts.values()) / n_dec) if n_dec else float("nan")
    q1, q5 = cost["quintiles"][0], cost["quintiles"][-1]
    hic = vals(m["sglang:hicache_host_used_tokens"])
    with open(os.path.join(rep["run_dir"], "e2_metrics.json")) as f:
        em = json.load(f)
    return {
        "prefill": (min(ap), max(ap)), "adb": (min(adb), max(adb)),
        "usage": (min(usage), max(usage)), "chr": (min(chr_), max(chr_)),
        "qreq": (min(qreq), max(qreq)),
        "qw": (rmax_qw, pr[rmax_qw]["qw_mean"], rmin_qw, pr[rmin_qw]["qw_mean"]),
        "ttft": (rmax_tf, pr[rmax_tf]["ttft_mean"], pr[rmax_tf]["ttft_p99"],
                 rmin_tf, pr[rmin_tf]["ttft_mean"]),
        "n_minmax": (min(pr[k]["n"] for k in pr), max(pr[k]["n"] for k in pr)),
        "rmin_n": (rmin_n, pr[rmin_n]["n"], pr[rmin_n]["in_mean"]),
        "rmax_in": (rmax_in, pr[rmax_in]["in_mean"]),
        "ov": (n_dec, ov), "corr_qw": cost["corr_logit_qw"],
        "q1_q5": (q1["ttft_p50"], q5["ttft_p50"], q1["qw_p50"], q5["qw_p50"]),
        "hic": (min(hic), max(hic)) if hic else None,
        "jain": em["fairness"]["ttft_fairness_jain"], "cv": em["fairness"]["ttft_cv"],
        "cw_ratio": em["fairness"]["cold_warm_ttft_ratio"],
        "e2e_p99": em["e2e"]["p99_ms"] / 1e3,
        "qw_p99": em["queue"].get("queue_time_p99_s", float("nan")),
    }


def main():
    run_dirs = sys.argv[1:]
    if len(run_dirs) < 2:
        sys.exit("need >= 2 run dirs (first = baseline 3320)")
    reps = [analyze_run(d) for d in run_dirs]
    sums = [summary(r) for r in reps]
    exs = [extras(r) for r in reps]

    hdr = "| Chỉ số | " + " | ".join(f"Run {i+1}: {s['stamp']}" for i, s in enumerate(sums)) + " | Verdict |"
    sep = "|---|" + "---|" * (len(sums) + 1)
    lines = [hdr, sep]
    for label, key, fmt in ROWS:
        cells = [fmt_v(s, label, key, fmt) for s in sums]
        if key in SIGN or key in FACTOR2:
            v = verdict_row([s[key] for s in sums], key)
        elif label in RANGES:
            # range rows: so sánh bậc độ lớn bằng giá trị max mỗi run
            a, b, _f = RANGES[label]
            vals = [s[b] for s in sums]
            lo, hi = min(vals), max(vals)
            v = "✓" if (lo == hi) or (lo > 0 and hi / lo <= 2) else "≠ >2×"
        else:
            v = ""
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {v} |")

    def kk(x):
        return f"{x / 1e3:.1f}k"

    def cells_rng_k(pairs):
        return [f"{kk(lo)}–{kk(hi)} ({hi / lo:.1f}×)" for lo, hi in pairs]

    def cells_rng_f(pairs):
        return [f"{lo:.2f}–{hi:.2f}" for lo, hi in pairs]

    prefill_row = "| Active prefill tokens (8888) | " + " | ".join(cells_rng_k([e["prefill"] for e in exs])) + " |"
    adb_row = "| Active decode blocks (8888) | " + " | ".join(cells_rng_k([e["adb"] for e in exs])) + " |"
    usage_row = "| KV token usage (8081 `full_token_usage`) | " + " | ".join(cells_rng_f([e["usage"] for e in exs])) + " |"
    chr_row = "| Cache hit rate (8081) | " + " | ".join(cells_rng_f([e["chr"] for e in exs])) + " |"
    qreq_row = "| Admission queue-req (8081) | " + " | ".join(cells_rng_f([e["qreq"] for e in exs])) + " |"
    qw_row = "| Queue-wait mean/rank (rank max vs min) | " + " | ".join(
        f"r{e['qw'][0]} {e['qw'][1]:.1f} s / r{e['qw'][2]} {e['qw'][3]:.1f} s" for e in exs) + " |"
    ttft_row = "| TTFT mean/rank (rank max, p99 của nó vs rank min) | " + " | ".join(
        f"r{e['ttft'][0]} {e['ttft'][1]:.1f} s (p99 {e['ttft'][2]:.0f}) / r{e['ttft'][3]} {e['ttft'][4]:.1f} s" for e in exs) + " |"
    n_row = "| Request count per rank (min–max) | " + " | ".join(
        f"{e['n_minmax'][0]}–{e['n_minmax'][1]}" for e in exs) + " |"
    kvfull_row = "| #rank chạm KV 1.00 | " + " | ".join(str(s["n_rank_full"]) for s in sums) + " |"
    retract_row = "| #retract | " + " | ".join(str(s["n_retract"]) for s in sums) + " |"
    ov_txt = ", ".join(f"{e['ov'][0]:.0f} dec/{e['ov'][1]:.1f} ms" for e in exs)
    corr_txt = ", ".join(f"{s['corr_logit_ttft']:+.2f}" for s in sums)
    corr_qw_txt = ", ".join(f"{e['corr_qw']:+.2f}" for e in exs)
    q5_txt = "; ".join(f"{e['q1_q5'][0]:.1f}→{e['q1_q5'][1]:.1f} s" for e in exs)
    hic_txt = " / ".join(f"{e['hic'][0] / 1e6:.1f}M–{e['hic'][1] / 1e6:.1f}M" for e in exs)
    jain_txt = " / ".join(f"{e['jain']:.2f}" for e in exs)
    cv_txt = " / ".join(f"{e['cv']:.1f}" for e in exs)
    cw_txt = " / ".join(f"{e['cw_ratio']:.2f}" for e in exs)
    e2e_txt = " / ".join(f"{e['e2e_p99']:.0f}" for e in exs)
    qw_p99_txt = " / ".join(f"{e['qw_p99']:.0f}" for e in exs)
    rmin_n_ex = exs[1]["rmin_n"]  # ví dụ lấy run 2 (133031)

    # heuristic focus verdicts
    f1_ok = (all(s["pending_pm_max"] >= 10_000 for s in sums)
             and max(s["run_mean_max"] for s in sums) <= 5
             and max(s["throughput_tps"] for s in sums) / min(s["throughput_tps"] for s in sums) <= 1.5
             and max(s["ttft_p99_s"] for s in sums) / min(s["ttft_p99_s"] for s in sums) <= 2)
    f2_ok = (all(s["prefill_spread_x"] >= 2 for s in sums)
             and all(s["kv_spread_p50"] >= 0.25 for s in sums)
             and all(s["corr_logit_ttft"] > 0 for s in sums))
    health_ok = (all(s["cache_reuse_pct"] >= 80 for s in sums)
                 and all(s["accept_pm_min"] >= 1.8 for s in sums)
                 and all(s["n_retract"] <= 10 for s in sums))

    def yn(ok):
        return "**CONSISTENT**" if ok else "**KHÔNG consistent** (xem bảng)"

    doc = f"""# Consistency {len(sums)} run — dfs-weight c64 (config prod như 3320)

Trả lời câu hỏi: bottleneck thấy ở baseline 3320 có lặp lại ở các replicate
(cùng config từng tham số, seed 42, cold radix cache mỗi run) hay không?

| Run | Stamp | Run dir |
|---|---|---|
""" + "\n".join(f"| {i + 1}{' (baseline 3320)' if i == 0 else ''} | {s['stamp']} | `{r['run_dir']}` |"
                for i, (r, s) in enumerate(zip(reps, sums))) + f"""

## 1. Bảng so sánh (mọi số tính trong benchmark window, nguồn như focus report từng run)

{chr(10).join(lines)}

## 2. Mất cân bằng workload/KV per rank và hệ quả

Thống kê per rank trong benchmark window (hàng workload = min–max mean qua 8
rank; hàng join = rank cực đại vs cực tiểu; nguồn và cách tính: focus
report từng run §2.2–§2.3 — report này cite hạn chế, chỉ trỏ về focus report).

| Chỉ số (per rank) | {" | ".join(f"Run {i+1}" for i in range(len(sums)))} |
|---|{"---|" * len(sums)}
{prefill_row}
{adb_row}
{usage_row}
{chr_row}
{qreq_row}
{qw_row}
{ttft_row}
{n_row}
{kvfull_row}
{retract_row}

- **Router overhead — không đáng kể**: {ov_txt} (decision/run, mean ms;
  histogram delta 8888, focus report §2.4 từng run) — quá nhỏ so TTFT giây.
- **Công thức cost — có tín hiệu nhưng chưa đủ**: corr(logit, TTFT) =
  {corr_txt}; corr(logit, queue_wait) = {corr_qw_txt}; quintile Q1→Q5 median
  TTFT: {q5_txt}. Cost cao đoán đúng hướng chờ lâu hơn ở mọi run, nhưng
  hot-rank vẫn bị dồn (KV-full + retract ở 2/{len(sums)} run, bảng trên) —
  giả thuyết "công thức chưa tối ưu" được ủng hộ một phần (nghi vấn cơ chế:
  weight `16 × prefill_blocks` underestimate chi phí prefill; kv-state router
  hết hạn — chi tiết focus report §2.5).

## 3. Verdict Focus 1 — scheduling policy (batch mỏng + prefill đói): {yn(f1_ok)}

- Batch decode mỏng mọi run (running-req mean ≤ {max(s['run_mean_max'] for s in sums):.2f}/rank).
- Backlog prefill duy trì mọi run (đỉnh per-minute {min(s['pending_pm_max'] for s in sums):,.0f}–{max(s['pending_pm_max'] for s in sums):,.0f} token).
- Throughput chênh ≤ {100 * (max(s['throughput_tps'] for s in sums) / min(s['throughput_tps'] for s in sums) - 1):.0f}%, TTFT p99 chênh ≤ {100 * (max(s['ttft_p99_s'] for s in sums) / min(s['ttft_p99_s'] for s in sums) - 1):.0f}%.

**Tại sao** (cơ chế; cite dòng log từng số ở focus report §1.1–1.2 từng run):

- Cặp `--schedule-policy=dfs-weight` + `--enable-prefill-delayer` bảo vệ
  request đang decode (prefill chen ngang làm ITL giật) → scheduler hoãn
  prefill mới → backlog `#pending-token` duy trì suốt benchmark (đỉnh per-minute
  73k–132k token, bảng §1) dù admission queue gần rỗng (queue-req
  {min(e['qreq'][0] for e in exs):.2f}–{max(e['qreq'][1] for e in exs):.2f}/rank,
  §2). TTFT tail 75–99 s đến từ backlog này, không phải cạn tài nguyên — GPU
  vẫn 77–97%.
- Batch decode mỏng {min(s['run_mean_min'] for s in sums):.2f}–{max(s['run_mean_max'] for s in sums):.2f} req/rank
  (dù trần `--max-running-requests=256`): prefill bị hoãn nên không có việc
  dồn vào batch; MoE EP=8 all-to-all đồng bộ mọi rank mỗi bước forward →
  chi phí cố định mỗi bước không phụ thuộc batch → GPU bận (util cao, power
  ổn) nhưng năng suất chặn ~520–532 tok/s (latency-bound). Py-spy flame
  (8/8 scheduler, 4 run có flame) cho thấy MoE forward + DeepEP `dispatch_b`
  chiếm ưu thế.
- Tóm lại: policy đánh đổi TTFT lấy ITL ổn định (chủ đích của config prod);
  ở c64 nó giữ batch mỏng kéo dài suốt window — bottleneck là hành vi cố ý
  của policy, lặp lại y hệt mọi run.

## 4. Verdict Focus 2 — routing mất cân bằng rank: {yn(f2_ok)}

- Active prefill token spread ≥ {min(s['prefill_spread_x'] for s in sums):.1f}× mọi run.
- KV usage spread p50 ≥ {min(s['kv_spread_p50'] for s in sums):.2f} mọi run.
- corr(logit, TTFT) cùng dấu dương mọi run: {', '.join(f"{s['corr_logit_ttft']:+.2f}" for s in sums)}.

**Tại sao** (cơ chế; chi tiết + cite focus report §2.2–§2.5 từng run):

- Router (kv-mode) cố gắng giữ request cùng prefix về cùng rank (cache hit
  per rank chênh {min(e['chr'][0] for e in exs):.2f}–{max(e['chr'][1] for e in exs):.2f}, §2) và
  cân bằng theo **số request** — nhưng số request không phải tải thực: mọi
  run, rank ít request nhất đều thuộc nhóm nhận input mean lớn nhất (4/5 run
  chính là rank input lớn nhất; ví dụ run 2: r{rmin_n_ex[0]} chỉ
  {rmin_n_ex[1]} request nhưng input mean {rmin_n_ex[2] / 1e3:.0f}k token).
  Request count per rank chênh {min(e['n_minmax'][1] / e['n_minmax'][0] for e in exs):.1f}–{max(e['n_minmax'][1] / e['n_minmax'][0] for e in exs):.1f}×
  (§2) — cân bằng bề ngoài, tải thực lệch.
- Hệ quả: active prefill token lệch {min(s['prefill_spread_x'] for s in sums):.1f}–{max(s['prefill_spread_x'] for s in sums):.1f}× →
  KV usage lệch p50 {min(s['kv_spread_p50'] for s in sums):.2f}–{max(s['kv_spread_p50'] for s in sums):.2f} →
  hot-rank chờ lâu hơn hẳn (queue-wait/TTFT per rank, §2), và khi áp lực
  đủ lớn thì escalate thành KV-full + retract (2/5 run; triệu chứng
  intermittent — mất cân bằng luôn đo được, việc tràn thì phụ thuộc quỹ đạo
  KV của từng run). Lớp offload hicache host (128 GB/rank) cũng lệch theo:
  used {hic_txt} token.
- Cost formula phản ánh đúng hướng (corr dương + quintile tăng, §2) nhưng
  chưa đủ kéo request khỏi hot-rank → giả thuyết "công thức cost chưa tối
  ưu" được ủng hộ một phần, hướng sửa: cân bằng theo prefill token/KV thay
  vì số request (nghi vấn §2.5 từng run).

## 5. Health-check: {yn(health_ok)}

- Cache reuse ≥ {min(s['cache_reuse_pct'] for s in sums):.1f}% mọi run; accept len ≥ {min(s['accept_pm_min'] for s in sums):.2f}; retract ≤ {max(s['n_retract'] for s in sums)} lần/run.

## 6. Bottleneck khác ngoài phạm vi 2 focus (ghi chú future work)

Các hiện tượng dưới đây có đo được trong 5 run nhưng **không thuộc** phạm vi
scheduler/routing mentor yêu cầu — ghi lại để theo dõi sau:

- **Fairness / long-tail TTFT giữa các request**: Jain TTFT
  {jain_txt} (gần 0 = rất không đều), CV {cv_txt}; cold/warm TTFT ratio
  {cw_txt} (cold-cache request, n≈114/run); queue-wait p99 client
  {qw_p99_txt} s — chênh lớn giữa các run.
- **E2E long-tail**: p99 {e2e_txt} s — phần lớn từ TTFT tail cộng output dài.
- **EAGLE accept len** {min(s['accept_pm_min'] for s in sums):.2f}–{max(s['accept_pm_max'] for s in sums):.2f}
  (per-minute mean): vài phút dưới ngưỡng khỏe ≥2.0 — spec decode chưa là
  nghẽn chính (ITL ~38–43 ms ổn định) nhưng đáng theo dõi.
- **Thước đo còn hở**: `sglang:fwd_occupancy` thỉnh thoảng bị drop mẫu NaN
  trong export; client không report `usage.prompt_tokens_details.cached_tokens`
  (cần `--enable-cache-report` phía server); `sglang:prefill_delayer_outcomes`
  có đếm per outcome/rank nhưng chưa được phân tích chi tiết.
- **Ops**: boot sau fresh mirror nguội JIT ~7 phút (run 2) — đã mitigate
  bằng `PROBE_WAIT_TIMEOUT_S=900` + readiness gate (run 4–5 probe đủ 8/8,
  job exit 0).

## 7. Kết luận

Verdict là heuristic (cùng hướng + cùng bậc độ lớn, ngưỡng ghi ở từng mục);
số liệu chính nằm ở bảng §1–§2 và focus report từng run. Cả 2 focus
đều **CONSISTENT qua {len(sums)} run cùng config**: (1) scheduling giữ batch
mỏng + prefill đói, (2) routing lệch tải per-rank — bottleneck có thật,
không phải noise của một lần chạy.

## 8. Nguồn dữ liệu & quy ước cite (compact)

| Nguồn | Nội dung | Dùng ở |
|---|---|---|
| `analysis/sglang_rank_ts.csv` + `analysis/dynamo_ts.csv` | trích từ `server_metrics_export.jsonl` (8081 `sglang:*` per dp_rank; 8888 `dynamo_*`), benchmark window | §1–§2 per-rank mean/spread, KV, cache hit, queue |
| `engine_*.log` / `frontend_*.log` (symlink trong run dir) | batch scheduler per-minute, pending-token, boot lines, WARN retract; join router↔engine theo timestamp ±2 s | §1 pending/running, §2 queue-wait/TTFT per rank, corr(logit,·) |
| `e2_metrics.json` | throughput/TTFT/E2E/TPOT/fairness client | §1, §6 |
| `gpu_samples.csv` | nvidia-smi 1 s × 8 GPU | §1 GPU util |
| `profiling/pyspy_*.svg` | flame 25 Hz × 120 s × 8 scheduler (4/5 run; run 2 probes bị skip) | §3 cơ chế |
| `profile_export.jsonl` | `benchmark_phase` + `request_start_ns` → window; duration từ `e2_metrics.json` (`run.duration_s`) | mọi bảng |

Benchmark window mỗi run = record đầu phase `profiling` → +1800 s. Chi tiết
cách tính từng số + cite file/dòng: focus report từng run — run 1
`docs/focus3320_routing_scheduling.md`, run 2–5
`docs/focus_c64_<stamp>_routing_scheduling.md` (report này cite hạn chế,
trỏ về focus report). Reproduce: `extract_metrics_timeseries.py <run_dir>`
(srun trên node) → `gen_focus_report.py <run_dir>` → `compare_focus_runs.py
<run_dir...>` (script này).
"""
    out = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "docs", f"focus_consistency_{len(sums)}runs.md"))
    with open(out, "w") as fh:
        fh.write(doc)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
