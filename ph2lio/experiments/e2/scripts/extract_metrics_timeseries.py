#!/usr/bin/env python3
"""Stream server_metrics_export.jsonl (4.4 GB per run) on a compute node and
extract per-dp_rank time series. Works for any profiling run dir.

Both scrape endpoints are in the file (aiperf scrapes BOTH --url 8888 and
--server-metrics 8081):
  - 8081  sglang:* gauges/labels carry dp_rank -> engine-side per-rank state
  - 8888  dynamo_frontend_* / dynamo_router_* -> frontend/router-side state

Output (long-format CSVs + a name inventory), under <run>/analysis/:
  sglang_rank_ts.csv      timestamp,metric,dp_rank,value
  dynamo_ts.csv           timestamp,metric,key,value   (key = dp_rank/worker/label-summary)
  metric_names_<ep>.txt   full metric-name inventory per endpoint

Run (from the login node):
  srun -p hardware-hcm -w hgx-h200-01 --cpus-per-task=8 --mem=16G \
    python3 -I <this script> <run_dir>
"""
import json, os, sys
from collections import defaultdict

# default = cell 3320 (historical); any run dir may be passed as argv[1]
RUN = (sys.argv[1] if len(sys.argv) > 1 else
       "/shared/loipp/sglang/ph2lio/experiments/e2/results/glm53/agentx_dynamo/profiling/dfs-weight/c64_20261014_112105")
SRC = f"{RUN}/server_metrics_export.jsonl"
OUT = f"{RUN}/analysis"
os.makedirs(OUT, exist_ok=True)

SGLANG_KEEP = {
    "sglang:num_running_reqs", "sglang:num_queue_reqs",
    "sglang:num_prefill_inflight_queue_reqs", "sglang:num_prefill_bootstrap_queue_reqs",
    "sglang:pending_prealloc_token_usage", "sglang:full_token_usage",
    "sglang:kv_used_tokens", "sglang:kv_available_tokens",
    "sglang:hicache_host_used_tokens", "sglang:hicache_host_total_tokens",
    "sglang:cache_hit_rate", "sglang:gen_throughput",
    "sglang:num_retracted_reqs", "sglang:num_paused_reqs",
    "sglang:prefill_delayer_outcomes", "sglang:prefill_effective_tokens",
}
DYNAMO_KEEP_PREFIX = ("dynamo_router_", "dynamo_frontend_worker_",
                      "dynamo_frontend_router_queue",
                      "dynamo_frontend_kv_hit_rate", "dynamo_frontend_ttft",
                      "dynamo_frontend_inter_token", "dynamo_frontend_cached_tokens")

names_seen = {}
f_sg = open(f"{OUT}/sglang_rank_ts.csv", "w")
f_dy = open(f"{OUT}/dynamo_ts.csv", "w")
f_sg.write("timestamp,metric,dp_rank,value\n")
f_dy.write("timestamp,metric,key,value\n")

n_lines = 0
with open(SRC) as f:
    for line in f:
        n_lines += 1
        rec = json.loads(line)
        url = rec["endpoint_url"]
        is_engine = ":8081" in url.rsplit("/", 1)[0]  # port part before /metrics
        ts = rec["timestamp_ns"] / 1e9
        inv = names_seen.setdefault("8081" if is_engine else "8888", set())
        for name, samples in rec.get("metrics", {}).items():
            inv.add(name)
            if is_engine:
                if name not in SGLANG_KEEP:
                    continue
                for s in samples:
                    rank = (s.get("labels") or {}).get("dp_rank", "-1")
                    f_sg.write(f"{ts:.1f},{name},{rank},{s.get('value', 0)}\n")
            else:  # 8888 frontend
                if not name.startswith(DYNAMO_KEEP_PREFIX):
                    continue
                for s in samples:
                    lab = s.get("labels") or {}
                    key = lab.get("dp_rank", lab.get("worker_id", lab.get("router_id", "")))
                    if not name.startswith("dynamo_frontend_router_queue") and len(lab) > 1:
                        # keep a compact label signature for multi-label metrics
                        extras = ",".join(f"{k}={v}" for k, v in lab.items()
                                          if k not in ("dp_rank", "worker_id", "router_id", "model"))
                        key = f"{key}|{extras}" if extras else key
                    if "buckets" in s:  # histogram: store sum/count only
                        f_dy.write(f"{ts:.1f},{name}_sum,{key},{s.get('sum', 0)}\n")
                        f_dy.write(f"{ts:.1f},{name}_count,{key},{s.get('count', 0)}\n")
                    else:
                        f_dy.write(f"{ts:.1f},{name},{key},{s.get('value', 0)}\n")

f_sg.close(); f_dy.close()
for tag, inv in names_seen.items():
    with open(f"{OUT}/metric_names_{tag}.txt", "w") as g:
        g.write("\n".join(sorted(inv)) + "\n")
print(f"lines: {n_lines}")
print(f"names: " + ", ".join(f"{tag}: {len(inv)}" for tag, inv in names_seen.items()))
print(f"out: {OUT}")
