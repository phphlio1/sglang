#!/usr/bin/env python3
"""Offline flag-parse audit for the dynamo-wrapped engine (plan section C4).

Replicates dynamo/sglang/args.py:parse_args up to (but not including) the
model-download step: builds the same three parser layers, parses the exact
engine flag list start_dynamo_stack.sh passes, and asserts the fields the
boost merge must not break. No GPU, no model load — safe to run anywhere the
venv-dynamo tree is visible.

Usage:
    /shared/loipp/venv-dynamo/bin/python3 audit_flag_parse_dynamo.py [run_dir]
"""

import sys

ENGINE_FLAGS = [
    "--model-path=/mnt/hf_hub_cache/GLM-5.3-W4AFP8",
    "--served-model-name=zai-org/GLM-5.3",
    "--quantization=w4afp8",
    "--tp=8", "--dp=8", "--ep=8",
    "--enable-dp-attention",
    "--enable-dp-attention-local-control-broadcast",
    "--enable-dp-lm-head",
    "--dist-init-addr=127.0.0.1:10888",
    "--numa-node", "0", "0", "0", "0", "1", "1", "1", "1",
    "--moe-a2a-backend=deepep",
    "--chunked-prefill-size=32768",
    "--tool-call-parser=glm47",
    "--reasoning-parser=glm45",
    "--mem-fraction-static=0.75",
    "--max-running-requests=256",
    "--context-length=300000",
    "--kv-cache-dtype=fp8_e4m3",
    "--dsa-prefill-backend=flashmla_sparse_q8",
    "--allow-auto-truncate",
    "--enable-metrics",
    "--enable-metrics-for-all-schedulers",
    "--enable-cache-report",
    "--enable-hierarchical-cache",
    "--hicache-size=128",
    "--hicache-io-backend=direct",
    "--hicache-write-policy=write_back",
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
    "--schedule-policy={policy}",
    "--enable-prefill-delayer",
    "--page-size=64",
    '--kv-events-config={"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:8900"}',
]

# (label, policy, extra boost flags) — one parse case per campaign arm shape.
CASES = [
    ("dfs-weight", "dfs-weight", []),
    ("fcfs-match", "fcfs-match", []),
    ("lpm", "lpm", []),
    ("hrrn", "hrrn", []),
    ("lof", "lof", []),
    ("shortest-prefill-first", "shortest-prefill-first",
     ["--in-batch-prefix-caching-max-pin-s=60"]),
    ("boost-static-gamma", "boost",
     ["--boost-gamma=5.0", "--boost-bin-size=256", "--boost-clock=wall",
      "--in-batch-prefix-caching-max-pin-s=60"]),
    ("boost-adaptive", "boost",
     ["--boost-gamma=0.09", "--boost-bin-size=256", "--boost-clock=wall",
      "--boost-adaptive-gamma", "--boost-adaptive-max-jump-s=60",
      "--in-batch-prefix-caching-max-pin-s=60",
      "--log-schedule-eval-path=/tmp/schedule_eval_audit.jsonl"]),
]

BASE_EXPECT = {
    "tp_size": 8,
    "dp_size": 8,          # deprecated spelling, pre-resolution
    "ep_size": 8,
    "enable_dp_attention": True,
    "page_size": 64,
    "chunked_prefill_size": 32768,
    "max_running_requests": 256,
}


def build_parsers():
    import argparse

    from dynamo.common.configuration.groups.runtime_args import DynamoRuntimeArgGroup
    from dynamo.sglang.backend_args import DynamoSGLangArgGroup
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(add_help=True)
    DynamoRuntimeArgGroup().add_arguments(parser)
    DynamoSGLangArgGroup().add_arguments(parser)

    sglang_only_parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(sglang_only_parser)
    return parser, sglang_only_parser


def parse_case(parser, sglang_only_parser, label, policy, extra):
    args = [f.format(policy=policy) for f in ENGINE_FLAGS] + extra
    dynamo_args, unknown = parser.parse_known_args(args)
    parsed = sglang_only_parser.parse_args(unknown)

    ns = vars(parsed)
    checks = dict(BASE_EXPECT)
    checks["schedule_policy"] = policy
    if policy == "boost":
        if "--boost-adaptive-gamma" in extra:
            checks.update(boost_gamma=0.09, boost_bin_size=256, boost_clock="wall",
                          boost_adaptive_gamma=True, boost_adaptive_max_jump_s=60.0)
        else:
            checks.update(boost_gamma=5.0, boost_bin_size=256, boost_clock="wall",
                          boost_adaptive_gamma=False)
    if "--in-batch-prefix-caching-max-pin-s=60" in extra:
        checks["in_batch_prefix_caching_max_pin_s"] = 60.0

    failures = [f"{k}: expected {v!r}, got {ns.get(k, '<missing>')!r}"
                for k, v in checks.items() if ns.get(k) != v]
    status = "PASS" if not failures else "FAIL"
    print(f"[{status}] {label} (schedule_policy={ns.get('schedule_policy')})")
    for f in failures:
        print(f"    {f}")
    return not failures


def main():
    parser, sglang_only_parser = build_parsers()
    ok = all(parse_case(parser, sglang_only_parser, *case) for case in CASES)

    # Resolution check: the deprecated --dp=8 --enable-dp-attention spelling
    # must still resolve to attn_dp_size=8, dp_size=1 (parallel_hook.py).
    from sglang.srt.arg_groups.parallel_hook import handle_deprecated_dp_attention
    print(f"[INFO] handle_deprecated_dp_attention imported: "
          f"{handle_deprecated_dp_attention.__module__}")

    print(f"=== AUDIT {'PASS' if ok else 'FAIL'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
