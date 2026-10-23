# e2A — boost-cache: bounded LPM-cache fusion into boost (1x RTX 6000 Pro)

Clone of the e2 frame for the first post-campaign policy experiment:
`boost-cache`, which adds a capped matched-prefix bonus (`lambda*min(matched,
M)`) to boost's phi so the policy recovers lpm-style cache reuse while
keeping boost's bounded-displacement anti-starvation. Design, code map,
sweep plan, and success criteria: `docs/e2a_design.md`. The e2 campaign this
builds on is archived under `../../archives/` (verdicts in
`docs/fixed_e2_rtx6000.md`).

## Layout

- `scripts/run_e2.sh` — the harness: boots the server (`start_server.sh`),
  drives the aiperf agent workload, and writes run dirs under
  `results/1x-rtx6000pro/agentx/<policy>/<label>_<ts>/`. Modes:
  `smoke | policy <policy> [conc] | sweep [conc...] | array`; policies:
  fcfs-match lpm hrrn dfs-weight boost. Every knob is an env var — see the
  usage block at the bottom of the file (boost arms: `BOOST_GAMMA` static,
  `BOOST_ADAPTIVE_GAMMA=1` + `BOOST_ADAPTIVE_MAX_JUMP_S` adaptive).
- `scripts/start_server.sh` — server boot for this rig (conda env `sglang`,
  CUDA shim at repo-root `.cuda_shim`, Qwen/Qwen3.8-27B-FP8, sm_120 flags).
  Policy-agnostic; policy/boost knobs flow in through `run_e2.sh`.
- `analysis/` — metrics pipeline. `agentx_metrics.py` is v1 and
  load-bearing: the harness itself calls it at the end of every run to write
  `e2_metrics.json`. `metrics_v2.py` is the additive pass
  (`e2_metrics_v2.json`), importing helpers from `starvation.py`.
  `final_tables.py` + `final_tables_manifest.json` build the campaign tables
  (the manifest is the authoritative list of the RTX campaign's 40 verified
  runs); `cross_check_metrics_v2.py` and `print_metrics_v2_table.py` render
  them; `test_metrics_v2.py` is the unit test.
- `docs/` — campaign record for this rig. Start with
  `fixed_e2_rtx6000.md` (final tables + verdicts), then
  `boost_tail_collapse_root_cause.md` and `metrics_v2.md` for methodology.

## Prerequisites (outside this tree)

- `../.venv-e2/` — the aiperf client venv (default `AIPERF` path in
  `run_e2.sh`; override with `AIPERF=...` if a clone needs its own).
- conda env `sglang`, the repo-root CUDA shim, and the model weights — all
  wired in `scripts/start_server.sh`.
- `results/` and `logs/` are generated at run time and git-ignored.

## Clone for a new experiment

    cd <repo>/ph2lio/experiments
    cp -r e2 e2A
    cd e2A && rm -rf results logs   # only if the source frame was used
    SEED=42 CONCURRENCY=16 bash scripts/run_e2.sh policy boost 16

Each clone owns its own `results/` tree and its own `docs/`; the venv and
the server environment are shared. Run new campaigns in a clone — keep this
frame clean so the next clone starts from a known-good baseline.

## Archived campaign data (not in git)

- `../../archives/e2_rtx6000_final40_full_20261024.tar` — the 40
  final-campaign run dirs (server exports gzip-compressed inside) plus the
  policy-level `schedule_eval*.jsonl`.
- `../../archives/e2_rtx6000_rest_small_20261024.tar.gz` — small files of
  every other complete run, plus the adaptive pilot and quarantine trees.
