# How to run E2

Single entrypoint: `scripts/run_e2.sh`. Serving config lives in
`scripts/start_server.sh` (user-maintained; policy is its `$1` argument —
`run_e2.sh` starts/reuses/restarts the server for you).

## Smoke test (pipeline check only — NOT a valid benchmark)

```bash
scripts/run_e2.sh smoke
```

300 s, 16 trace entries, `--unsafe-override`, stamped `submission_valid=false`.
First run also builds the dataset cache (~10-20 min); later runs are faster.

## Official runs

```bash
scripts/run_e2.sh policy <policy> [concurrency]   # e.g. policy dfs-weight 4
scripts/run_e2.sh sweep [concurrency...]           # all 5 policies (full experiment)
```

Policies: `fcfs lpm hrrn dfs-weight lof`. Default concurrency 4 (env
`CONCURRENCY`). Each point runs the full 1800 s scenario and lands in
`results/1x-rtx6000pro/agentx/<policy>/c<conc>_<ts>/` (never overwrites), with
`e2_metrics.json` computed automatically at the end.

## Slurm (later)

The script carries an inert `#SBATCH` header; on a cluster:

```bash
sbatch --array=0-4 scripts/run_e2.sh array   # task 0..4 -> fcfs lpm hrrn dfs-weight lof
sbatch scripts/run_e2.sh policy fcfs         # single point
```

All knobs are env vars (`--export=ALL,CONCURRENCY=8`):
`SGLANG_URL MODEL TOKENIZER CONCURRENCY DURATION SEED OUTPUT_DIR
SMOKE_POLICY SMOKE_DURATION NUM_ENTRIES HEALTH_TIMEOUT STOP_SERVER`.

Boost-arm knobs (also env vars, consumed by `start_server.sh`):

- `BOOST_CLOCK` (default `wall`): the boost clock. `wall` stamps arrival in
  wall seconds and converts work to seconds-of-service via a measured
  token-clock rate, so the effective gamma and the gamma-Ada fixed point are
  cache-state-invariant; `tokens` is the paper-exact counted-token clock
  (the pre-fix behavior the RTX campaign ran — the RTX tables are
  tokens-clock numbers).
- `BOOST_GAMMA` (default: unset -> the clock-mode default, `1e-4` per-token
  under `tokens`, `0.1` 1/s under `wall`): static gamma, or gamma0 when
  `BOOST_ADAPTIVE_GAMMA=1`.
- `BOOST_BIN_SIZE` (default 256), `BOOST_ADAPTIVE_GAMMA` (default 0).
- `IN_BATCH_PREFIX_CACHING_MAX_PIN_S` (default 60, every cache-aware
  policy): wall-second bound on the in-batch dedup follower pin; `0`
  disables (the pre-fix unbounded hold).

Details: `docs/agentx_experiment.md`.
