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
`results/agentx/<policy>/c<conc>_<ts>/` (never overwrites), with
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

Details: `docs/agentx_experiment.md`.
