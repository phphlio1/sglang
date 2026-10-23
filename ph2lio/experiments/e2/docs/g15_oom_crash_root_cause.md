# Root cause: g15.0 H200 crash chain (hard exhaustion -> OOM spin -> CUDA IMA)

Cell c64_g15.0_wall_k256_20261020_202439 (GLM-5.3 DPA8, c64 seed 42, gamma 15
= near-FCFS). Engine log on the server:
results/glm53/agentx_dynamo/logs/engine_20261020_202439.log. Cell invalid:
420 records served vs ~1210 in the valid cells, hit 77.5% vs 87-88%, client
ran the grace window alone (84.6% errors). Local tree line numbers below;
server a9d2b9ecd8 differs slightly (crash frames: scheduler.py:4895,
scheduler_components/batch_result_processor.py:928).

## Timeline (engine log)

- 21:02:28-33 steady: token usage 0.66-0.69, 2 running, 1 queued. DPA8
  interleaves 8 per-rank schedulers in one log -- usage lines from different
  ranks alternate, they are not one trajectory.
- 21:02:34.4 usage 0.98: the giant chain lands (near-FCFS admits giants in
  arrival order).
- 21:02:35.8 a prefill completes with reported usage 1.00 on the filling
  rank.
- 21:02:36.05-36.31 five alloc_paged_token_slots_extend failures in 260 ms:
  "Try to allocate 4096/4096/4096/4096/3937 tokens. Available full tokens:
  128 -> 64 -> 64 -> 0 -> 0", full_evictable_size=0 throughout (every page
  locked by running reqs -- nothing to evict). Each handled by the
  Prefill-OOM backout (scheduler.py:4174-4198): free req pool rows, return
  reqs to the waiting queue, skip the step.
- 21:02:36.49 process_batch_result_decode -> result.copy_done.synchronize()
  raises torch.AcceleratorError: CUDA illegal memory access. SIGQUIT
  teardown by 21:02:43.

## Defects

1. The admission budget admits at hard exhaustion. The backout's own
   comment states it: "The admission budget said this batch fits but the KV
   pool disagreed." The try/except inside get_new_batch_prefill is the only
   guard, and it fires after the admission decision, inside
   prepare_for_extend.
2. The backout never pauses admission. _oom_recovery_grace_steps=10 only
   silences the invariant checker (scheduler.py:5065-5074); the next step
   (~50 ms) re-admits the same front-of-queue req -- hence 5 raises in
   260 ms, stopped only by the crash.
3. The >=3-consecutive-OOM chunk cap cannot succeed at exhaustion:
   min(chunk, max(avail, page_size)) floors the attempt at page_size even
   when avail=0 (scheduler.py:3946-3953) -- a guaranteed-doomed retry.
4. The IMA is async-reported (torch's own traceback says so); the
   copy_done.synchronize() frame is the observation point, not the cause.
   The backout itself is CPU-only (req_to_token_pool.free -> free_rows ->
   free_slots.extend, a Python list -- no GPU write), so "the backout frees
   memory under an in-flight kernel" is ruled out. The illegal access
   happened inside the burst window: either the extend-alloc path enqueued
   device work on the exhausted pool before raising, or the usage-1.00
   prefill at 21:02:35.8 already wrote out of range. Pinning it needs
   compute-sanitizer or CUDA_LAUNCH_BLOCKING on a repro cell.

## Fix proposal (pre-phase-2 reliability item)

fcfs-match at c64 has the same arrival-order giant dynamics, so this is a
phase-2 risk, not a g15 curiosity:

- A (root): admission-time guard -- if available_size() + evictable_size()
  < page_size, admit no prefill this step (early return, no exception path,
  no spin). Decodes drain the pool; admission resumes when available
  recovers.
- B: drop the max(avail, page_size) floor -- skip admission while
  avail < page_size (A subsumes this if placed in the budget).
- C (driver): a cell that still lacks e2_metrics.json after 2 attempts gets
  a SKIP marker so the resume loop cannot retry forever unattended.

Policy correlation: only the near-FCFS arm built the giant chain; g1.0-g10.0
ran clean at the same load (contamination scan in
boost_tail_collapse_root_cause.md, H200 phase-1 section).
