Round-1 pre-recompute fixture: the exact target rendering from the follow-up
task (Task 7.3), verified once against `analysis/print_results_table.py` BEFORE
the warmup-exclusion fix recomputes e2_metrics.json (the fix shifts the
eval-log-derived columns: hit, cold/warm n, cold/warm starv, gap99, starv) and
before round-2 runs land in results/agentx. Known tolerated diffs:
- fcfs c16 starv `.549` (task typo; generator emits uniform `0.549`)
- footnote (c) precision `+92.1%` vs task prose `+92%`
- footnote (b) corrected per Task 1a's verdict (fcfs has real admission-time
  cache reuse; the 0.000 hit is sort-time bookkeeping, not absent cache)

| Policy | c | γ | n | hit | q_p95 (s) | q_p99 (s) | starv | cold n | cold starv | warm n | warm starv | gap99 (s) | RPM | oTPM | TTFT (ms) | TPOT p95 (ms) | retr | q_p99 Δ vs LPM | Anti-starv. vs LPM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lpm (baseline)** | 4 | — | 217 | 0.883 | 0.005 | 2.830 | 0.000 | 10 | 0.000 | 210 | 0.000 | +6.385 | 7.1 | 5371 | 1851 | 28.4 | 0 | — | — |
| boost | 4 | 1e-4 | 223 | 0.888 | 0.005 | 0.009 | 0.000 | 10 | 0.000 | 216 | 0.000 | −0.004 | 7.3 | 5922 | 1798 | 27.5 | 0 | −99.7% | 0% ¹ |
| fcfs ‡ | 4 | — | 214 | 0.000 | 0.005 | 2.850 | 0.000 | 218 | 0.000 | 0 | — | — | 7.0 | 5165 | 1736 | 26.8 | 0 | +0.7% | 0% ¹ |
| **lpm (baseline)** | 8 | — | 141 | 0.894 | 0.005 | 2.590 | 0.000 | 16 | 0.000 | 131 | 0.000 | −2.041 | 4.6 | 5839 | 1987 | 24.1 | 0 | — | — |
| boost | 8 | 5e-5 | 143 | 0.894 | 0.005 | 3.570 | 0.007 | 16 | 0.000 | 133 | 0.008 | −3.090 | 4.7 | 5865 | 2230 | 24.3 | 0 | +37.8% | −0.7 pp ¹ |
| boost | 8 | 1e-4 | 140 | 0.894 | 0.005 | 3.600 | 0.000 | 16 | 0.000 | 130 | 0.000 | −3.093 | 4.6 | 5836 | 2170 | 22.2 | 0 | +39.0% | 0% ¹ |
| boost | 8 | 2e-4 | 139 | 0.893 | 0.005 | 3.600 | 0.000 | 16 | 0.000 | 130 | 0.000 | −3.091 | 4.5 | 5678 | 2278 | 21.5 | 0 | +39.0% | 0% ¹ |
| fcfs ‡ | 8 | — | 139 | 0.000 | 0.005 | 2.805 | 0.000 | 145 | 0.000 | 0 | — | — | 4.6 | 5826 | 2099 | 24.3 | 0 | +8.3% | 0% ¹ |
| **lpm (baseline)** | 16 | — | 201 | 0.803 | 73.929 | 94.750 | 0.380 | 34 | 0.265 | 185 | 0.378 | −14.247 | 6.6 | 5622 | 21468 | 98.6 | 6 | — | — |
| boost | 16 | 1e-4 | 161 | 0.753 | 118.000 | 183.600 | 0.512 | 33 | 0.273 | 145 | 0.524 | −21.509 | 5.2 | 4868 | 26949 | 115.7 | 0 | +93.8% | −34.7% |
| fcfs ‡ | 16 | — | 151 | 0.000 | 97.833 | 178.143 | .549 | 167 | 0.509 | 0 | — | — | 4.9 | 4697 | 32312 | 166.8 | 5 | +88.0% | −44.5% |
