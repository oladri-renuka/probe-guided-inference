# Probe-Guided Routing: Benchmark Results

| Strategy | Throughput (req/s) | Avg tokens/req | Accuracy (completed) | Convergence rate | p50 latency (s) | p95 latency (s) |
|---|---|---|---|---|---|---|
| baseline | 0.005 | 6939 | 59.0% | 61.0% | 17249.9 | 35225.4 |
| probe_terminate | 0.008 | 4657 | 66.0% | 46.0% | 10937.2 | 23385.4 |
| probe_deprioritize | 0.005 | 6987 | 59.5% | 59.5% | 16385.8 | 35077.4 |

## Scheduler Activity

| Strategy | Admitted | Finished | Probe-terminated | Preemptions |
|---|---|---|---|---|
| baseline | 200 | 200 | 0 | 0 |
| probe_terminate | 200 | 200 | 59 | 0 |
| probe_deprioritize | 275 | 200 | 0 | 75 |

## probe_terminate: False Termination Rate

Of the requests probe_terminate cut short at token 150, this is the fraction that baseline's untouched run of the same problem shows WOULD have converged (see benchmark/report.py `false_termination_rate` docstring for the ground-truth methodology).

- Terminated: 59
- False terminations (would have converged): 28
- False termination rate: 47.5%

## probe_deprioritize: Convergent vs Divergent Latency

| Group | n | p50 latency (s) | mean latency (s) |
|---|---|---|---|
| Convergent (predicted) | 126 | 10112.6 | 10224.3 |
| Divergent (predicted) | 74 | 28482.7 | 29072.7 |

Baseline p50 latency for the SAME (predicted-convergent) requests: 18323.9s. probe_deprioritize change: +44.8%.