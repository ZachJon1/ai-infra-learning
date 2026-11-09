# Benchmark analysis

## Concurrency sweep

Primary source: terminal output from the Day 55 tests, transcribed into `../results/concurrency_summary.csv`.

| Concurrent clients | N requests | RPS | Output tok/s | Mean E2E ms | p95 E2E ms | p99 E2E ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 1.284 | 70.602 | 778.886 | 814.863 | 823.915 |
| 2 | 40 | 2.913 | 160.203 | 686.516 | 693.865 | 694.465 |
| 4 | 160 | 6.069 | 333.794 | 658.863 | 684.592 | 685.983 |
| 8 | 160 | 10.864 | 597.542 | 735.580 | 760.779 | 774.609 |
| 16 | 160 | 19.149 | 1053.179 | 833.472 | 860.079 | 860.618 |

- Approximate throughput scaling concurrency 1 -> 16: **14.91x**, but this was not matched-duration or matched-sample-size benchmarking.
- Lowest observed mean and p95 latency at concurrency 4. At 16, p95 latency is about 25.6% higher than at 4, while RPS is 3.16x.
- 560/560 completed; this does not prove a sustainable reliability SLO or identify saturation. Additional 32/64-concurrency tests would need careful thermal/power supervision.
- The workload used a fixed 38-token prompt, 55 generated tokens per request, and repeated identical content. Warm cache behavior likely matters.
- Percentile values are exploratory, especially 40-request p99.

## Cache comparison

Each ON/OFF run used concurrency 4, 32 requests, `max_tokens=64` and three warmups. Results are copied from user-generated JSON files, without modifications.

| Workload | Cache | RPS | Output tok/s | p95 ms | Hit tokens / query tokens |
|---|---|---:|---:|---:|---|
| Shared | ON | 5.454 | 349.039 | 761.477 | 10800 / 11736 |
| Shared | OFF | 4.486 | 287.081 | 906.434 | Metrics unavailable/N/A |
| Unique | ON | 3.108 | 197.043 | 1475.619 | 4352 / 43624 |
| Unique | OFF | 2.933 | 185.941 | 1513.797 | Metrics unavailable/N/A |

Cache ON compared to OFF within each workload:

| Metric | Shared prefix | Unique context |
|---|---:|---:|
| RPS improvement | +21.6% | +6.0% |
| p95 latency reduction | 16.0% | 2.5% |
| Token-level prefix cache hit ratio ON | 92.02% | 9.98% |

**Important qualification:** The shared workload and varied-context workload also differed in total query tokens, structure and input-token distributions. Even with caching OFF, shared RPS exceeded varied-context RPS. Do **not** attribute the entire shared/unique gap to caching. Each ON/OFF pairing used one run only, at differing times, and therefore has environmental confounds. Repeat and randomize order before claiming general gains.

## Service measurement definitions

- `requests_per_s = completed-successful requests / measured wall seconds`.
- `output_tokens_per_s = sum of successful completion tokens / measured wall seconds`.
- E2E latency: elapsed client time until non-streaming response is fully received. Does not provide TTFT/TPOT.
- Error rate must be reported with success-only latency percentiles; errors are excluded from measured latency distributions.
- All reported numerical measurements apply to the precise test setup, not a deployed production system.

## Suggested next experiment

Repeat each cache ON/OFF pairing at least 3–5 times with alternating order and cooldown, record matched input-token distributions, validate engine mode from startup logs, and compute means/dispersion across trials. Add streaming TTFT/TPOT and arrival-rate-driven benchmarking later.
