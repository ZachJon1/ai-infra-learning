# Day 42 — Latency vs. Throughput in AI Inference Systems

**Track:** AI Infrastructure & ML Systems — 220-Day Roadmap  
**Phase:** 3 — Model Serving / Inference Infrastructure  
**Environment:** Local Dockerized FastAPI/PyTorch inference service  
**Date:** October 25, 2026  
**Status:** Completed initial CPU/GPU concurrency benchmarks; further validation planned

## Objective

Investigate how client concurrency affects **throughput** and **end-to-end request latency** for a small inference service. Compare CPU-backed and GPU-backed endpoints, examine p50/p95/p99 latency, evaluate a *hypothetical* p99 service-level objective (SLO), and identify where further instrumentation is necessary.

This exercise builds on earlier work involving Dockerized inference (Day 38), GPU-backed inference (Day 39), batching (Day 40), and dynamic batching (Day 41).

## Service under test

Two existing Docker containers exposed the same demonstration prediction API:

| Container | Local address | Purpose |
| --- | --- | --- |
| `inference-gpu` | `http://127.0.0.1:8002` | GPU-configured inference endpoint |
| `inference-cpu` | `http://127.0.0.1:8003` | CPU-configured inference endpoint |

The GPU-configured endpoint passed its health check:

```bash
curl -s http://127.0.0.1:8002/health
# {"status":"healthy"}
```

Example inference request:

```bash
curl -s -X POST http://127.0.0.1:8002/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[1.0,2.0,3.0]}'
# {"prediction_value":2.4,"model_version":"demo-linear-v1"}
```

**Scope note:** The endpoint is named/configured as GPU-backed, but the HTTP benchmark alone does not verify per-request CUDA execution or measure GPU kernel time. Hardware utilization and device execution require separate instrumentation.

## Methodology

The `benchmark.py` client uses Python's standard library (`urllib.request` and `ThreadPoolExecutor`) to send HTTP POST requests and record request/response elapsed time.

| Parameter | Value |
| --- | --- |
| Concurrent client workers | `1, 4, 8, 16, 32` |
| Baseline workload | 300 measured requests **per concurrency level**, per endpoint |
| Repeated GPU workload | 3 runs × 3,000 measured requests **per concurrency level** |
| Warmup | 20 sequential requests before each measured concurrency level |
| Input | `{"features":[1.0,2.0,3.0]}` |
| Success criterion | Valid HTTP response containing `prediction_value` |
| Timeout | 10 seconds per HTTP request |
| Throughput | Successful responses divided by total elapsed benchmark time (RPS) |
| Latency | Client-side wall-clock time per successful HTTP request (milliseconds) |
| Percentile calculation | Nearest rank over successful response latencies |

All concurrency levels were tested **in ascending order**; the same client machine hosted the inference containers. This is a **closed-loop, fixed-concurrency** benchmark with a fixed number of requests, *not* a constant-arrival-rate load test. Latency measures the HTTP request-response interval and does not separately capture time spent waiting for a client worker, batching queue time, preprocessing, GPU execution, or postprocessing. Throughput measurements include client orchestration overhead.

### How to reproduce

From the Day 42 directory, using the existing `benchmark.py` and running containers:

```bash
cd ~/ai-infra/day42-latency-throughput

docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# Initial comparison (300 requests per concurrency level)
python benchmark.py \
  --url http://127.0.0.1:8002/predict \
  --requests 300 \
  --output gpu_baseline.csv

python benchmark.py \
  --url http://127.0.0.1:8003/predict \
  --requests 300 \
  --output cpu_baseline.csv

# Repeat the GPU benchmark (3,000 requests per concurrency level)
for run in 1 2 3; do
  python benchmark.py \
    --url http://127.0.0.1:8002/predict \
    --requests 3000 \
    --output "gpu_repeat_${run}.csv"
done
```

`--requests` sets requests **for each concurrency setting**, not the total across all settings.

## Results

Results below are measurements from this local experiment; they are not general performance claims for CPUs, GPUs, FastAPI, or PyTorch.

### Initial GPU vs. CPU baseline — 300 requests per concurrency

| Concurrency | GPU throughput (RPS) | GPU p50 (ms) | GPU p95 (ms) | GPU p99 (ms) | CPU throughput (RPS) | CPU p50 (ms) | CPU p95 (ms) | CPU p99 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 824.31 | 1.10 | 1.66 | 1.91 | 957.68 | 0.95 | 1.44 | 1.95 |
| 4 | 1,541.25 | 2.49 | 3.14 | 3.91 | 1,755.14 | 2.22 | 2.80 | 3.04 |
| 8 | 1,420.30 | 5.26 | 7.19 | 13.25 | 1,677.60 | 4.62 | 5.28 | 5.87 |
| 16 | 1,411.80 | 10.39 | 15.24 | 16.31 | 1,723.57 | 9.09 | 10.09 | 10.72 |
| 32 | 1,467.18 | 21.03 | 23.13 | 24.26 | 1,599.44 | 19.23 | 19.99 | 20.52 |

**Errors:** 0 for every configuration in these initial tests.

At concurrency 4, the CPU endpoint completed approximately **1,755 RPS**, versus approximately **1,541 RPS** for the GPU-configured endpoint. The toy model is extremely small, so fixed GPU/framework overhead may offset any acceleration benefits. This observation does **not** imply that CPU inference is generally faster.

### Repeated GPU tests — 3,000 requests per concurrency, three runs

**Throughput (requests/second):**

| Concurrency | Run 1 | Run 2 | Run 3 |
| ---: | ---: | ---: | ---: |
| 1 | 1,115.84 | 1,089.12 | 306.58 |
| 4 | **1,605.27** | **1,544.11** | 741.58 |
| 8 | 1,592.93 | 1,528.32 | **985.37** |
| 16 | 1,530.53 | 1,280.78 | 892.43 |
| 32 | 1,518.80 | 1,163.23 | 898.94 |

**p99 latency (milliseconds):**

| Concurrency | Run 1 | Run 2 | Run 3 |
| ---: | ---: | ---: | ---: |
| 1 | 1.57 | 1.29 | 7.05 |
| 4 | **3.46** | **3.56** | **8.51** |
| 8 | 6.52 | 6.69 | 12.57 |
| 16 | 16.04 | 19.68 | 28.78 |
| 32 | 59.28 | 43.46 | 49.66 |

**p50 and p95 latency (milliseconds):**

| Concurrency | Run 1 p50 / p95 | Run 2 p50 / p95 | Run 3 p50 / p95 |
| ---: | ---: | ---: | ---: |
| 1 | 0.82 / 1.20 | 0.86 / 1.04 | 3.02 / 5.13 |
| 4 | 2.42 / 3.09 | 2.51 / 3.20 | 5.24 / 7.23 |
| 8 | 4.92 / 5.76 | 5.12 / 6.04 | 7.91 / 10.85 |
| 16 | 10.07 / 12.99 | 12.23 / 15.09 | 17.39 / 22.99 |
| 32 | 20.05 / 24.12 | 26.22 / 36.63 | 34.86 / 42.53 |

**Errors:** 0 in all repeated tests: **45,000 successful measured GPU requests** (3 runs × 5 concurrency settings × 3,000 requests).

## Analysis

### 1. Higher concurrency does not guarantee higher throughput

In the first two repeated runs, throughput peaked at concurrency **4**, around **1,544–1,605 RPS**. More concurrent requests did not meaningfully improve completed requests per second. At concurrency 32, p99 latency was much higher, reaching **43–59 ms** across runs.

The observed pattern is consistent with a bottleneck somewhere in the system, but **it does not establish GPU saturation**. Possible limiting stages include client-side Python/thread scheduling, HTTP handling, server execution, CPU-side work, and GPU-related overhead.

### 2. Run-to-run variability is significant

Run 3 was slower even at concurrency 1 (**306.58 RPS**, versus **1,115.84 RPS** in Run 1). This variation occurred without changing the requested concurrency level and therefore cannot be explained solely by higher concurrency. The cause remains **unverified**. Potential contributors include system load, CPU contention, power/thermal state, client overhead, and other shared-host effects.

### 3. Latency SLO tradeoff

For learning purposes, use a **hypothetical** objective of **p99 ≤ 10 ms** (not a deployed production SLO).

- **Concurrency 4:** p99 of 3.46, 3.56, and 8.51 ms; passes in all three repeated runs.
- **Concurrency 8:** p99 of 6.52, 6.69, and 12.57 ms; fails in Run 3.
- **Concurrency 16 or 32:** fails in all three repeated runs.

**Provisional choice:** Concurrency **4** was the most consistent initial candidate under the illustrative SLO. Do not treat this result as a production-ready concurrency cap until testing is longer, representative, and instrumented.

### 4. Small inference models may not benefit from GPU acceleration

The GPU-configured endpoint was slower than the CPU endpoint in the initial benchmark. A tiny linear model provides too little parallel work to assume GPU acceleration will dominate request-handling overhead. Larger tensor workloads, optimized batching, and direct measurements of actual GPU work are necessary before drawing performance conclusions.

## Limitations and threats to validity

- The benchmark client and server ran on the **same host**, creating possible resource contention.
- Only **one CPU baseline run** was recorded; the GPU endpoint was repeated three times.
- Tests ran concurrency levels in the same **ascending order**, which can confound results with time-varying machine conditions.
- Each concurrency configuration processed a **fixed request count**, so wall-clock test duration varied with throughput.
- Traffic was generated at fixed client concurrency, not at a controlled request-arrival rate; results cannot establish maximum **sustainable open-loop arrival rate**.
- p99 values are estimates from short local runs, not evidence of sustained SLO compliance.
- The benchmark records **end-to-end client-observed HTTP latency**, not server queue time or kernel execution latency.
- Metrics for GPU utilization, GPU memory, CPU utilization, power/thermals, and internal batching delays were **not collected alongside these runs**.
- The demo model is not representative of large transformer inference or compute-heavy GPU workloads.
- Request failures would count toward error rate, but the script's latency percentiles use **successful responses only**.

## Next experiments

1. **Collect utilization while benchmarking:** sample `docker stats`, CPU usage, and `nvidia-smi` throughout the run rather than afterward.
2. **Separate the client from the server:** generate load from another host or dedicated container with reserved resources to test whether the benchmark driver is limiting throughput.
3. **Increase workload duration and randomize test order:** use longer intervals, multiple runs, and confidence/variation summaries.
4. **Measure pipeline stages:** add server timing for request queueing, preprocessing, model execution, and postprocessing.
5. **Test traffic patterns:** compare constant-arrival-rate load, bursts, and sustained traffic rather than fixed concurrency alone.
6. **Test a larger GPU-relevant model and meaningful batch sizes:** determine when parallel GPU compute overtakes fixed per-request overhead.
7. **Compare additional concurrency settings near the knee:** for example, 2, 4, 6, and 8, with an explicit latency target and failure-rate constraint.

### Useful diagnostic commands

```bash
# Container CPU and memory snapshot
docker stats --no-stream inference-gpu inference-cpu

# GPU device snapshot
nvidia-smi

# Host load
uptime

# Most CPU-intensive processes
ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -15
```

These commands provide snapshots, not time-aligned proof of a bottleneck. Collect them **during** a benchmark to support a diagnosis.

## Repository artifacts

Expected contents of the existing `day42-latency-throughput/` directory:

```text
day42-latency-throughput/
├── README.md
├── benchmark.py
├── gpu_baseline.csv
├── cpu_baseline.csv
├── gpu_repeat_1.csv
├── gpu_repeat_2.csv
└── gpu_repeat_3.csv
```

The CSV files listed above were produced in the local lab; they are **not embedded** in this README. Keep their original values rather than substituting fabricated examples.

## Key takeaways

- **Latency** is per-request completion time; **throughput** is completed work per second.
- **p50, p95, and p99** reveal distribution shape and tail behavior that averages conceal.
- **Concurrency and batching** can improve efficiency, but additional in-flight requests can increase latency without increasing throughput once a bottleneck is reached.
- **Little's Law:** `average requests in system = average throughput × average time in system`, for a stable system with consistent time units.
- Optimize for **latency and reliability objectives at required throughput**, not simply maximum GPU utilization.
- Repeated benchmarking matters: the unexplained variation in Run 3 is an investigation target, not something to ignore or explain away.

---

**Next lesson:** Day 43 — Model Warmup and Cold Starts.
