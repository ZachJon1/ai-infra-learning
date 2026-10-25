# Day 41 — Dynamic Batching for GPU Inference

**Track:** AI Infrastructure & ML Systems — 220-Day Roadmap  
**Phase:** 3 — Model Serving / Inference Infrastructure  
**Environment:** Python, PyTorch, NVIDIA CUDA (with CPU fallback)  
**Status:** Lab completed; scheduler bug reproduced and fixed; burst and paced benchmarks collected

## Project overview

This lab builds a small **dynamic inference batcher**: independent requests enter a queue, a background scheduler assembles compatible requests into batches, and a PyTorch model executes those batches on the GPU. Each request receives its own result through a `concurrent.futures.Future`.

The goal is to learn the scheduling and performance tradeoffs underlying real inference systems, not to present this educational prototype as a production server.

### Learning objectives

- Distinguish preassembled/static batching, dynamic batching, and LLM continuous batching.
- Implement `max_batch_size` and `max_wait_ms` dispatch controls.
- Preserve the mapping between an independently submitted request and its response.
- Investigate and fix a queue-draining bug caused by expired batching deadlines.
- Measure throughput, batch size, p50/p95 request latency, and p95 queue wait.
- Explain why throughput gains and latency costs depend on traffic patterns and serving capacity.

## Files

```text
day41-dynamic-batching/
├── dynamic_batcher.py   # Queue, background batching worker, PyTorch inference, and metrics
├── benchmark.py         # Burst and paced arrival-pattern experiments
├── README.md            # Project overview, reproduction steps, and measured findings
└── STUDENT_NOTES.md     # Explanations, worked examples, troubleshooting, and review
```

The Python scripts are the **complete updated versions from the Day 41 lab**. The Markdown documents are intended to live beside those scripts; they do not replace them.

## Architecture

```text
      Independent inference requests
             A   B   C   D ...
              \  |  |  /
               \ |  | /
                v v  v
        +-------------------+
        | Thread-safe queue |
        +-------------------+
                  |
                  v
        +--------------------+
        | Single batch worker|
        | - take first item  |
        | - drain ready items|
        | - wait if needed   |
        | - stop at size/time|
        +--------------------+
                  |
                  v
        +--------------------+
        | torch.stack(inputs)|
        | PyTorch model      |
        | CUDA synchronize   |
        +--------------------+
                  |
                  v
       Individual Future results
```

**Dispatch behavior:** Collect requests already waiting, up to the batch-size limit. If the queue becomes empty before the batch is full, wait only until the oldest request's batching deadline. A full batch may be dispatched early; an incomplete batch may be dispatched at its deadline.

> `max_wait_ms` limits **additional batch-formation waiting when a worker is available**. It does **not** bound time spent in backlog behind earlier batches, total end-to-end request latency, or overload-induced queue growth.

## Reproduce locally

From the existing AI-infrastructure virtual environment:

```bash
cd ~/ai-infra/day41-dynamic-batching

# Confirm PyTorch and whether CUDA is accessible
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Functional test: independently submitted requests, grouped in batches
python dynamic_batcher.py

# Performance comparisons: burst and paced traffic
python benchmark.py
```

**Prerequisites:** Python environment with `torch` installed and the two Python scripts in this folder. The functional code supports CPU fallback; figures documented below are from a CUDA-enabled run. No additional service, Docker container, or HTTP endpoint is required for this lab.

### Expected functional behavior

The functional test submits 32 requests (`torch.randn(128)`) through a model with layers `Linear(128, 256)`, `ReLU`, and `Linear(256, 10)`. With `max_batch_size=8` and `max_wait_ms=10`, rapid submissions can form four batches of eight, as observed below:

```text
Batch size= 8 | Inference=42.479 ms
Batch size= 8 | Inference=0.139 ms
Batch size= 8 | Inference=0.109 ms
Batch size= 8 | Inference=0.097 ms
Completed requests: 32
Output shape: torch.Size([10])
```

This is **one recorded run**, not a deterministic timing expectation. The first batch was much slower than later ones because first-use and initialization overhead had not been excluded in the functional test.

## Debugging story: expired deadline caused singleton batches

### Before the fix

The original batching loop computed a deadline from the first request's arrival time, then immediately stopped gathering requests if that deadline had already passed. After the first long GPU execution, many pending requests were already older than the batching deadline, so the worker dispatched them one at a time rather than draining the populated queue.

- **Observed:** 32 completed requests in **25 batches** — one batch of eight and 24 singleton batches.
- **Mean batch size:** 32 / 25 = **1.28**.

### Fix

The corrected worker first uses `queue.get_nowait()` to collect items **already waiting**. Only when the queue is empty does it consider waiting with `queue.get(timeout=remaining)` or dispatching after the deadline.

- **Observed in three reruns:** 32 completed requests in **four batches of eight** each.
- **Mean batch size:** 32 / 4 = **8.00**.
- **Effect:** The scheduler avoided 21 unnecessary batch executions in this 32-request test. This is a **batch-count reduction**, not a measured 84% throughput gain.

**Generalizable lesson:** An expired batch-formation deadline should prevent *additional waiting*, not prevent the scheduler from taking compatible requests that are already queued.

## Benchmark methodology

`benchmark.py` warms up both batch-size-1 and batch-size-8 execution before testing three policies:

| Policy | `max_batch_size` | `max_wait_ms` |
|---|---:|---:|
| A — No batching | 1 | 0 |
| B — Dynamic / short wait | 8 | 5 |
| C — Dynamic / longer wait | 8 | 20 |

Two traffic patterns are tested:

1. **Burst:** submit 512 requests as rapidly as the Python loop can enqueue them. This reveals how quickly the implementation drains a backlog.
2. **Paced:** submit 128 requests, targeting a 2 ms interval (**500 requests/s offered load**). This examines an arrival-limited service.

Definitions for reported metrics:

- **Achieved throughput:** completed requests / total wall-clock elapsed time, including request submission and completion.
- **Mean batch size:** completed requests / number of executed batches.
- **Request latency:** time from `submit()` timestamp to result preparation and `Future` completion. This is **not HTTP end-to-end latency**.
- **Queue wait:** time from `submit()` timestamp to batch dispatch. It includes time waiting behind earlier batches **and** time spent assembling the current batch.
- **p50/p95:** nearest-rank request-level percentiles recorded by the benchmark.

### Recorded measurements — burst traffic (512 requests)

| Metric | No batching | Dynamic 5 ms | Dynamic 20 ms |
|---|---:|---:|---:|
| Batches | 512 | 64 | 64 |
| Average batch size | 1.00 | 8.00 | 8.00 |
| Throughput (req/s) | 6,229.6 | **23,682.2** | 23,386.3 |
| p50 latency (ms) | 41.369 | 11.009 | 10.569 |
| p95 latency (ms) | 77.095 | **19.519** | 19.986 |
| p95 queue wait (ms) | 76.953 | 19.189 | 19.718 |

For this single burst run, dynamic batching at 5 ms showed **3.80× observed throughput** versus no batching, and approximately **74.7% lower p95 request latency**. These are comparative results from this educational workload, not production performance guarantees.

Because all dynamic batches filled to eight, the 5 ms and 20 ms settings behaved similarly under burst load; the timeout rarely constrained batch formation.

### Recorded measurements — paced traffic (500 offered requests/s; 128 requests)

| Metric | No batching | Dynamic 5 ms | Dynamic 20 ms |
|---|---:|---:|---:|
| Batches | 128 | 43 | 16 |
| Average batch size | 1.00 | 2.98 | 8.00 |
| Throughput (req/s) | 500.8 | 496.5 | 502.5 |
| p50 latency (ms) | **1.077** | 4.315 | 7.874 |
| p95 latency (ms) | **1.395** | 6.402 | 15.204 |
| p95 queue wait (ms) | 0.200 | 5.219 | 14.102 |

Paced throughput remained approximately 500 req/s in all three cases because the service was **offered** only about 500 req/s. Larger batches did not produce additional completed requests; they increased request waiting time. In this experiment, a 20 ms batching window created full batches but substantially higher p95 latency.

### Example policy decision (hypothetical SLO)

If the service requires **p95 request latency < 10 ms at 500 req/s**, the measured paced results would place configurations A and B within the threshold, while C would exceed it. Configuration B (5 ms) could be a candidate if burst handling or GPU efficiency is also important. A production decision would need repeated measurements, actual model traffic, tail latency, and cost/utilization evidence.

## Operational lessons

- A configured batching timeout is **not a queueing or response-time guarantee**. Under burst traffic, p95 queue time exceeded 5 ms because requests waited for the single worker to finish earlier batches.
- Queue length rising over time alongside full batches and high GPU utilization suggests **arrival demand may exceed serving capacity**. Confirm by comparing arrival rate, completion rate, and GPU execution behavior.
- Raising queue capacity postpones overload rejection but does **not** fix a sustained capacity deficit. Consider scaling replicas, improving processing capacity, bounded queues, rate limits, and admission control.
- Under low arrival rates, an aggressive batching window may add latency without improving achieved throughput.
- For LLMs, **continuous batching** extends the idea by scheduling at decoding iterations so completed sequences can leave and others can join without waiting for all generations to finish.

## Limitations and next steps

**Prototype limitations**

- Uses a small synthetic MLP with fixed-shape vectors; not the Day 40 CNN or real LLM workloads.
- One background worker; no production scheduling fairness, deadline enforcement, bounded queue, cancellation, or graceful shutdown.
- No FastAPI/HTTP, networking, authentication, Kubernetes placement, or multi-replica load balancing.
- One recorded benchmark run per configuration; no confidence intervals, p99, sustained-load capacity testing, or cost/GPU-utilization correlation.
- Timing is application-level wall-clock timing with CUDA synchronization; GPU kernel timings and transfer overhead are not separately profiled.

**Planned extensions**

1. Connect batching to the Day 40 model and the existing inference service.
2. Add queue depth, arrival rate, completed requests, rejection rate, and p99 latency metrics.
3. Run repeated trials at multiple offered loads and input sizes, then profile GPU use.
4. Introduce bounded queues, timeout/cancellation semantics, and controlled worker shutdown.
5. Compare this educational implementation with NVIDIA Triton's production batching controls later in the roadmap.

## Portfolio / interview talking point

> Implemented a PyTorch-based dynamic inference batcher using a thread-safe queue, a background scheduler, and per-request Futures. Diagnosed a deadline-handling bug that caused 24 unnecessary singleton batches, fixed backlog draining, and benchmarked three policies under burst and paced request arrivals. In a 512-request burst test, the 5 ms configuration measured 3.80× the throughput of unbatched execution, while paced traffic highlighted the p95 latency cost of longer aggregation windows.

