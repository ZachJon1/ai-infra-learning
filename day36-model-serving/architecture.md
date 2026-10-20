# Architecture Specification — Production-Style ML Inference Platform

## Goal and scope

Build one evolving inference service that can be deployed, profiled, monitored, and debugged. It will reuse concepts and artifacts from the earlier Docker, Kubernetes, and GPU scheduling labs. Day 36 is a **design milestone**, not a production deployment.

## Initial stack

| Layer | Initial selection | Later extension |
| --- | --- | --- |
| Model | Small pretrained PyTorch classifier (candidate: MobileNetV3 Small) | Alternative runtime/model |
| API | FastAPI | Request limits, detailed telemetry |
| Execution | CPU first | CUDA-backed execution |
| Packaging | Python virtual environment | Docker image |
| Orchestration | Local execution | Kubernetes + NVIDIA Device Plugin |
| Configuration | Environment/application settings | Helm values |
| Monitoring | Stage timers and logs | Prometheus, Grafana, GPU telemetry |
| Advanced serving | One worker, single-request baseline | Batching, Triton or vLLM where appropriate |

Model selection is **proposed**, not finalized or benchmarked. Pretrained weights may require download and local caching; do not check large weights or secrets into Git without deliberate review of storage and licensing implications.

## Request architecture

```text
Client → [optional ingress/load balancer] → HTTP API / FastAPI
                                           |
                                    request validation
                                           |
                                     preprocessing
                                           |
                                    PyTorch model
                                           |
                                    CPU or CUDA GPU
                                           |
                                    postprocessing
                                           |
                                      HTTP response

                 Per-stage timings / request metrics / logs
```

The ingress and Kubernetes parts are conceptual until those increments are built. API and model runtime may share a process in the first implementation.

## API contract (proposed)

| Endpoint | Responsibility | Readiness semantics |
| --- | --- | --- |
| `POST /predict` | Validate input; preprocess; run inference; postprocess | Only available when model/runtime is ready |
| `GET /health/live` | Detect an irrecoverably unhealthy process | Avoid checking external dependencies or transient latency |
| `GET /health/ready` | Indicate ability to accept predictions | False during model load, shutdown, or inability to serve |
| `GET /metrics` | Request/error counters, timings; future telemetry | To be implemented incrementally |

Example **illustrative** response:

```json
{
  "prediction": "tabby_cat",
  "confidence": 0.96,
  "model_version": "v1"
}
```

## Worker lifecycle

1. Start process and initialize its framework/runtime.
2. Load verified model weights once per worker.
3. Allocate model storage and, when enabled, move the model to the GPU.
4. Optionally run representative warmup requests.
5. Mark the API ready when actual serving is possible.
6. Serve requests and collect per-stage metrics.
7. During shutdown, become unready, stop accepting new work, and drain in-flight requests within a configured grace period.

**Startup probe:** allow realistic cold-start time; ensure probe behavior matches the chosen application endpoint.  
**Readiness probe:** exclude workers that cannot serve from normal Kubernetes Service routing.  
**Liveness probe:** detect conditions where restart is an appropriate recovery mechanism; avoid cascading restarts during temporary overload.

## Performance evaluation plan

Measure on the actual hardware and document the environment. Do not invent benchmark numbers.

| Measurement | Why |
| --- | --- |
| Model load / time-to-ready | Startup reliability and deployment planning |
| End-to-end p50/p95/p99 | User-facing performance and tail latency |
| Queue waiting time/depth | Detect overload and bottlenecks |
| Preprocessing / inference / postprocessing times | Isolate slow stages |
| Completed requests per second | Realized throughput |
| Error / timeout rate | Reliability under load |
| CPU and GPU utilization | Resource bottleneck clues |
| GPU VRAM peak and allocation | Model footprint and OOM diagnosis |

Benchmark approach (when implementation exists): establish a single-request baseline; allow warmup; run fixed-duration load at documented concurrency levels; report latency distributions, achieved throughput, error rates, and hardware utilization together. Include model version, input size, batch size, framework/runtime versions, CPU/GPU hardware, and methodology.

## Resource and rollout decisions

- A single Kubernetes node exposing one allocatable `nvidia.com/gpu` typically cannot schedule two pods that each request `nvidia.com/gpu: 1` without intentionally configured sharing/partitioning.
- Successful GPU scheduling does **not** guarantee enough VRAM for weights, activation tensors, and runtime memory.
- On a single GPU, `maxSurge: 0` with a rollout that permits unavailability can unblock replacement, **but may create downtime**; extra GPU capacity is generally needed for straightforward zero-downtime replacement.
- More GPU workers do not automatically raise throughput when compute or VRAM capacity is already exhausted.
- CPU fallback is a possible availability option only if measured performance meets acceptable reduced-service requirements.
- Prefer bounded queues/backpressure, load shedding, smarter batching, and added compute capacity as dictated by measurements.

## Milestones

| Days | Increment | Evidence |
| --- | --- | --- |
| 36 | Architecture and service requirements | These Markdown documents |
| 37 | FastAPI + PyTorch CPU inference | Functional local endpoint/tests |
| 38 | Dockerized inference | Dockerfile, reproducible run steps |
| 39–45 | GPU-backed inference and first tuning | Real device/memory/latency measurements |
| 46–55 | Advanced runtime, concurrency, load test | Comparative benchmarks and capstone walkthrough |

## Open decisions

- Confirm the actual pretrained model and permitted distribution of its weights.
- Confirm GPU model and available VRAM on the deployment host.
- Choose input size, target workload, and SLOs **after baseline measurement**.
- Decide whether to implement per-request queue timings in the API or load generator first.
