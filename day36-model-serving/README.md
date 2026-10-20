# Day 36 — Production Model-Serving Fundamentals

**Track:** AI Infrastructure / ML Systems — 220-Day Roadmap  
**Phase:** 3 — Model Serving / Inference Infrastructure  
**Status:** Completed conceptual lesson and architecture design; implementation begins on Day 37.

## Learning outcomes

- Distinguished model training (weight optimization) from production inference (reliable, efficient predictions).
- Explained the request path: API → validation/preprocessing → inference runtime → CPU/GPU execution → postprocessing → response.
- Related request latency, queue depth, throughput, concurrency, tail latency, GPU utilization, and cost.
- Designed initialization, model loading, optional warmup, readiness, liveness, and graceful shutdown behavior.
- Evaluated GPU constraints, Kubernetes rollout limitations, and CUDA out-of-memory failures.

## Architecture

See [architecture.md](architecture.md) for the proposed stack, API contract, workload lifecycle, and performance-measurement plan.

**Proposed first implementation:** Python + PyTorch (small pretrained image classifier, e.g., MobileNetV3 Small) + FastAPI, initially CPU-backed. Then Docker, NVIDIA GPU execution, Kubernetes, Helm, and monitoring in subsequent days.

```text
Client
  |
HTTP API / routing
  |
FastAPI worker (validation, health checks)
  |
Preprocessing → PyTorch model → Postprocessing
                     |
                 CPU / GPU
  |
Prediction response
  +--> metrics and logs (incrementally added)
```

## Key formulas

- **End-to-end latency:** `L = L_queue + L_preprocess + L_inference + L_postprocess + L_network`
- **Throughput (RPS):** `completed_requests / duration_seconds`
- **Little's Law (steady state):** `average_in_flight = throughput × average_time_in_system`

Track **p50, p95, and p99** latency, not just averages. High GPU utilization does not imply acceptable tail latency.

## Troubleshooting exercises

See [troubleshooting.md](troubleshooting.md) for four scenarios discussed and their diagnoses, including startup probes, GPU saturation, single-GPU rolling updates, and CUDA OOM.

## Deliverables / evidence

- [x] Documented production-serving concepts
- [x] Proposed inference architecture and API contract
- [x] Defined measurements and benchmark methodology
- [x] Analyzed four production troubleshooting scenarios
- [ ] Implement inference API (Day 37)
- [ ] Containerize API (Day 38)
- [ ] Measure GPU-backed inference (Day 39 onward)
- [ ] Collect actual p50/p95/p99 latency and throughput results
- [ ] Deploy inference service to Kubernetes

**Evidence note:** No measured inference benchmarks or running inference API are claimed for Day 36. This day's outputs are design documents and a troubleshooting analysis.

## Concepts to retain

Production model serving turns a trained ML model into a reliable service that processes inference requests with acceptable latency, throughput, availability, and cost. End-to-end latency includes queuing, preprocessing, execution, postprocessing, and network time, so stage-level metrics and p95/p99 tail latency are important. Load models during worker initialization, guard readiness with Kubernetes probes, and avoid restarting functional services simply because they are overloaded. A Kubernetes GPU allocation does not reserve a particular amount of VRAM, and adding more workers to a saturated GPU may make performance worse. Safe operations require capacity planning, bounded queues or backpressure, graceful shutdown, resource-aware rollouts, and evidence-based troubleshooting.

## Next: Day 37

Implement a local FastAPI inference service with a loaded PyTorch model, validated inputs, `/predict`, `/health/live`, and `/health/ready`. The `/metrics` endpoint and full benchmarking will be introduced incrementally.
