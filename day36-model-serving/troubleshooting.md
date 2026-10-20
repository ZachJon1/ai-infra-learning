# Day 36 — Production Inference Troubleshooting Notes

These are **hypothetical design exercises**, not incidents observed in a live deployment.

## 1. Startup probe terminates a model-loading worker

**Symptoms:** Pod Running, not Ready, four restarts; model loads in about 90 seconds; startup probe checks every 5 seconds and fails after 10 attempts.

**Diagnosis:** Roughly `5 × 10 = 50 seconds` of startup failure budget is shorter than the required model-loading time. A failing startup probe can trigger a container restart before initialization completes.

**Repair:** Increase the startup budget with margin based on measured cold starts. For example:

```yaml
startupProbe:
  httpGet:
    path: /health/startup
    port: 8000
  periodSeconds: 5
  failureThreshold: 30
```

This is an example configuration; the API must actually implement `/health/startup` and use appropriate semantics. Verify with `kubectl describe pod`, restart counts, previous logs, and readiness status.

## 2. Saturation, queue growth, and rising p99

**Hypothetical metrics:** GPU utilization 65% → 99%; throughput 80 → 100 RPS; p99 latency 120 → 1,200 ms; queue depth 5 → 180.

**Diagnosis:** Likely serving-capacity bottleneck and increasing queue waits. Two queue snapshots do not alone prove unbounded sustained growth; inspect time series, arrivals, completions, timeouts, and per-stage timings.

**Response:** Confirm GPU/CPU/queue bottlenecks; optimize relevant stages; test batching if appropriate; add **real compute capacity** when scaling; apply bounded queues, backpressure, or controlled rejection. Do not blindly create additional workers on a saturated single GPU.

## 3. Rolling update stalls on a one-GPU node

**Symptoms:** Existing pod requests `nvidia.com/gpu: 1` and holds the node's only allocatable GPU; new surge pod is Pending with insufficient GPU.

**Diagnosis:** No schedulable GPU remains for the replacement pod. Memory sharing and Kubernetes GPU resource allocation are distinct constraints.

**Repair:** Consider another GPU node for zero-downtime rollout. On a single-GPU learning environment, use a rollout that allows old capacity to terminate before replacement, acknowledging downtime:

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 0
    maxUnavailable: 1
```

GPU time-slicing or other explicit sharing mechanisms require deliberate configuration and capacity/isolation evaluation; they do not create additional physical VRAM.

## 4. CUDA out-of-memory after successful scheduling

**Symptoms:** Pod gets the requested GPU but model execution fails with `torch.cuda.OutOfMemoryError`.

**Diagnosis:** Device allocation by Kubernetes is not a VRAM-size guarantee. Model weights, activations, batch size, concurrency, runtime allocations, and other GPU processes can exceed available memory.

**Investigation:** Examine `nvidia-smi`, actual peak allocated and reserved memory in PyTorch (`torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`), request/batch sizes, concurrency, and process memory usage. PyTorch counters alone do not account for all GPU memory consumers.

**Mitigations:** Use inference mode, reduce batch size/concurrency, fix unintended tensor retention, choose a smaller or quantized model where quality permits, or use hardware with more available VRAM. Benchmark after each change.

## Operator lessons

Measure before scaling; distinguish a **Pending** pod from a scheduled pod experiencing **CUDA OOM**; use startup/readiness/liveness checks for different purposes; account for model loading and GPU capacity during deployment; and prioritize latency/error SLOs over GPU utilization alone.
