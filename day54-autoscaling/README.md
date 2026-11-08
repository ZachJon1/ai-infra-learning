# Day 54 — Autoscaling Inference Services

**Track:** AI Infrastructure & ML Systems — 220-Day Roadmap  
**Phase:** 3 — Model Serving / Inference Infrastructure  
**Previous:** Day 53 — Load Testing Inference Systems  
**Next:** Day 55 — Model-Serving Capstone

## Goal

Design and troubleshoot an autoscaling strategy for a GPU-backed inference service that maintains acceptable latency **without wasting GPU capacity**. This lesson builds on Kubernetes HPA (Day 15), GPU scheduling, batching, vLLM, KV caching, and inference load testing.

> **Status:** Concepts, design exercises, and troubleshooting scenarios completed. The KEDA configuration below is a **reference example**, not a claim of successful deployment or measured production performance. Validate it against your cluster and traffic generator before reporting results.

## Learning objectives

- Explain the difference between scaling inference Pods and provisioning GPU nodes.
- Select scaling signals for GPU/LLM workloads rather than relying only on CPU usage.
- Understand HPA and KEDA, including Prometheus-based queue-depth scaling.
- Account for GPU allocation, model loading, readiness, scale-up delays, and scale-down stabilization.
- Diagnose failures where desired replicas increase but usable serving capacity does not.
- Evaluate autoscaling against latency, throughput, reliability, and cost metrics.

## 1. Architecture

```text
                              Client requests
                                     |
                            API gateway / router
                                     |
                              Inference Service
                                     |
                      +--------------+--------------+
                      |              |              |
                  vLLM Pod 1     vLLM Pod 2      vLLM Pod N
                  (1 GPU)        (1 GPU)         (1 GPU)
                      |              |              |
                      +------- /metrics endpoints ---+
                                     |
                                 Prometheus
                                     |
                          KEDA Prometheus scaler
                                     |
                          Kubernetes HPA (replicas)
                                     |
                          Deployment desired count
                                     |
                          Kubernetes Pod scheduler
                                     |
                    Existing GPU nodes / new GPU nodes
```

**Important:** KEDA/HPA adjust the *desired Pod count*. The Kubernetes scheduler must still place the Pods, and a GPU node autoscaler (when available) must separately provision additional node capacity. Pods serve traffic only after scheduling, startup, model loading, and readiness checks succeed.

## 2. Choosing the right scaling signals

| Signal | What it tells us | Recommended role |
|---|---|---|
| Waiting requests / queue wait time | Requests are accumulating before execution | Strong **leading scaling signal** |
| P95 latency, P95 TTFT, TPOT | Whether users experience acceptable response times | **Service-level validation** and diagnosis |
| GPU utilization | GPU is busy; does not independently imply an SLO violation | Supporting resource-pressure metric |
| KV-cache usage | Cache pressure and possible inference scheduling constraints | Diagnostic and tuning metric |
| Tokens/s, requests/s | Actual serving throughput | Capacity and cost planning |
| Desired vs Ready Pods | Difference between scaling decisions and usable capacity | Scaling-health metric |
| GPU-hours, cost per million tokens | Infrastructure efficiency | Cost-efficiency validation |

For vLLM, available Prometheus metric families include `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `vllm:time_to_first_token_seconds`, and `vllm:e2e_request_latency_seconds`. Metric names and labels can vary by vLLM version; confirm at the running server's `/metrics` endpoint.

### Why not simply scale at 95% GPU utilization?

High GPU utilization can mean efficient use of hardware. If latency is healthy and the waiting queue is small, adding replicas may increase cost without improving the user experience. Conversely, CPU usage might remain low while GPU-backed inference requests accumulate.

## 3. HPA and queue-based sizing

The simplified HPA ratio calculation is:

```text
desired_replicas = ceil(current_replicas * current_metric / target_metric)
```

Example (CPU): 4 current replicas, 90% utilization, 60% target:

```text
ceil(4 * 90/60) = 6 total replicas
```

For an **aggregate queue-depth metric** exposed to HPA as `AverageValue`, the simplified target is:

```text
desired_replicas = ceil(total_waiting_requests / target_waiting_per_replica)
```

Example: 56 waiting requests and a target of 8 waiting per replica:

```text
ceil(56/8) = 7 total replicas
```

If there are already 3 replicas, that means **4 additional replicas**, not 7 additional replicas. Actual HPA behavior also respects min/max replicas, policy limits, stabilization, tolerance, and metric/readiness handling.

## 4. KEDA + Prometheus reference configuration

**Prerequisites:** Kubernetes cluster; a Deployment named `vllm-inference` in namespace `inference`; KEDA installed; Prometheus collecting vLLM metrics; network access from KEDA to Prometheus; appropriate GPU node resources.

The accompanying file [`vllm-scaledobject.yaml`](./vllm-scaledobject.yaml) provides a reference `ScaledObject` with:

- Minimum **1** replica to retain some warm serving capacity.
- Maximum **4** replicas as an *illustrative cap*; set it based on real GPU availability and capacity tests.
- Target **8 waiting requests per replica**, using Prometheus + `AverageValue`.
- **300-second** scale-down stabilization to resist rapid down/up oscillation.
- A scale-up policy permitting at most **2 Pods per 60 seconds**.
- `ignoreNullValues: "false"` so an absent Prometheus result is treated as an error, rather than silently being interpreted as no demand. Also configure operational alerts and an appropriate failure policy.

Inspect and modify the example before applying it:

```bash
# First confirm the target Deployment, KEDA CRD, and GPU capacity.
kubectl get deploy -n inference
kubectl get crd scaledobjects.keda.sh
kubectl get nodes -o wide
kubectl describe nodes | grep -A 9 -E '^Allocatable:|^Allocated resources:'

# Confirm vLLM exposes metrics (adjust port if your Deployment differs).
kubectl -n inference port-forward deploy/vllm-inference 8000:8000
# In a second terminal:
curl -s http://localhost:8000/metrics | grep 'vllm:num_requests_waiting'

# Only after adapting the Prometheus address/metric labels and max replicas:
kubectl apply -f vllm-scaledobject.yaml
kubectl get scaledobject -n inference
kubectl get hpa -n inference
kubectl describe hpa -n inference
kubectl get pods -n inference -w
```

**Metric-query warning:** The sample PromQL selector `job="vllm"` must match the labels on *your* Prometheus time series and be scoped to the intended inference Deployment. Do not use a global queue sum if it includes unrelated workloads. Avoid adding `or vector(0)` simply to hide missing telemetry: missing metrics are not proof of an empty queue. Validate query results directly in Prometheus before enabling scaling.

**Single-GPU laptop/workstation:** If each inference Pod requests one exclusive `nvidia.com/gpu` and only one GPU is allocatable, set `maxReplicaCount: 1` for the real GPU workload. Use a separate CPU-only mock service to demonstrate multi-replica behavior, or document the multi-GPU architecture conceptually; do not claim an unperformed multi-GPU experiment.

## 5. Break/fix runbook

### Case A: HPA wants 4 Pods, but only 1 is Ready

```bash
kubectl get hpa -n inference
kubectl get deploy,pods -n inference -o wide
kubectl describe pod <pending-pod-name> -n inference
kubectl get events -n inference --sort-by=.lastTimestamp
kubectl describe node <gpu-node-name>
```

**Investigate in order:** scheduling events (`Insufficient nvidia.com/gpu`, selectors, taints/tolerations, affinity), node allocatable vs allocated resources, device-plugin health, then container start/model loading/readiness if the Pod has actually been scheduled. A Pending unscheduled Pod may not have application logs yet.

### Case B: Replicas increase but P95 latency remains high

Possible explanations: new Pods are still downloading/loading weights or warming up; new Pods have not become Ready; traffic is unevenly routed; queueing/batching configuration remains inefficient; KV-cache pressure, prefill/decode contention, storage, or networking is the true bottleneck.

**Compare** desired vs Ready Pods, queue length, queue wait time, P95 TTFT, end-to-end latency, token throughput, and GPU resource usage before assuming more replicas are the solution.

### Case C: Traffic drops and the system repeatedly scales down/up

Keep an appropriate **scale-down stabilization window**, with enough warm replicas for expected bursts. This saves repeated model load costs but incurs ongoing GPU-hours. Benchmark the tradeoff rather than assuming one universal window is correct.

### Case D: Prometheus/KEDA telemetry fails

Distinguish **missing metrics** from a **true zero queue**. Verify scrape targets, query labels, Prometheus connectivity, KEDA operator logs, and `ScaledObject`/HPA conditions. Decide explicitly how many fallback replicas to retain on scaler failure; alert on telemetry outages.

## 6. Design exercise (completed conceptually)

**Requirements (hypothetical):** baseline 40 requests/s; peak 200 requests/s; P95 TTFT objective ≤ 2 seconds; 90-second model startup; vLLM on Kubernetes; one GPU per Pod.

**Design decisions:**

1. Use **waiting requests / queue depth** as the main scale-out indicator, via Prometheus and KEDA.
2. Validate the outcome against **P95 TTFT**, TPOT, end-to-end latency, throughput, and rejection/error rates.
3. Maintain **enough warm replicas for regular traffic**; determine the actual count using single-replica capacity measurements and appropriate headroom.
4. Use gradual scale-down to avoid repeated expensive 90-second starts when traffic briefly falls.
5. Buffer only a **bounded** number of requests. When a spike exceeds available capacity, use admission control/rate limiting and explicit overload responses rather than pretending queued requests will always meet the latency target.
6. Increase GPU node capacity when HPA requests exceed currently schedulable GPU resources, where a node autoscaler and infrastructure permit it.
7. Evaluate both **user-facing SLO attainment** and **GPU-hours / cost per million output tokens**.

**Unresolved capacity question:** The scenario does *not* provide sustainable throughput per inference replica, token distributions, or startup/node-provisioning measurements. Therefore neither the required minimum replicas nor the correct maximum can be inferred reliably from requests/s alone.

## 7. Suggested benchmark and evidence table

Do **not** fill these with invented values. Collect results under baseline, spike, and recovery workloads.

| Scenario | Input req/s | Desired/Ready Pods | P95 TTFT | P95 E2E latency | Peak queue | Tokens/s | GPU-hours | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Not run |
| Spike | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Not run |
| Recovery | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Not run |

For the Day 55 capstone, preserve: manifest versions, cluster/GPU details, traffic-generator configuration, Pod/event snapshots, graphs, commands, before/after comparisons, and at least one reproducible failure/fix.

## 8. Interview takeaways

- **HPA scales replicas; it does not create GPUs.** Additional replicas can remain Pending when hardware is unavailable.
- **KEDA connects external demand signals to HPA.** GPU/LLM autoscaling should usually incorporate inference workload signals, not CPU alone.
- **Queue depth is useful for earlier intervention; P95 TTFT is a validation target.** Measure both.
- **A scaling decision is not usable capacity.** GPU scheduling, weight loading, warmup, and readiness can introduce long delays.
- **Stabilization trades idle GPU cost for resilience to bursts.** Tune it based on actual workload patterns.
- **Missing monitoring is an operational fault, not zero traffic.** Design safe failure handling.

## 9. Key details to retain

Inference autoscaling adjusts serving capacity to meet request demand while controlling GPU cost. HPA changes replica counts, while KEDA can feed it Prometheus-derived external metrics such as vLLM queue depth. Queue length or waiting time often provides an earlier scale-out signal than latency, whereas P95 TTFT, TPOT, throughput, and errors show whether the scaling policy actually protects users. High GPU utilization alone does not require scaling. Successful scale-out depends on schedulable GPU nodes, model loading, warmup, and readiness; stabilization windows and warm replicas reduce thrashing but incur cost. Always distinguish desired from Ready replicas, treat missing telemetry as a failure to investigate, and judge the design using both performance and GPU economics.

## References

- [Kubernetes — Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/)
- [KEDA — Prometheus scaler](https://keda.sh/docs/2.20/scalers/prometheus/)
- [KEDA — ScaledObject specification](https://keda.sh/docs/2.20/reference/scaledobject-spec/)
- [vLLM — Production metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
