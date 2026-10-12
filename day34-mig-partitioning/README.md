# Day 34 — MIG and GPU Partitioning

> AI Infrastructure / ML Systems — 220-Day Roadmap  
> Lab date: October 11, 2026  
> Status: **Draft for manual GitHub review**; nothing has been committed or pushed.

## Purpose

Explain and evaluate full-GPU allocation, NVIDIA Multi-Instance GPU (MIG), and GPU sharing; demonstrate real GPU contention and a handled CUDA allocator OOM on an RTX 3070; and diagnose Kubernetes GPU scheduling failures. **No actual MIG hardware was used.**

## Test environment

| Component | Observation / source |
|---|---|
| Host GPU | NVIDIA GeForce RTX 3070 Laptop GPU, 8,192 MiB physical VRAM |
| NVIDIA driver | 580.178.04 |
| Host PyTorch | 2.13.0+cu130; CUDA available |
| CUDA-visible GPU memory | 7,832 MiB during controlled allocator experiment |
| MIG mode | `Current: N/A`, `Pending: N/A` — unsupported hardware |
| Kubernetes | Minikube `gpu-lab`, Kubernetes v1.35.1, Docker driver |
| Device plugin image | `nvcr.io/nvidia/k8s-device-plugin:v0.18.2` (from running DaemonSet) |
| Node taint | `dedicated=gpu:NoSchedule` |
| Restored GPU capacity/allocatable | `nvidia.com/gpu: 1 / 1` |

The values above come from user-executed commands. The laptop does **not** implement real MIG partitions.

## What was learned

**Full GPU:** a workload receives exclusive scheduled use of the physical GPU when there is no configured GPU sharing; this can waste capacity for small workloads.

**MIG:** on supported NVIDIA datacenter GPUs, a physical GPU may be configured into GPU Instances (GIs), with allocated compute and memory resources. Compute Instances (CIs) are subdivisions of GI compute resources; distinct CIs under one GI share the GI's memory resources. Resource-profile geometry limits the combinations that can coexist.

**Time-slicing/ordinary sharing:** multiple workloads access shared GPU resources without MIG's dedicated per-instance compute and memory partitions. Configuring time-sliced replicas does not imply a fixed memory quota or SM share for each workload.

**Kubernetes:** NVIDIA device-plugin resource advertisement connects device discovery to scheduler accounting. `migStrategy: single` can expose MIG devices as `nvidia.com/gpu`; `mixed` exposes named resources such as `nvidia.com/mig-2g.10gb`. Setting the strategy does not itself configure physical partitions. Pods need the exact resource type, sufficient free allocation, and compatible placement constraints.

## Lab evidence: measured on the RTX 3070

### Real CUDA process sharing

A repeatable script, `gpu_share_probe.py`, performed CUDA matrix multiplication while retaining an explicitly allocated tensor. **The 512 MiB command argument was a tensor allocation, not a memory-enforced limit.**

| Metric | Solo | Concurrent A | Concurrent B |
|---|---:|---:|---:|
| Duration (s) | 40.00 | 40.00 | 40.01 |
| Matrix multiplications | 27,108 | 13,780 | 13,780 |
| Iterations/s | 677.65 | 344.46 | 344.45 |
| PyTorch tensor allocated (MiB) | 539.0 | 539.0 | 539.0 |
| Per-process memory in concurrent `nvidia-smi` (MiB) | not measured | 722 | 722 |

Concurrent aggregate throughput: **688.91 iterations/s**, **101.7%** of the solo result; individual concurrent throughput: **50.8%** of solo per process. These are **single-run observations**, not validated steady-state speedup or performance fairness. Process-level throughput was nearly halved, consistent with contention for the same physical GPU. The GPU was **not split into hardware partitions**.

During the concurrent run, five utilization samples read **100%**; one `nvidia-smi` snapshot showed **1,763 MiB** of total GPU memory in use, **82°C**, and **124 W / 125 W**. Other sampled memory readings ranged around 1,762–1,774 MiB before the jobs stopped. GPU memory after completion was **305 MiB**; a pre-lab reading was **378 MiB**. Different idle background usage should not be interpreted as workload memory leakage.

Reproduce using the existing local files:

```bash
cd ~/ai-infra/day34-mig-partitioning
python3 -u gpu_share_probe.py --name=solo --memory-mib=512 --duration=40 > solo.log 2>&1
# For concurrent test, launch two instances of gpu_share_probe.py in background
# and save output to model-a.log and model-b.log.
python3 analysis_script.py
```

**Methodology caveats:** Replicate trials, control warmup and GPU temperature/power clocks, compare request-tail latency separately, and avoid interpreting a single +1.7% aggregate change as a statistically meaningful gain.

### Real, controlled allocator OOM (not a MIG test)

The `controlled_allocator_oom.py` script imposed a **10% PyTorch CUDA caching-allocator cap** on the RTX 3070:

| Observation | Value |
|---|---:|
| CUDA-visible total memory | 7,832 MiB |
| Artificial allocator cap | 783 MiB |
| Initial successful tensor allocation | 256 MiB |
| Second attempted tensor allocation | 768 MiB |
| Physical GPU memory free at OOM (one test) | 6.95 GiB |
| Result | Expected `torch.OutOfMemoryError` was caught; process remained alive |

The allocator exception occurred because the process tried to exceed its program-level cap, **not because physical VRAM was exhausted**. CPU memory limits, scheduler GPU extended resources, and PyTorch allocator limits are different mechanisms.

### Real, controlled OOM alongside another process

In another test, Process A caught the allocator OOM while Process B kept computing:

| Process B outcome | Measured value |
|---|---:|
| Completed iterations | 23,009 |
| Elapsed | 25.00 s |
| Rate | 920.29 iterations/s |

This confirms **completion despite a handled allocator OOM in another process**. It does **not** establish MIG fault isolation, guaranteed independence, or reproducible throughput improvement. Compared with the previous 40-second baseline, duration, thermal conditions and workload timing differed.

## Lab evidence: simulated / reference only

### Python MIG inventory simulation

`mig_scheduler_sim.py` modeled an A100 40 GB inventory with one device each of `1g.5gb`, `2g.10gb`, and `3g.20gb`. It did **not** create or benchmark real partitions. With classifier→`1g`, embeddings→`2g`, reranker→`2g`, vision→`3g`, the **reranker stayed Pending**. After changing reranker→`3g`, the **vision model stayed Pending**. This simulates *exact-match resource accounting*, not Kubernetes' full scheduling algorithm or physical placement validation.

### Reference MIG manifest

`mig-inference-pod.yaml` requested `nvidia.com/mig-2g.10gb: 1` with a matching `dedicated=gpu:NoSchedule` toleration and an `accelerator=a100-mig` node selector. `kubectl create --dry-run=client -f mig-inference-pod.yaml -o yaml` succeeded. **Dry-run did not establish scheduleability or actual MIG hardware support.** This manifest must **not** be applied as a real MIG workload to the RTX 3070.

### Real Kubernetes synthetic scheduling failure

`mig-profile-unavailable.yaml` was applied to Minikube and requested an unavailable `nvidia.com/mig-2g.10gb` resource. The real Kubernetes scheduler reported:

```text
0/1 nodes are available: 1 Insufficient nvidia.com/mig-2g.10gb.
preemption: 0/1 nodes are available: 1 Preemption is not helpful for scheduling.
```

The actual node advertised `nvidia.com/gpu: 1` and **no** `nvidia.com/mig-*` resource. The intentionally Pending pod was deleted after inspection. This was a **real Kubernetes scheduler result from an intentionally impossible request**, not a real MIG partition failure.

## Reusable troubleshooting and production decisions

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for the taint/device-plugin incident, missing MIG resource, and runtime allocator OOM, including diagnostic commands.

For production deployments, choose GPU assignments using peak memory **and** compute demand, p95/p99 latency under actual concurrency, queue depth, resource fragmentation, cost per SLO-compliant request, and disruption caused by MIG reconfiguration. For one hypothetical two-A100 design, the balanced A100 40 GB MIG node can host `3g.20gb × 1`, `2g.10gb × 1`, and `1g.5gb × 2`, while a separate unpartitioned GPU can handle one larger LLM workload. This is an initial architectural hypothesis, **not a proven SLO-compliant production deployment**.

## Repository layout (suggested)

```text
day34-mig-partitioning/
├── README.md                   # This report
├── TROUBLESHOOTING.md          # Incident/runbook documentation
├── benchmarks.csv              # Measurements and simulated entries, clearly labeled
├── gpu_share_probe.py          # Existing local lab script
├── controlled_allocator_oom.py # Existing local lab script
├── mig_scheduler_sim.py        # Existing local simulation
├── analysis_script.py          # Existing local comparison script
├── mig-inference-pod.yaml      # Reference manifest (dry-run only)
├── mig-profile-unavailable.yaml# Synthetic scheduling failure manifest
└── *.log                       # Optional raw measured logs, checked for sensitive data
```

## References

- [NVIDIA MIG User Guide — Concepts](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html)
- [NVIDIA MIG User Guide — Supported MIG Profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-mig-profiles.html)
- [NVIDIA GPU Operator — MIG Manager](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-mig.html)
- [NVIDIA Kubernetes Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
- [Kubernetes Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)


