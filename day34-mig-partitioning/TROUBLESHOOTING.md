# Day 34 — GPU Scheduling and Isolation Troubleshooting

**Environment:** RTX 3070 Laptop GPU, Minikube `gpu-lab`, NVIDIA Device Plugin v0.18.2. The GPU does **not** support MIG. All MIG inventory examples are simulated or synthetic scheduler tests.

## Incident 1 — Host GPU works, Kubernetes GPU capacity is zero

**Symptom:** `nvidia-smi` works on the host and `minikube ssh -- nvidia-smi -L` detects the RTX 3070, but the Kubernetes node advertises zero GPU capacity and allocatable resources.

**Observed state:** `gpu-lab` was Ready. The node had `dedicated=gpu:NoSchedule`; the NVIDIA Device Plugin DaemonSet had `0` desired/current pods, with only `nvidia.com/gpu:NoSchedule op=Exists` in its tolerations. The existing toleration did not match the dedicated taint.

**Root cause:** GPU node scheduling policy prevented the device-plugin DaemonSet from being eligible for the node. This is a **DaemonSet**, not a ReplicaSet.

**Diagnosis:**

```bash
kubectl describe node gpu-lab | grep -A 5 Taints
kubectl -n kube-system describe daemonset nvidia-device-plugin-daemonset
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu
```

**Fix applied (real):**

```bash
kubectl -n kube-system patch daemonset nvidia-device-plugin-daemonset \
  --type='json' \
  -p='[{"op":"add","path":"/spec/template/spec/tolerations/-","value":{"key":"dedicated","operator":"Equal","value":"gpu","effect":"NoSchedule"}}]'
```

**Verified after fix:** Device Plugin DaemonSet: desired/current/ready `1/1/1`; pod Running. Node GPU capacity `1`, allocatable `1`.

**Operational guardrail:** A successful rollout command with `Desired=0` does not prove functional service. Check desired/current/ready counts and `Capacity` vs `Allocatable`. The patch may be replaced if a Minikube addon reconciles or recreates the DaemonSet; document and implement a persistent configuration mechanism separately.

## Incident 2 — Kubernetes Pending pod: MIG profile unavailable

**Test type:** **Real Kubernetes scheduling with an intentionally impossible resource request**, not a MIG hardware test.

**Symptom:** Pod `day34-mig-failure` remained Pending and unassigned (`Node: <none>`), despite tolerating `dedicated=gpu:NoSchedule`.

**Event observed:**

```text
0/1 nodes are available: 1 Insufficient nvidia.com/mig-2g.10gb.
preemption: 0/1 nodes are available: 1 Preemption is not helpful for scheduling.
```

**Root cause:** The RTX 3070 node advertises `nvidia.com/gpu: 1`, **not** `nvidia.com/mig-2g.10gb`. No matching extended resource exists to schedule.

**Diagnosis:**

```bash
kubectl describe pod day34-mig-failure
kubectl get node gpu-lab -o jsonpath='{.status.allocatable}{"\n"}'
```

**Corrective approach:** On a real MIG-capable cluster, inspect the physical MIG geometry, MIG Manager state, plugin strategy (`single` vs `mixed`), registered device health, and free profile allocation. On the laptop, there is no MIG-based corrective action; an application may instead request the full `nvidia.com/gpu` if its deployment design permits. CPU/RAM changes and node labels cannot create a missing MIG device.

**Cleanup:** `kubectl delete -f mig-profile-unavailable.yaml` was issued after capturing evidence.

## Incident 3 — CUDA OOM with free physical GPU VRAM

**Test type:** **Real PyTorch allocator cap**, not device-level OOM or MIG isolation.

**Symptom:** `torch.OutOfMemoryError` when trying to allocate a 768 MiB tensor despite approximately 6.95 GiB of reported GPU memory being free.

**Root cause:** `torch.cuda.set_per_process_memory_fraction(0.10)` capped the process's caching allocator around 783 MiB. A prior 256 MiB allocation plus the requested 768 MiB exceeded this imposed allocator budget. The exception was caught and execution continued.

**Diagnosis:** Inspect PyTorch OOM output, process memory allocations, `nvidia-smi`, allocator settings, peak tensor usage, model/batch size and other GPU consumers. A CPU memory limit adjustment is not a VRAM remedy.

**Failure containment test:** While a second process ran a GPU compute workload, Process A hit and handled the controlled OOM. Process B finished 23,009 iterations in 25.00 s (920.29 iterations/s). This only proves completion under a handled allocator exception; it does not demonstrate hardware or fault isolation.

## Fast troubleshooting sequence

1. **Discoverability:** host `nvidia-smi` → `minikube ssh -- nvidia-smi -L` → device-plugin DaemonSet running/registered → Node `capacity` and `allocatable`.
2. **Scheduling:** Pod `describe` Events → exact GPU extended resource name and count → taints/tolerations, selectors/affinity, priorities and competing allocations.
3. **Execution:** Did the pod get a node and start a container? Investigate runtime device allocation, CUDA compatibility, application errors and GPU memory/compute pressure.
4. **Isolation/performance:** Check p50/p95/p99 latency, queue time, throughput and per-instance metrics; distinguish shared-GPU contention from MIG resource constraints. Respect reconfiguration disruption.

## Practical interview statement

> A node can be Ready and a GPU can work in `nvidia-smi` even while Kubernetes advertises zero GPU allocatable resources. In our lab, a custom `NoSchedule` taint prevented the NVIDIA Device Plugin DaemonSet from running. Adding the matching toleration restored the plugin and `nvidia.com/gpu: 1` allocatable capacity. Separately, a synthetic request for an unavailable MIG profile stayed Pending because Kubernetes extended resource names must match, not because CUDA or VRAM had failed.
