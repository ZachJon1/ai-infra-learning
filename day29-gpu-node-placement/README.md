# Day 29 - GPU Node Labels, Taints, Tolerations and Scheduling

## Objective

Practice production-style Kubernetes GPU placement and troubleshooting using:

- Node labels and `nodeSelector`
- GPU-node taints
- Pod tolerations
- NVIDIA extended resources (`nvidia.com/gpu`)
- NVIDIA Device Plugin
- GPU scheduling failure diagnosis

The lab used a Minikube GPU cluster backed by an NVIDIA GeForce RTX 3070.

## Environment

```text
Minikube profile: gpu-lab
Container runtime: Docker
GPU: NVIDIA GeForce RTX 3070
GPU memory: 8 GB
NVIDIA driver: 580.178.04
CUDA reported by driver: 13.0
NVIDIA Device Plugin: v0.18.2
```

The GPU-enabled Minikube profile was started with:

```bash
minikube start -p gpu-lab \
  --driver=docker \
  --container-runtime=docker \
  --gpus=all \
  --cpus=4 \
  --memory=6144
```

## GPU Node Configuration

The GPU node was labeled:

```bash
kubectl label node gpu-lab accelerator=gpu
kubectl label node gpu-lab workload=inference
```

The node was protected using a custom taint:

```bash
kubectl taint node gpu-lab dedicated=gpu:NoSchedule
```

This prevents workloads without a matching toleration from being scheduled onto the GPU node.

---

## Experiment 1 — Node Selector Without Toleration

`blocked-gpu-pod.yaml` selects the GPU node:

```yaml
nodeSelector:
  accelerator: gpu
```

but does not tolerate:

```text
dedicated=gpu:NoSchedule
```

Result:

```text
gpu-placement-blocked   0/1   Pending
```

Scheduler event:

```text
0/1 nodes are available: 1 node(s) had untolerated taint(s)
```

### Lesson

A matching node selector does not override a node taint.

The selector determines where the Pod is eligible to run, while the toleration determines whether it is permitted onto a tainted node.

---

## Experiment 2 — Matching Toleration

`allowed-gpu-pod.yaml` contains both:

```yaml
nodeSelector:
  accelerator: gpu
```

and:

```yaml
tolerations:
  - key: dedicated
    operator: Equal
    value: gpu
    effect: NoSchedule
```

Result:

```text
gpu-placement-allowed   1/1   Running   gpu-lab
```

However, the workload does not request:

```yaml
nvidia.com/gpu: 1
```

so it runs on the GPU node without being allocated the GPU.

### Lesson

Running **on a GPU node** is not equivalent to being **allocated a GPU**.

---

## Experiment 3 — Real GPU Allocation

`real-gpu-pod.yaml` adds:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

The final workload successfully ran:

```text
real-gpu-pod   1/1   Running
```

Running `nvidia-smi` inside the Kubernetes Pod confirmed access to the physical RTX 3070.

```text
NVIDIA GeForce RTX 3070
Driver Version: 580.178.04
CUDA Version: 13.0
```

This validates the complete path:

```text
Pod GPU request
      ↓
Kubernetes Scheduler
      ↓
Kubelet
      ↓
NVIDIA Device Plugin
      ↓
Container GPU access
      ↓
Physical RTX 3070
```

---

## Troubleshooting Incident — Device Plugin Lost After Tainting GPU Node

A particularly useful failure occurred after restarting the NVIDIA Device Plugin DaemonSet.

The GPU node had:

```text
dedicated=gpu:NoSchedule
```

but the NVIDIA Device Plugin DaemonSet originally only contained:

```text
nvidia.com/gpu:NoSchedule
```

as a toleration.

The already-running device-plugin Pod survived when the custom taint was initially added because `NoSchedule` does not evict existing Pods.

However, after restarting the DaemonSet, the old device-plugin Pod was deleted and its replacement could not schedule onto the tainted GPU node.

Observed state:

```text
DESIRED   CURRENT   READY
0         0         0
```

GPU resources eventually disappeared from Kubernetes:

```text
CAPACITY      0
ALLOCATABLE   0
```

while `nvidia-smi` still worked directly on the Minikube node.

This demonstrated that the physical GPU and driver were healthy; the problem was at the Kubernetes/device-plugin layer.

### Fix

The custom GPU taint was added as a toleration to the NVIDIA Device Plugin DaemonSet.

After repair:

```text
DESIRED   CURRENT   READY
1         1         1
```

and:

```text
CAPACITY      1
ALLOCATABLE   1
```

The real GPU workload could then be scheduled successfully.

### Lesson

Custom taints on specialized GPU nodes must also be tolerated by infrastructure DaemonSets that need to run on those nodes.

---

## Experiment 4 — GPU Capacity Exhaustion

The RTX 3070 provides one Kubernetes GPU resource:

```text
nvidia.com/gpu: 1
```

With `real-gpu-pod` already consuming that GPU, a second GPU workload was created.

Result:

```text
too-many-gpus   0/1   Pending
```

Scheduler event:

```text
0/1 nodes are available: 1 Insufficient nvidia.com/gpu
```

### Lesson

This failure is different from an untolerated taint.

```text
Untolerated taint
→ scheduling-policy problem

Insufficient nvidia.com/gpu
→ GPU resource-capacity problem

UnexpectedAdmissionError after scheduling
→ kubelet/device-plugin/GPU-allocation problem
```

---

## Key Takeaways

GPU scheduling requires separating three concepts:

1. **Placement** — labels, `nodeSelector`, and node affinity determine which nodes a workload may target.
2. **Permission** — taints protect specialized nodes and tolerations permit selected workloads onto them.
3. **Allocation** — `nvidia.com/gpu` explicitly requests an actual GPU resource.

A Pod can therefore run on a GPU node without owning a GPU.

Useful troubleshooting signals include:

```text
untolerated taint
→ scheduling policy

Insufficient nvidia.com/gpu
→ GPU capacity

Pod scheduled + GPU allocation failure
→ kubelet / NVIDIA Device Plugin / device health
```

The lab also demonstrated that GPU-node taints affect not only application workloads but potentially the infrastructure DaemonSets required to expose the GPU to Kubernetes.
