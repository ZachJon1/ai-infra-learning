# Day 35 — Verified Incident Records

## Incident 01 — GPU Device Plugin had zero eligible nodes

**Date:** 2026-10-17  
**Cluster:** `gpu-lab` (Minikube / Kubernetes v1.35.1)

### Symptoms

- Host `nvidia-smi` detected the NVIDIA GeForce RTX 3070 Laptop GPU.
- Device Plugin DaemonSet: `DESIRED 0`, `CURRENT 0`, `READY 0`.
- Node GPU resource: `nvidia.com/gpu` allocatable `0` (the observed capacity value changed across checks).

### Diagnosis

- Node taint: `dedicated=gpu:NoSchedule`.
- Device-plugin DaemonSet toleration: `nvidia.com/gpu:NoSchedule` (key did not match).
- No node selector or required node affinity preventing placement in the DaemonSet specification.

**Root cause:** The DaemonSet did not tolerate the node's dedicated GPU taint and therefore targeted zero eligible nodes.

### Remediation

Added exact toleration `dedicated=gpu:NoSchedule` to the DaemonSet pod template with a JSON patch.

### Verified outcome

- DaemonSet `DESIRED 1`, `CURRENT 1`, `READY 1`, `AVAILABLE 1`.
- Device Plugin Pod Running, `0` restarts at capture time.
- Log confirmed registration of `nvidia.com/gpu` with kubelet.
- Node GPU capacity `1`, allocatable `1`.

### Prevention

Ensure platform DaemonSets carry the necessary tolerations for intentional GPU-only node taints. Put addon-specific settings under configuration management.

---

## Incident 02 — GPU resource over-request prevented scheduling

**Date:** 2026-10-17  
**Namespace:** `day35`

### Symptoms

- Pod `day35-gpu-overrequest`: `Pending`, no node assigned.
- Request/limit: `nvidia.com/gpu: 2`.
- Available node resource advertised: one GPU.
- Pod event:

```text
Warning  FailedScheduling ... 0/1 nodes are available: 1 Insufficient nvidia.com/gpu. no new claims to deallocate, preemption: 0/1 nodes are available: 1 Preemption is not helpful for scheduling.
```

### Diagnosis

Pod satisfied the GPU node's selector and taint toleration but asked for more GPUs than the eligible node could allocate.

**Root cause:** GPU resource request of 2 exceeded the single GPU allocatable on the node.

### Remediation

Preserved `gpu-overrequest-broken.yaml`; edited the working manifest from `nvidia.com/gpu: 2` to `1`; deleted and recreated the test Pod.

### Verified outcome

The pod transitioned `ContainerCreating → Running → Completed` and printed:

```text
CUDA available: True
Visible GPU count: 1
GPU test successful
```

### Prevention

Validate workload GPU requirements against the intended node class and resource budget before deployment. Do not lower a genuinely required multi-GPU workload's request merely to make it schedule.

---

## Additional experiment — Single-GPU contention

**Job:** `day35-gpu-contention`  
**Requested:** `completions: 2`, `parallelism: 2`, each Pod `nvidia.com/gpu: 1`.

Observed pods:

```text
day35-gpu-contention-24w8c  Running
day35-gpu-contention-ssf9c  Pending
...
day35-gpu-contention-24w8c  Completed
day35-gpu-contention-ssf9c  ContainerCreating → Running → Completed
```

**Inference supported by sequence:** Without GPU sharing, the workload ran sequentially on the only allocatable GPU. The exact `FailedScheduling` event for the Pending contention Pod was not retained in the supplied record.

---

## Unresolved observation — Post-success diagnostic lines

Both PyTorch GPU execution logs and the corrected over-request test included messages such as `ERROR: driverInitFileInfo 578 result=11` after success output. The first Job Pod was verified `Completed` with exit code `0`; the later test reached `Completed`. Source/cause of these diagnostic lines has not been confirmed. Preserve as an open issue, not a confirmed failed job.
