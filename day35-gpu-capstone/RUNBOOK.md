# GPU Pod / NVIDIA Device Plugin Troubleshooting Runbook

**Scope:** Local Minikube + NVIDIA GPU lab. Use read-only diagnosis before modifying live components.

## 1. Identify the failing layer

| Symptom | Inspect | Common hypotheses |
|---|---|---|
| `nvidia-smi` fails on host | Driver, host GPU status | Host driver / device / system problem |
| Host GPU works but `nvidia.com/gpu` not allocatable | Device Plugin DaemonSet, kubelet registration, node taints | Plugin not scheduled, not healthy, or cannot discover GPU |
| Plugin DaemonSet `DESIRED=0` | Node taints, plugin tolerations, affinity and selectors | **No eligible nodes** |
| Plugin `DESIRED=1`, `READY=0` | Pod status, events, logs | Image, startup, GPU discovery, runtime, permissions issue |
| GPU Pod `Pending` | `kubectl describe pod`, node allocatable and allocated GPU | GPU resource insufficiency or scheduling restriction |
| Pod Running, but PyTorch reports CUDA unavailable | GPU limit/request, runtime device injection, driver libs, CUDA build | Missing GPU injection or application compatibility |
| Pod succeeds but emits NVIDIA diagnostic lines | Pod termination, application output, driver logs | **Unresolved until correlated** |

## 2. Read-only commands

```bash
nvidia-smi
kubectl config current-context
kubectl get nodes -o wide
kubectl describe node gpu-lab
kubectl -n kube-system get daemonsets
kubectl -n kube-system describe daemonset nvidia-device-plugin-daemonset
kubectl -n kube-system get pods -l name=nvidia-device-plugin-ds -o wide
kubectl -n kube-system logs -l name=nvidia-device-plugin-ds --tail=100
kubectl get pods -A
kubectl get events -n day35 --sort-by=.lastTimestamp
```

For a specific Pending pod:

```bash
kubectl describe pod -n day35 <POD_NAME>
```

Check `Events`, `Node-Selectors`, `Tolerations`, `Requests`, `Limits`, and `Node` assignment. A pod that is **Pending** is not necessarily suffering a CUDA error: it may never have been scheduled.

## 3. Incident A: GPU device plugin not scheduled

**Observed:** `DESIRED=0`, `READY=0`, GPU allocatable `0`.

```text
Node taint:     dedicated=gpu:NoSchedule
Plugin allows:  nvidia.com/gpu:NoSchedule
```

These are different keys, so the plugin did not tolerate the dedicated taint. Lab repair:

```bash
kubectl -n kube-system patch daemonset \
  nvidia-device-plugin-daemonset \
  --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/tolerations/-","value":{"key":"dedicated","operator":"Equal","value":"gpu","effect":"NoSchedule"}}]'
```

**Validate:**

```bash
kubectl -n kube-system get daemonsets
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.capacity.nvidia\.com/gpu}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
```

Observed after fix: `DESIRED=1`, `READY=1`, `gpu-lab 1 1`, plus Device Plugin log `Registered device plugin for 'nvidia.com/gpu' with Kubelet`.

> **Caution:** This patch modifies an addon-managed `kube-system` workload. For production use, update managed configuration instead of relying on a manual patch. Avoid broad tolerations that defeat the intended GPU reservation policy.

## 4. Incident B: GPU Pod over-requests devices

**Observed:** A Pod requested GPU limit `2`; the node had only `1` allocatable GPU.

```text
FailedScheduling: 0/1 nodes are available: 1 Insufficient nvidia.com/gpu
```

**Fix if the application only needs one GPU:** Change GPU limit to `1`, delete the Pending test pod, recreate from the corrected manifest, and inspect logs. Do **not** fix a resource mismatch by restarting a healthy device plugin.

```bash
kubectl delete pod day35-gpu-overrequest -n day35
kubectl apply -f gpu-overrequest.yaml
kubectl get pod day35-gpu-overrequest -n day35
kubectl logs -n day35 pod/day35-gpu-overrequest
```

Observed:

```text
CUDA available: True
Visible GPU count: 1
GPU test successful
```

For workloads truly requiring two GPUs, provisioning sufficient compatible resources or changing the workload topology—not simply lowering limits—is necessary.

## 5. Low utilization but no schedulable GPU

- GPU **capacity** = total resources advertised by the device plugin.
- GPU **allocatable** = total resources Kubernetes can allocate on the node, not free-now count.
- Approximate remaining GPU resource availability = allocatable minus GPU requests of active allocated Pods (assuming ordinary exclusive GPU allocation, without sharing).
- `nvidia-smi` utilization measures GPU activity, not Kubernetes resource allocation. An idle workload can continue to reserve the device.
- `nodeSelector` restricts candidate nodes; a `toleration` permits placement on a tainted node but does not itself select that node.

## 6. Open issue: NVIDIA diagnostic messages after successful job

Observed lines appended after successful work:

```text
ERROR: driverInitFileInfo 578 result=11
ERROR: init 664 result=11
ERROR: init 250 result=11
```

**Known:** CUDA returned `True`, matrix multiplication finished, and the original Job Pod terminated `Completed` with exit code `0`. A later one-GPU CUDA test also completed. **Unknown:** the subsystem emitting the lines and whether they affect later workloads.

If reproduced, retain the full stdout/stderr, timestamp, image, node status, pod container termination details, and driver/system logs, and compare with a minimal non-PyTorch CUDA container. Do not declare a driver failure from these lines alone.

## 7. Practical production considerations

- Add CPU/memory requests and limits for predictable scheduling and eviction behavior. A pod may be `BestEffort` even with a GPU limit if no CPU/memory resources are specified.
- Capture failed scheduling events **before they age out or the workload recovers**.
- Keep version, deployment, benchmark method and exact observed output in incident records.
- Use a managed DaemonSet/Helm/addon configuration for persistent toleration policy.
