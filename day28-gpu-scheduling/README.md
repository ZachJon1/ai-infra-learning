# Day 28 — Scheduling GPU Workloads in Kubernetes

**Track:** AI Infrastructure & ML Systems · **Phase:** GPU Infrastructure  
**Lab date:** 2026-10-03 · **Status:** GPU scheduling and CUDA vector addition demonstrated

## Goals

- Make an NVIDIA GPU visible to Kubernetes through the device plugin.
- Schedule a GPU-requesting Pod and distinguish allocation from utilization.
- Reproduce contention when two Pods request one available GPU.
- Diagnose GPU discovery and Linux resource-limit failures.
- Run and validate a CUDA workload from a Kubernetes Pod.

## Environment

| Component | Lab environment |
| --- | --- |
| Host | Ubuntu 24.04 |
| GPU | NVIDIA GeForce RTX 3070, 8 GiB |
| NVIDIA driver | 580.178.04 (reported CUDA compatibility 13.0) |
| Minikube | v1.38.1, Docker driver, `gpu-lab` profile |
| Kubernetes | v1.35.1 |
| Container runtime | Docker 29.2.1 |
| NVIDIA device-plugin image | `nvcr.io/nvidia/k8s-device-plugin:v0.18.2` |
| Namespace | `day28-gpu` |

## Files

- `gpu-pod.yaml` — requests one GPU using a lightweight BusyBox worker.
- `gpu-contender.yaml` — second Pod requesting one GPU to observe contention.
- `cuda-test.yaml` — NVIDIA CUDA vector-addition test.

## Observations and root-cause investigation

Initially, the host `nvidia-smi` succeeded and `docker run --rm --gpus all ubuntu:24.04 nvidia-smi` succeeded, but Kubernetes did not advertise a GPU. The GPU-requesting Pod remained `Pending` with:

```text
0/1 nodes are available: 1 Insufficient nvidia.com/gpu
```

The NVIDIA device-plugin DaemonSet existed but its Pod was in `CrashLoopBackOff`. Its logs showed:

```text
failed to create FS watcher for /var/lib/kubelet/device-plugins/: too many open files
```

Docker inspection showed a GPU device request (`Count=-1`, `Capabilities=[["gpu"]]`) and `/dev/nvidia*` device nodes were visible in the Minikube container. On the host, `fs.inotify.max_user_instances` was `128`. We increased it to `8192` and restarted the NVIDIA device-plugin DaemonSet. The rollout succeeded, and the node then reported `nvidia.com/gpu: 1`.

The before/after results strongly support inotify instance exhaustion as the immediate plugin startup blocker; they do not independently establish the source of every NVIDIA initialization diagnostic line.

### Persistent host setting

Stored in `/etc/sysctl.d/90-ai-infra-inotify.conf`:

```ini
fs.inotify.max_user_instances = 8192
```

Applied with:

```bash
sudo sysctl -p /etc/sysctl.d/90-ai-infra-inotify.conf
kubectl --context=gpu-lab rollout restart daemonset/nvidia-device-plugin-daemonset -n kube-system
```

**Note:** `fs.inotify.max_user_instances` is an inotify-instance limit per real user, **not** the same as the process file-descriptor limit (`ulimit -n`).

## Reproduce scheduling

```bash
kubectl --context=gpu-lab create namespace day28-gpu --dry-run=client -o yaml \
  | kubectl --context=gpu-lab apply -f -
kubectl --context=gpu-lab apply -f gpu-pod.yaml
kubectl --context=gpu-lab get pods -n day28-gpu -o wide
kubectl --context=gpu-lab describe node gpu-lab
```

Observed: `gpu-demo` became `Running` on `gpu-lab`; node allocated GPU resources showed `nvidia.com/gpu: 1` requested out of one allocatable GPU. The BusyBox output `Scheduled` verified container startup but did **not** itself prove CUDA computation.

### GPU contention

```bash
kubectl --context=gpu-lab apply -f gpu-contender.yaml
kubectl --context=gpu-lab describe pod gpu-contender -n day28-gpu
```

Observed: with `gpu-demo` running and reserving the only GPU, `gpu-contender` stayed `Pending` with `Insufficient nvidia.com/gpu`. This is reservation-based scheduling, regardless of low physical GPU utilization. We later deleted `gpu-contender` before running the CUDA test. **The expected Pending-to-Running transition after deleting `gpu-demo` was not captured in the shared terminal transcript**, so it should not be claimed as verified here.

## CUDA workload verification

After freeing the GPU:

```bash
kubectl --context=gpu-lab apply -f cuda-test.yaml
kubectl --context=gpu-lab get pod cuda-test -n day28-gpu
kubectl --context=gpu-lab logs cuda-test -n day28-gpu
kubectl --context=gpu-lab get pod cuda-test -n day28-gpu \
  -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}{"\n"}'
```

Observed:

```text
STATUS: Completed
[Vector addition of 50000 elements]
CUDA kernel launch with 196 blocks of 256 threads
Test PASSED
Done
```

The sample successfully reported host-to-device copy, CUDA kernel launch, device-to-host copy, and result verification. Its log also contained `driverInitFileInfo`/`init` diagnostic messages of undetermined origin; verifying exit code `0` remains a follow-up check.

## Key lessons

- **Capacity vs allocation:** `Allocatable: nvidia.com/gpu: 1` can coexist with zero *unallocated* GPUs when one Pod has requested it.
- **GPU allocation vs utilization:** Idle compute cycles do not automatically make a whole reserved GPU schedulable to another Pod.
- **Device plugin:** The `nvidia.com/gpu` resource depends on successful device-plugin registration.
- **Failure boundaries:** Check host driver, Docker GPU access, Minikube device exposure, plugin readiness, node allocatable resources, and scheduler events in that order.
- **Sharing:** Helm packages Kubernetes applications; it does not enable GPU sharing. Features such as NVIDIA time-slicing or MPS require separate configuration.
- **Validation:** A Pod reaching `Running` confirms startup; a passing CUDA sample confirms actual GPU execution.

## Cleanup (optional)

```bash
kubectl --context=gpu-lab delete namespace day28-gpu
```

This deletes the exercise Pods and their namespace, not the `gpu-lab` cluster or the persistent host inotify setting.
