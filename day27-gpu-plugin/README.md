# Day 27 — Kubernetes NVIDIA Device Plugin

## Objective

Configure Kubernetes to discover and allocate NVIDIA GPUs, deploy a CUDA workload, investigate device-plugin failures, and demonstrate GPU resource contention.

## Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 series, 8 GB |
| NVIDIA driver | 580.178.04 |
| Driver-supported CUDA version | 13.0 |
| Minikube | v1.38.1 |
| Kubernetes | v1.35.1 |
| Container runtime | Docker |
| Cluster profile | gpu-lab |
| GPU resource | nvidia.com/gpu |

## Architecture

The NVIDIA Device Plugin runs on eligible GPU nodes and communicates with their kubelets. It discovers available devices, reports device health, and registers GPU resources with Kubernetes.

The Kubernetes scheduler considers advertised GPU resources when placing workloads. After scheduling, the kubelet coordinates device allocation, while the NVIDIA container runtime enables access to the assigned GPU.

## Cluster Configuration

Created a separate GPU-enabled Minikube profile with the Docker driver and `--gpus=all`. Minikube automatically enabled the NVIDIA Device Plugin addon.

Initially, the device plugin entered `CrashLoopBackOff`, and Kubernetes did not advertise any `nvidia.com/gpu` resources.

## Troubleshooting: Device Plugin Failure

**Observed error:**

`failed to create FS watcher for /var/lib/kubelet/device-plugins/: too many open files`

Several system components also experienced failures, including kube-proxy and the storage provisioner.

Inspection of the host revealed relatively low inotify settings:

| Setting | Initial value | Updated value |
|---|---:|---:|
| max_user_instances | 128 | 8192 |
| max_user_watches | 65536 | 524288 |

Applied temporary Linux kernel configuration changes using `sysctl` and restarted the NVIDIA Device Plugin DaemonSet.

The plugin subsequently registered `nvidia.com/gpu` successfully with the kubelet.

**Verified resources:**

| Resource | Capacity | Allocatable |
|---|---:|---:|
| nvidia.com/gpu | 1 | 1 |

## CUDA Workload Validation

Deployed a Kubernetes Pod using NVIDIA's CUDA vector-addition sample with `nvidia.com/gpu: 1`.

The workload performed vector addition on 50,000 elements.

**Observed output:**

`CUDA kernel launch with 196 blocks of 256 threads`

`Test PASSED`

`Done`

The Pod completed successfully with exit code 0 and no container restarts.

## GPU Resource Contention Experiment

Created two Pods, `gpu-holder` and `gpu-contender`, each requesting one GPU.

While `gpu-holder` was running, `gpu-contender` remained Pending.

Kubernetes reported:

`0/1 nodes are available: 1 Insufficient nvidia.com/gpu`

After deleting `gpu-holder`, Kubernetes scheduled `gpu-contender`, which transitioned through `ContainerCreating` to `Running`.

This confirmed that Kubernetes prevents overlapping whole-GPU allocations under the current device-plugin configuration.

## Key Lessons

The NVIDIA Device Plugin integrates GPU resources into Kubernetes through node-local discovery and kubelet registration. A DaemonSet provides a plugin instance on each eligible node. GPU Capacity and Allocatable describe node resource availability, while the scheduler separately accounts for existing Pod requests.

GPU allocation failures can originate from multiple layers, including Linux kernel resource limits, the container runtime, the device plugin, and Kubernetes scheduling.

Successful troubleshooting requires examining the complete infrastructure stack rather than assuming every GPU-related failure originates from CUDA or GPU hardware.

## Outcome

Successfully configured a GPU-enabled Kubernetes lab, resolved a device-plugin registration failure, executed CUDA computations, and demonstrated resource exhaustion and scheduling recovery.
