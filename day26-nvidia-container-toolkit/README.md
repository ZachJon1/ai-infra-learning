# Day 26 — NVIDIA Container Toolkit

**Status:** Completed. GPU visibility, PyTorch computation, and GPU-hiding experiments verified on my laptop.

## Objectives

- Explain the roles of the host NVIDIA driver, container runtime, CUDA libraries, and PyTorch.
- Verify GPU access and actual CUDA computation inside Docker.
- Select a GPU and diagnose failures from the host upward.

## Verified environment

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU |
| VRAM | 8,192 MiB |
| Host driver | 580.178.04 |
| Driver-reported CUDA level | 13.0 |
| NVIDIA runtime | Registered with Docker |
| Toolkit CLI | `/usr/bin/nvidia-ctk` |
| Container image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |
| PyTorch | `2.5.1+cu124` |
| Container CUDA build | 12.4 |

The toolkit was already configured, so no reinstallation was required. The CUDA level in `nvidia-smi` describes driver support; it does not identify the host's installed CUDA Toolkit. Containers can supply their own compatible CUDA libraries.

## Results from the completed session

| Experiment | Observed result |
| --- | --- |
| `docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi` | GPU visible inside Docker |
| PyTorch CUDA detection | `CUDA available: True` |
| 1,024 × 1,024 matrix multiplication, followed by synchronization | Completed; result on `cuda:0` |
| `NVIDIA_VISIBLE_DEVICES=void` | `CUDA available: False` |
| `NVIDIA_VISIBLE_DEVICES=0` | `CUDA available: True`, `GPUs: 1` |
| `NVIDIA_VISIBLE_DEVICES=none` | `CUDA available: False` |

These results came from commands executed on my laptop during the lesson. The accompanying scripts consolidate those commands for repeat runs.

## Repeat the lab

Prerequisites: working host NVIDIA driver, Docker access, and NVIDIA Container Toolkit configured for Docker. Run from this directory:

```bash
bash run-lab.sh
```

This checks host and container visibility, executes a CUDA matrix multiplication, and checks `void`, `none`, and GPU `0`. It starts disposable containers and does not change drivers or Docker configuration. The PyTorch image may require a substantial download if it is not already cached.

## GPU exposure

| Setting | GPU devices | Requested driver capabilities |
| --- | --- | --- |
| `all` | All available GPUs | Enabled |
| `0` | Host GPU 0 | Enabled |
| `none` | No GPUs | Enabled |
| `void` | No GPUs | Disabled |

Driver capabilities include `compute` and `utility`. PyTorch returning `False` for both `none` and `void` does not by itself demonstrate the difference in mounted driver components.

On a hypothetical four-GPU host, select host GPU 2 with either command:

```bash
docker run --rm --gpus '"device=2"' ubuntu nvidia-smi

docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=2 ubuntu nvidia-smi
```

Host GPU 2 is the third GPU. My laptop has only host GPU 0. A selected host GPU can appear as `cuda:0` inside the application; verify its UUID when identifying physical hardware. Selecting a device controls visibility but does not reserve it exclusively against other containers.

## Troubleshooting order

1. **Host:** Run `nvidia-smi`. If it fails, investigate the host driver and GPU first.
2. **Container:** Run the Docker `nvidia-smi` test. If the host works but this fails, inspect toolkit/runtime configuration, GPU selection, and permissions.
3. **Framework:** Inspect `torch.__version__`, `torch.version.cuda`, and `torch.cuda.is_available()`. Check for a CUDA-enabled build, compatible libraries/driver, and application visibility settings.
4. **Computation:** Execute an actual CUDA operation and synchronize. Investigate any runtime or memory errors.
5. **Service:** Load the model, warm it up, check readiness, send representative inference requests, and inspect latency, memory use, and logs.

**Correction retained from the interview:** A working host GPU does not establish GPU access inside containers. Successful container `nvidia-smi` does not establish working PyTorch computation.

Use fixed image versions for reproducibility. A digest provides stronger identity than a tag, which can be moved. This lab verifies GPU infrastructure; it does not measure inference performance or establish production readiness.

## Key concepts to retain

NVIDIA Container Toolkit exposes selected GPU devices and required host driver components to containers. The host driver must work first, while the container provides compatible CUDA libraries and a CUDA-enabled framework. Host visibility, container visibility, framework detection, and successful computation are separate checkpoints. Troubleshoot through host driver → container runtime → CUDA/PyTorch → inference service, and validate warmup, readiness, request handling, latency, and memory use before accepting production traffic.

**Next:** Day 27 — Kubernetes NVIDIA Device Plugin and `nvidia.com/gpu` resources.

## References

- [NVIDIA Container Toolkit: Docker configuration and GPU selection](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)
- [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [PyTorch CUDA availability API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html)
