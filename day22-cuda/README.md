# Day 22 — CUDA Runtime, Drivers, Toolkit & Compatibility

## Objective

Understand the different layers of the NVIDIA CUDA software stack and learn how to diagnose compatibility problems between:

- NVIDIA GPU drivers
- CUDA Toolkit
- CUDA Runtime
- CUDA Driver API
- PyTorch CUDA builds
- Containers
- Physical GPUs

The main goal was to stop treating "CUDA" as a single component and instead understand it as a layered software stack.

---

## My Current GPU Stack

Hardware and software detected on my system:

```text
GPU: NVIDIA GeForce RTX 3070 Laptop GPU
VRAM: 8 GB

NVIDIA Driver: 580.178.04
Driver-supported CUDA version: 13.0

CUDA Toolkit: 13.0
nvcc: 13.0.88

PyTorch: 2.13.0+cu130
PyTorch CUDA build: 13.0

CUDA available in PyTorch: True
GPU count: 1
```

Commands used:

```bash
nvidia-smi

nvcc --version

which nvcc

ls -l /usr/local | grep cuda

ldconfig -p | grep libcuda
```

PyTorch verification:

```bash
python3 - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## CUDA Software Stack

A simplified view of the GPU software stack:

```text
Application
    │
    ▼
PyTorch / CUDA Application
    │
    ▼
CUDA Runtime / CUDA Libraries
    │
    ▼
CUDA Driver API (libcuda.so)
    │
    ▼
NVIDIA Driver
    │
    ▼
Physical GPU
```

The CUDA Toolkit exists mainly on the development side:

```text
CUDA Toolkit
├── nvcc
├── CUDA headers
├── runtime libraries
├── development libraries
├── debugging tools
└── profiling tools
```

---

## NVIDIA Driver

The NVIDIA driver provides the host operating system with access to the GPU.

On my system:

```text
Driver Version: 580.178.04
```

`nvidia-smi` reports:

```text
CUDA Version: 13.0
```

This does **not** mean that CUDA Toolkit 13.0 must be installed.

It indicates the CUDA level supported by the installed NVIDIA driver.

---

## CUDA Toolkit and `nvcc`

The CUDA Toolkit provides development tools used to build CUDA applications.

My installed Toolkit:

```text
CUDA Toolkit: 13.0
nvcc: 13.0.88
```

Location:

```text
/usr/local/cuda-13.0/bin/nvcc
```

`nvcc --version` therefore answers a different question from `nvidia-smi`.

```text
nvidia-smi
    → NVIDIA driver and driver-supported CUDA level

nvcc --version
    → installed CUDA Toolkit/compiler
```

These versions do not necessarily have to match.

---

## `libcudart.so` vs `libcuda.so`

One important distinction:

### `libcudart.so`

The CUDA Runtime library.

On my system:

```text
/usr/local/cuda/.../libcudart.so.13
```

It belongs to the CUDA runtime/toolkit side.

### `libcuda.so`

The CUDA Driver API interface.

On my system:

```text
/lib/x86_64-linux-gnu/libcuda.so
```

It is provided by the NVIDIA driver side.

Simplified relationship:

```text
Application
    │
    ▼
libcudart.so
CUDA Runtime
    │
    ▼
libcuda.so
CUDA Driver API
    │
    ▼
NVIDIA Driver
    │
    ▼
GPU
```

---

## PyTorch CUDA Version

My PyTorch installation reports:

```text
PyTorch: 2.13.0+cu130
torch.version.cuda = 13.0
```

The `cu130` suffix indicates that the PyTorch build targets CUDA 13.0.

This is different from both:

```text
nvcc --version
```

and:

```text
nvidia-smi
```

Therefore, three CUDA versions may appear on a system:

```text
nvidia-smi
    → driver-supported CUDA level

nvcc
    → locally installed CUDA Toolkit

torch.version.cuda
    → CUDA version targeted by the PyTorch build
```

These versions do not need to be identical for the system to work.

---

## PyTorch Does Not Always Need the System CUDA Toolkit

A prebuilt CUDA-enabled PyTorch package can contain the CUDA user-space libraries it needs.

Therefore, this is possible:

```text
nvcc: command not found
```

while:

```python
torch.cuda.is_available()
```

returns:

```text
True
```

This means the machine may be capable of **running CUDA-enabled PyTorch applications** even though the CUDA compiler is not installed.

The CUDA Toolkit and `nvcc` become important when compiling CUDA applications or custom CUDA extensions.

---

## CUDA Compatibility

The important compatibility relationship is usually:

```text
CUDA application/runtime
        │
        ▼
Compatible NVIDIA Driver
        │
        ▼
GPU
```

A newer NVIDIA driver can generally support applications built against older CUDA releases.

Example:

```text
Driver: CUDA 13 capable
PyTorch: CUDA 12.8
```

This can be completely valid.

The more problematic case is:

```text
New CUDA runtime
        │
        ▼
Old NVIDIA driver
        │
        ✕
        ▼
GPU
```

This may result in errors such as:

```text
CUDA driver version is insufficient for CUDA runtime version
```

The host NVIDIA driver should be one of the first components investigated.

---

## Containers and CUDA

A CUDA-enabled container may contain:

```text
PyTorch
CUDA runtime
cuBLAS
cuDNN
other CUDA libraries
```

but it normally relies on the **host NVIDIA driver** to access the physical GPU.

```text
Container
│
├── PyTorch
├── CUDA runtime
├── cuBLAS
└── cuDNN
      │
      ▼
GPU integration
      │
      ▼
Host libcuda.so
      │
      ▼
Host NVIDIA Driver
      │
      ▼
GPU
```

Therefore:

```text
CUDA libraries inside container ≠ automatic GPU access
```

The GPU must be explicitly exposed to the container through the appropriate NVIDIA container/runtime integration.

---

## Running vs Compiling CUDA

Another important distinction:

```text
Running CUDA code
        ≠
Compiling CUDA code
```

A machine may run a CUDA-enabled application successfully without `nvcc`.

`nvcc` is primarily required when CUDA code needs to be compiled.

---

## GPU Memory vs GPU Utilization

My `nvidia-smi` output showed approximately:

```text
Memory Usage: 310 MiB / 8192 MiB
GPU Utilization: 0%
```

This is not contradictory.

GPU memory may be allocated while no computation is currently occurring.

For example:

```text
Model loaded into VRAM
        │
        ▼
No requests arriving
        │
        ▼
High/allocated VRAM
GPU utilization = 0%
```

Therefore:

```text
VRAM usage ≠ GPU compute utilization
```

---

## Troubleshooting Strategy

Instead of immediately assuming the physical GPU is broken, diagnose each layer:

```text
Application
    ↓
PyTorch
    ↓
CUDA Runtime / Libraries
    ↓
CUDA Driver API
    ↓
NVIDIA Driver
    ↓
GPU
```

Useful checks include:

```bash
nvidia-smi
```

Check whether the host driver and GPU are visible.

```bash
nvcc --version
```

Check the installed CUDA Toolkit/compiler.

```bash
ldconfig -p | grep libcuda
```

Check CUDA driver/runtime libraries.

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

Check whether PyTorch can successfully access the GPU.

---

## Example Failure Scenarios

### `nvcc` not found but PyTorch works

```text
nvidia-smi                  ✅
torch.cuda.is_available()   True
nvcc                        command not found
```

Likely explanation:

The GPU driver works and PyTorch has the runtime libraries it needs, but the CUDA Toolkit/compiler is missing or not in `PATH`.

---

### New CUDA runtime with old driver

```text
CUDA application: 13.x
NVIDIA Driver: 535
```

Possible result:

```text
CUDA driver version is insufficient for CUDA runtime version
```

First investigation:

```text
Host NVIDIA driver compatibility
```

---

### CUDA container cannot see GPU

Host:

```text
nvidia-smi works
```

Container:

```text
torch.cuda.is_available() == False
```

Likely issue:

```text
Container GPU access / NVIDIA container runtime configuration
```

The host GPU and driver must be correctly exposed to the container.

---

## Key Takeaways

CUDA should be treated as a **software stack rather than a single installation**. `nvidia-smi` reports information about the NVIDIA driver and the CUDA level supported by that driver, `nvcc --version` reports the locally installed CUDA Toolkit/compiler, and `torch.version.cuda` reports the CUDA version targeted by the PyTorch build. These versions do not need to match exactly. `libcudart.so` belongs to the CUDA runtime side, while `libcuda.so` exposes the CUDA Driver API and comes from the NVIDIA driver stack. Prebuilt CUDA-enabled applications such as PyTorch can run without a system CUDA Toolkit or `nvcc`, but they still require a compatible NVIDIA driver and access to the physical GPU. For containers, CUDA libraries inside the image are not sufficient—the container must be given access to the host GPU and NVIDIA driver. GPU problems should therefore be diagnosed layer by layer instead of treating every failure as a generic "CUDA problem."

---

## Next

**Day 23 — GPU Monitoring with `nvidia-smi`**

Topics will include:

- GPU utilization
- VRAM utilization
- temperature
- power usage
- performance states
- running processes
- continuous monitoring
- identifying idle vs saturated GPUs
- diagnosing GPU bottlenecks
