# Day 21 — GPU Architecture Refresher

## Objectives

This session focused on understanding how GPU hardware executes AI workloads and how GPU architecture affects AI infrastructure decisions.

Topics covered:

- Streaming Multiprocessors (SMs)
- CUDA cores
- Tensor Cores
- Threads, warps, blocks, and grids
- GPU memory hierarchy
- VRAM capacity vs. memory bandwidth
- FP32, FP16, BF16, and TF32
- GPU occupancy and latency hiding
- Compute-bound vs. memory-bandwidth-bound workloads
- PyTorch CUDA memory allocation and caching
- GPU troubleshooting fundamentals

---

## GPU Execution Model

A simplified execution path is:

```text
PyTorch operation
      ↓
CUDA library/runtime
      ↓
CUDA kernel
      ↓
Grid
      ↓
Thread blocks
      ↓
Blocks scheduled onto SMs
      ↓
Threads grouped into 32-thread warps
      ↓
Warp scheduler
      ↓
CUDA Cores / Tensor Cores
```

### Streaming Multiprocessor

An NVIDIA GPU contains multiple Streaming Multiprocessors (SMs).

An SM contains resources such as:

- CUDA cores
- Tensor Cores
- Registers
- Shared memory
- Warp schedulers

The SM schedules groups of threads called warps onto its execution resources.

---

## Threads and Warps

CUDA work is organized as:

```text
Kernel
  ↓
Grid
  ↓
Thread Blocks
  ↓
Threads
```

Threads are scheduled in groups of:

```text
32 threads = 1 warp
```

For example:

```text
512 threads / 32 threads per warp = 16 warps
```

Having multiple warps available allows the GPU to hide latency. If one warp is waiting for memory, the SM can execute another ready warp.

### Warp Divergence

When threads within the same warp take different execution branches, execution can become less efficient.

```text
if condition:
    branch A
else:
    branch B
```

If some threads execute A while others execute B, the warp may need to execute the branches separately.

---

## CUDA Cores vs. Tensor Cores

### CUDA Cores

CUDA cores are general-purpose arithmetic execution units used for operations such as:

- Floating-point arithmetic
- Integer arithmetic
- Addition
- Multiplication

### Tensor Cores

Tensor Cores are specialized hardware designed for high-throughput matrix operations.

They are especially important for:

- Neural-network training
- Neural-network inference
- Matrix multiplication
- Transformer workloads
- LLMs

Tensor Cores support lower-precision numerical formats such as FP16, BF16, and TF32, depending on GPU architecture.

---

## Numerical Precision and VRAM

Approximate storage requirements:

```text
FP32 = 4 bytes/value
FP16 = 2 bytes/value
BF16 = 2 bytes/value
```

For a 10-billion-parameter model:

```text
FP32:
10B × 4 bytes ≈ 40 GB

FP16:
10B × 2 bytes ≈ 20 GB
```

Lower precision can therefore significantly reduce VRAM requirements while also enabling higher Tensor Core throughput.

---

## GPU Memory Hierarchy

Simplified hierarchy:

```text
Registers
   ↓
Shared Memory / L1
   ↓
L2 Cache
   ↓
VRAM
   ↓
CPU RAM
   ↓
Storage
```

Resources closer to the execution units are generally faster but smaller.

---

## What Consumes VRAM?

Training workloads may require memory for:

- Model weights
- Activations
- Gradients
- Optimizer state
- Temporary buffers
- CUDA/framework runtime state

LLM inference commonly requires:

- Model weights
- Activations
- KV cache
- Temporary buffers
- Runtime/framework overhead

A model fitting in VRAM based only on parameter size does not guarantee that the complete workload will fit.

---

## VRAM Capacity vs. Memory Bandwidth

### Capacity

VRAM capacity determines:

> How much data can fit on the GPU.

Examples:

```text
8 GB
24 GB
48 GB
80 GB
```

### Bandwidth

Memory bandwidth determines:

> How quickly data can move between VRAM and GPU compute resources.

High VRAM usage does not mean that a workload is memory-bandwidth bound.

---

## Compute-Bound vs. Memory-Bandwidth-Bound

### Compute-bound

The arithmetic execution resources are the limiting factor.

```text
Data arrives fast enough
        ↓
Compute resources stay busy
        ↓
More compute throughput could improve performance
```

### Memory-bandwidth-bound

The execution units could perform additional work but are waiting for data.

```text
Compute ready
     ↓
Waiting for memory
     ↓
Memory bandwidth limits performance
```

High VRAM capacity utilization alone does not prove a memory-bandwidth bottleneck.

---

## Arithmetic Intensity

Arithmetic intensity describes the amount of computation performed relative to memory traffic.

```text
High arithmetic intensity
→ more computation per byte transferred
→ potentially compute-bound

Low arithmetic intensity
→ little computation per byte transferred
→ potentially memory-bandwidth-bound
```

---

## GPU Occupancy

Occupancy roughly describes the number of active warps on an SM relative to the maximum number the SM can support.

High resource consumption per block, such as excessive register or shared-memory use, can reduce the number of blocks and warps that fit on an SM.

However:

```text
100% occupancy ≠ maximum performance
```

Warps may still be stalled due to:

- Memory access
- Synchronization
- Warp divergence
- Poor instruction efficiency

---

# Hands-On GPU Lab

## Test System

GPU detected:

```text
NVIDIA GeForce RTX 3070 Laptop GPU
```

GPU architecture:

```text
GA104M
```

Compute capability:

```text
8.6
```

Physical VRAM reported by `nvidia-smi`:

```text
8192 MiB
```

CUDA-visible memory reported by PyTorch:

```text
~7.65 GiB
```

NVIDIA driver:

```text
580.178.04
```

Driver-reported CUDA compatibility:

```text
CUDA 13.0
```

PyTorch CUDA build:

```text
13.0
```

---

## GPU Detection

Commands:

```bash
nvidia-smi
nvidia-smi -L
lspci | grep -i nvidia
```

PyTorch:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.version.cuda)
```

Observed:

```text
CUDA available: True
GPU count: 1
GPU: NVIDIA GeForce RTX 3070 Laptop GPU
Compute capability: (8, 6)
```

---

## FP32 Allocation Experiment

Created:

```python
x = torch.zeros((5000, 5000), device="cuda")
```

Number of elements:

```text
25,000,000
```

FP32 uses:

```text
4 bytes per element
```

Expected size:

```text
25,000,000 × 4
= 100,000,000 bytes
≈ 95.37 MiB
```

Observed:

```text
Tensor size:        95.37 MiB
PyTorch allocated:  96 MiB
PyTorch reserved:   96 MiB
```

---

## FP16 Allocation Experiment

Created:

```python
y = torch.zeros(
    (5000, 5000),
    dtype=torch.float16,
    device="cuda"
)
```

FP16 uses:

```text
2 bytes per element
```

Observed:

```text
FP16 tensor:        47.68 MiB
Total allocated:   144 MiB
Total reserved:    144 MiB
```

This demonstrated approximately:

```text
FP16 memory ≈ 1/2 FP32 memory
```

for the same number of values.

---

## PyTorch vs. `nvidia-smi` Memory

With both tensors allocated:

```text
PyTorch allocated: 144 MiB
PyTorch reserved:  144 MiB
```

However:

```text
nvidia-smi
python3 GPU memory: 304 MiB
```

The values differ because `nvidia-smi` observes the broader CUDA process memory footprint, including resources outside live PyTorch tensor allocations.

Examples include:

- CUDA context
- CUDA runtime/libraries
- Driver allocations
- Temporary resources
- PyTorch allocations

Therefore:

```text
torch.cuda.memory_allocated()
        ≠
total process memory shown by nvidia-smi
```

---

## PyTorch Caching Allocator

After deleting the tensors:

```python
del x
del y
```

Observed:

```text
Allocated: 0 MiB
Reserved: 144 MiB
```

The tensors were no longer live, but PyTorch retained the memory in its caching allocator for future reuse.

After:

```python
torch.cuda.empty_cache()
```

Observed:

```text
Allocated: 0 MiB
Reserved:  0 MiB
```

However, `nvidia-smi` still showed:

```text
python3: ~160 MiB
```

This demonstrates that clearing the PyTorch cache does not destroy the CUDA context or all process-level GPU resources.

---

## Important Troubleshooting Distinctions

### High VRAM usage does not mean high compute utilization

Example:

```text
VRAM usage:       95%
GPU utilization:  20%
```

This means:

```text
Most memory capacity is occupied
```

while:

```text
GPU compute resources are relatively underutilized
```

It does **not** prove that the workload is memory-bandwidth bound.

Additional metrics are required.

---

## Key Takeaways

A GPU contains many Streaming Multiprocessors containing CUDA cores, Tensor Cores, registers, shared memory, and warp schedulers. CUDA workloads are launched as kernels containing grids, thread blocks, and threads, with threads executed in groups of 32 called warps. Tensor Cores accelerate matrix-heavy AI workloads, while numerical precision affects both performance and VRAM requirements. VRAM capacity determines how much data can fit, while bandwidth determines how quickly data can reach the compute resources. GPU memory may contain weights, activations, gradients, optimizer state, KV cache, temporary buffers, runtime state, and allocator caches. High VRAM usage does not imply high GPU utilization or a memory-bandwidth bottleneck. Effective GPU troubleshooting therefore requires examining compute utilization, memory bandwidth, VRAM allocation, CPU and I/O bottlenecks, transfers, batching, and the complete software/hardware stack.
