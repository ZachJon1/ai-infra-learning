# Day 2 — GPU Benchmarking, PyTorch CUDA & VRAM

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Verify that PyTorch can access the GPU.
- Understand synchronization when measuring GPU execution time.
- Inspect GPU utilization and VRAM consumption.
- Build intuition for CPU/GPU execution differences.

## Core topics
- CUDA availability in PyTorch
- Moving tensors/models between CPU and GPU
- GPU synchronization
- GPU utilization
- VRAM allocation
- Benchmarking methodology

## Commands / checks to retain
```bash
nvidia-smi
```

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

For reliable GPU timing, synchronize around the region being measured:

```python
torch.cuda.synchronize()
# start timing
# GPU work
torch.cuda.synchronize()
# stop timing
```

## AI-infrastructure relevance
GPU infrastructure work requires distinguishing model/application bottlenecks from GPU availability, memory, utilization, and synchronization effects.

## Interview takeaway
GPU work is asynchronous. Timing GPU code without synchronization can produce misleading measurements, and utilization alone does not explain whether a workload is compute-, memory-, or input-bound.
