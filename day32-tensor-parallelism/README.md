# Day 32 — Model Parallelism & Tensor Parallelism

## Overview

This lab explores model parallelism, tensor parallelism,
pipeline parallelism, and GPU communication in AI systems.

The practical experiments use PyTorch to demonstrate how
linear layers can be partitioned while preserving numerical
correctness.

## Learning Objectives

- Distinguish data parallelism from model parallelism.
- Explain tensor and pipeline parallelism.
- Implement column-wise and row-wise weight partitioning.
- Reconstruct partial outputs correctly.
- Demonstrate communication-efficient consecutive layers.
- Understand GPU communication and scaling trade-offs.

## 1. Parallelism Strategies

### Data Parallelism

Each GPU holds a copy of the model and processes different
training batches. Gradients are synchronized across workers.

Primary benefit: increased training throughput.

### Tensor Parallelism

Individual model layers are partitioned across GPUs.

**Column parallelism**

For Y = XW, split W along its output-feature dimension.

Each partition computes a different subset of output features.
The complete output can be reconstructed by concatenation.

**Row parallelism**

Partition W along its input-feature dimension and partition
X correspondingly.

Each partition computes a contribution to the same output.
The complete result is reconstructed by summation.

### Pipeline Parallelism

Groups of complete layers are assigned to different GPUs.

Microbatching allows pipeline stages to execute concurrently,
reducing the fraction of time lost to pipeline bubbles.

## 2. Hands-on Experiment

File: `tensor_parallel_demo.py`

### Requirements

- Python 3
- PyTorch
- CUDA-capable GPU (optional; CPU fallback supported)

### Run

```bash
python tensor_parallel_demo.py
```

### Experiment A: Column Parallelism

Input shape: (2, 4)
Weight shape: (4, 8)

Weight partitions:
- GPU partition 0: (4, 4)
- GPU partition 1: (4, 4)

Partial output shapes: (2, 4) each

Reconstruction: torch.cat(..., dim=1)

Observed result: Matches full output = True

### Experiment B: Row Parallelism

Weight partitions:
- Partition 0: (2, 8)
- Partition 1: (2, 8)

Input partitions: (2, 2) each
Partial output shapes: (2, 8) each

Reconstruction: element-wise addition

Observed result: Matches full output = True

### Experiment C: Consecutive Tensor-Parallel Layers

Two linear transformations were evaluated:

1. Column-parallel first layer.
2. Row-parallel second layer.

The intermediate partitions were passed directly into the
corresponding second-layer partitions.

This avoids reconstructing the full intermediate tensor.

Final output shape: (2, 4)

Observed results:

- Full result shape: torch.Size([2, 4])
- Parallel result shape: torch.Size([2, 4])
- Results match: True

## 3. GPU Communication

**AllGather:** Collects distributed tensor partitions and
makes the combined result available across participating GPUs.

**AllReduce (SUM):** Combines corresponding partial results
using element-wise reduction.

**NCCL:** Provides GPU collective communication operations
used in distributed training and inference systems.

High-bandwidth GPU interconnects can reduce communication
overhead for tensor-parallel workloads.

## 4. Infrastructure Engineering Considerations

- Tensor parallelism reduces per-GPU weight memory but
  introduces communication and synchronization costs.
- Pipeline parallelism introduces pipeline bubbles and
  requires inter-stage activation transfers.
- NVLink-connected GPUs are often favorable for
  communication-intensive tensor parallelism.
- Combining TP within nodes and PP across nodes may reduce
  expensive inter-node collective communication.
- Increasing GPU count does not guarantee lower latency
  or higher throughput.

Important performance metrics include:
- Inference latency (p50/p95)
- Throughput (requests or tokens per second)
- GPU compute utilization
- GPU VRAM usage
- Collective communication duration
- Inter-node network traffic

## 5. Experimental Limitations

These experiments were executed using CUDA on a single GPU.

Both simulated partitions reside on the same device.
The experiments validate numerical correctness, not
distributed performance.

No real multi-GPU NCCL communication, inter-node execution,
or distributed speedup was measured.

## Key Takeaway

Efficient model parallelism requires balancing model memory,
GPU computation, collective communication, and network
topology. The objective is not merely distributing work
across more GPUs, but improving measurable system performance.
