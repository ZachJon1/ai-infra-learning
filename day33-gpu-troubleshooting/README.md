
# Day 33 — GPU Utilization Troubleshooting

## Objective

Diagnose GPU underutilization, identify performance bottlenecks, and evaluate optimizations using NVIDIA monitoring tools and PyTorch.

## Hardware and Environment

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8 GB)
- Driver: 580.178.04
- CUDA: 13.0
- PyTorch: 2.13.0+cu130
- Platform: Linux

## Experiment 1 — GPU Starvation

Compared paced GPU execution against continuous execution using 2048 × 2048 matrix multiplication.

| Metric | Paced | Continuous |
|---|---:|---:|
| Operations/sec | 3.96 | 477.32 |
| Average CUDA time | 1.827 ms | 2.070 ms |
| Peak allocated VRAM | 56.1 MiB | 56.1 MiB |

**Result:** 120.5× higher throughput when artificial waiting was removed.

**Finding:** Low GPU utilization can result from insufficient work being supplied to the GPU, rather than poor GPU computational performance.

## Experiment 2 — CPU-to-GPU Transfer Bottleneck

Compared repeated CPU-to-GPU tensor transfers against reusing GPU-resident tensors.

| Metric | Repeated transfers | GPU resident |
|---|---:|---:|
| Operations/sec | 131.43 | 469.47 |
| Average kernel time | 2.075 ms | 2.105 ms |
| Peak allocated VRAM | 56.1 MiB | 56.1 MiB |

**Result:** 3.57× higher throughput after removing unnecessary repeated transfers.

**Finding:** Similar GPU kernel execution times can coexist with substantially different application throughput due to overhead outside the kernel.

## Reproduce the Experiments

Activate the existing Python environment:

```bash
activate-ai
```

Run the GPU starvation experiment:

```bash
python gpu_experiment.py paced
python gpu_experiment.py continuous
```

Run the transfer bottleneck experiment:

```bash
python transfer_bottleneck.py copy-each
python transfer_bottleneck.py resident
```

Monitor GPU activity separately:

```bash
nvidia-smi -l 1
```

## Important Observations

- GPU utilization measures sampled kernel activity, not peak computational efficiency.
- VRAM allocation remained constant despite large changes in throughput.
- CPU preprocessing, transfer overhead, synchronization, and insufficient work can cause GPU underutilization.
- During sustained computation, the GPU reached approximately 99% utilization and 125 W power draw.
- NVIDIA memory utilization repeatedly reported 100% even during idle periods. This remains an unresolved telemetry observation, not confirmed memory-bandwidth saturation.

## Limitations

The benchmarks use synthetic matrix multiplication, not a production inference service.

The GPU-resident optimization assumes reusable inputs; different inference requests still require correct data transfers.

Results represent measurements on one RTX 3070 Laptop GPU and should not be generalized to other hardware without testing.

## Documentation

See [Day 33 Lab Report](Day33_GPU_Utilization_Troubleshooting_Lab_Report.md) for detailed experimental findings, troubleshooting methodology, and interview notes.

## Key Takeaway

Effective GPU troubleshooting requires identifying the limiting resource, measuring application-level performance, isolating one variable, and verifying the improvement.

The objective is not simply maximum GPU utilization, but reliable throughput, latency, and resource efficiency.
