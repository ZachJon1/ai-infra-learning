# Day 40 — GPU Inference Batching: Benchmark and Engineering Notes

**Roadmap:** AI Infrastructure & ML Systems (Day 40/220)  
**Date:** October 23, 2026  
**Environment:** PyTorch with CUDA (`Device: cuda`)  
**Model:** Small untrained CNN (Conv2d → ReLU → Conv2d → ReLU → AdaptiveAvgPool2d → Flatten → Linear); input image shape `[3,224,224]`.

## Objective

Compare sequential per-input inference against batched inference, then evaluate throughput, forward-pass latency, and PyTorch allocated GPU memory for batch sizes 1, 4, 8, and 16.

## Experiment 1 — Sequential vs. batched inference

The program processed 400 inputs per mode. Inputs resided on the GPU; the benchmark warmed up both paths and synchronized CUDA around each full benchmark loop.

| Metric | Sequential (8 separate forwards per group) | Batched (one forward per group of 8) |
|---|---:|---:|
| Inputs processed | 400 | 400 |
| Wall-clock time | 0.0672 s | 0.0106 s |
| Throughput | 5,952.95 inputs/s | 37,616.01 inputs/s |
| Relative speedup | Baseline | **6.32×** |

**Correctness check:** Outputs had shape `torch.Size([8, 10])`; `torch.allclose` reported `True`. Numerical closeness does not assert identical bitwise outputs.

## Experiment 2 — Batch-size sweep

The batch-size sweep used 100 timed forwards per batch size, with a CUDA synchronization after each forward. Reported latencies include Python dispatch/synchronization overhead; they exclude HTTP routing, preprocessing, queueing, and network delivery. Peak memory is PyTorch allocated memory (not total device VRAM usage).

| Batch size | Average execution latency (ms) | Throughput (inputs/s) | Peak allocated memory (MiB) |
|---:|---:|---:|---:|
| 1 | 0.184 | 5,449.11 | 10.25 |
| 4 | 0.192 | 20,880.93 | 16.57 |
| 8 | 0.274 | 29,160.65 | 24.99 |
| 16 | 0.463 | 34,539.17 | 41.84 |

### Observations

- Batch 16 had the highest measured throughput in this sweep.
- Throughput increased approximately **18.4%** from batch 8 to batch 16, while peak PyTorch allocated memory increased approximately **67.4%** and execution latency rose approximately **69.0%**.
- Batch 8 achieved about **84.4%** of batch 16 throughput with approximately **59.7%** of its peak allocated memory.
- Diminishing throughput returns mean the largest batch is not automatically the best production configuration.
- A hypothetical steady demand of 32,000 inputs/s represents approximately **92.6%** of measured batch-16 execution throughput, leaving only **7.4%** nominal headroom **if** that throughput transferred to a real service; such a transfer has not been established.

## Troubleshooting and design notes

1. **Out of memory:** Reduce batch size, monitor memory at several batch sizes, choose hardware with sufficient VRAM, and consider appropriate lower precision or model optimization. Inspect both PyTorch allocation and device-wide usage.
2. **Input layout:** A model expecting `[N, C, H, W]` must receive correctly ordered dimensions. Convert `[N, H, W, C]` with `x.permute(0,3,1,2).contiguous()` rather than using an arbitrary `reshape`.
3. **Queueing latency:** Waiting to fill a batch can hurt sparse-traffic latency. Dynamic batching should cap both batch size and wait time (Day 41).
4. **Production choice:** Test end-to-end throughput, p50/p95/p99 latency, memory safety, concurrent load, and capacity headroom before selecting the production batch size.

## Benchmark limitations and follow-up

- Small untrained demonstration CNN; conclusions may differ for real vision and LLM serving workloads.
- No real HTTP or Kubernetes service benchmark yet; no p95 or p99 request latency collected.
- Benchmark runs were not repeated across trials with variance/error bars.
- CUDA hardware identity, software versions, and GPU-wide utilization were not captured in this report.
- Next: Day 41 dynamic batching with a maximum queueing delay, followed by end-to-end latency-versus-throughput analysis on Day 42.

## Files / commands in the local lab

```bash
cd ~/ai-infra/day40-batching
python batching.py
python batching_sweep.py
nvidia-smi
```

Keep this note alongside the scripts and add device/software metadata, reproducible benchmark settings, and service-level measurements when available. **No GitHub commit or push is performed by this report.**

## Key details to retain

Batching combines compatible inputs into a larger tensor so a model can execute them in fewer forward passes, often improving GPU parallelism and amortizing overhead. In this CUDA experiment, batches provided a 6.32× throughput speedup compared with individual execution; increasing batch size further improved throughput but with diminishing returns, increased per-batch execution latency, and greater allocated GPU memory. Select a production batch size using service-level latency SLOs, demand, memory safety, and spare capacity—not maximum throughput alone—and use dynamic batching to avoid excessive waiting under light traffic.
