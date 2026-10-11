# Day 33 — GPU Utilization Troubleshooting

**Study track:** AI Infrastructure / ML Systems, Phase 2 (GPU Infrastructure)  
**Date:** October 10, 2026  
**Hardware:** NVIDIA GeForce RTX 3070 Laptop GPU, 8 GiB VRAM  
**Software observed:** NVIDIA driver 580.178.04; CUDA 13.0; PyTorch 2.13.0+cu130; one CUDA GPU available  
**Lab directory (user machine):** `~/ai-infra/day33-gpu-troubleshooting/`

## Investigation objective

Distinguish GPU busy-time utilization from computational efficiency, identify host-side causes of low throughput, and evaluate controlled fixes using synchronized GPU timing and application-level throughput.

## Experiment A — GPU starvation (paced vs continuous)

The same 2048 × 2048 GPU matrix multiplication was repeatedly executed in two modes. The *paced* version slept for 250 milliseconds after each operation; the *continuous* version did not.

| Metric | Paced | Continuous |
|---|---:|---:|
| Total operations | 80 | 9,547 |
| Wall-clock elapsed | 20.20 s | 20.00 s |
| Throughput | 3.96 operations/s | 477.32 operations/s |
| Average CUDA event time | 1.827 ms | 2.070 ms |
| Total CUDA event time | 0.15 s | 19.77 s |
| Peak PyTorch VRAM allocation | 56.1 MiB | 56.1 MiB |

**Result:** Approximately **120.5× more operations/s** without deliberate pacing. The principal improvement came from reducing time spent waiting for more work, not from accelerating each matrix multiplication. Identical tensor allocation did not imply identical GPU activity.

## Experiment B — Repeated CPU-to-GPU copies (copy-each vs resident)

CPU-resident input tensors were either copied into preallocated GPU buffers on every iteration (*copy-each*) or kept on the GPU and reused without repeating copies (*resident*). Both used a 2048 × 2048 GPU matrix multiplication.

| Metric | Copy each iteration | GPU-resident inputs |
|---|---:|---:|
| Total operations | 1,578 | 5,634 |
| Wall-clock elapsed | 12.01 s | 12.00 s |
| Throughput | 131.43 operations/s | 469.47 operations/s |
| Average matrix multiplication CUDA event time | 2.075 ms | 2.105 ms |
| Total matrix multiplication CUDA event time | 3.27 s | 11.86 s |
| Peak PyTorch VRAM allocation | 56.1 MiB | 56.1 MiB |

**Result:** Approximately **3.57× more operations/s** when unnecessary CPU-to-GPU copies were removed. The similar GPU kernel times, together with different overall throughput, point to overhead outside the measured kernel. The 8.74 seconds of copy-each time not captured by the matrix-multiplication CUDA events includes transfers, host-side overhead, synchronization, and other execution costs; it was **not** measured as exclusively transfer time.

**Limitation:** Reusing a GPU-resident input is only correct when its contents remain valid across operations. Real inference requests with different inputs still require correct data movement.

## Observed GPU telemetry

- At rest: `nvidia-smi` commonly reported approximately 0% GPU utilization, around 339–360 MiB VRAM allocated, roughly 20–22 W, and performance state P8. Desktop/graphics processes also consumed memory.
- During sustained continuous computation: approximately 99% GPU utilization, 125 W, P0, increasing temperature to about 75°C, and SM clock roughly 1335 to 1290 MHz across the active interval.
- Anomalous/unresolved observation: `utilization.memory` repeatedly reported 100% even during otherwise idle periods. This field is an activity metric, **not VRAM percent full**. Do not infer saturated memory bandwidth from that reading alone; corroborate with alternative monitoring/profiling and investigate reporting/driver issues if it persists.
- Sustained work approached the reported 125 W power cap. Clocks and temperature alone do not prove harmful throttling; inspect GPU clock-event reasons and workload characteristics if throughput degrades.

## Troubleshooting runbook

1. **Establish the service objective:** required throughput, P95 latency, cost, and workload characteristics.
2. **Collect baselines:** `nvidia-smi` GPU utilization, VRAM allocated, memory activity, power, temperature, clock, performance state; CPU per-thread utilization; disk/network loading; application latency and throughput.
3. **Locate waiting vs execution:** time preprocessing, data loading, CPU-to-GPU transfers, kernel execution, postprocessing and synchronization separately. Remember that CUDA is asynchronous; synchronize or use CUDA events as appropriate for correct timings.
4. **Test one hypothesis at a time:** preloaded data, elimination of unnecessary copies, batching, worker count, pinned host memory, async transfers with appropriate synchronization, or optimized kernels.
5. **Validate:** rerun comparable workloads and compare end-to-end throughput and latency, GPU time, memory use, clocks, power, and correctness; preserve raw results.
6. **For Kubernetes:** check pod scheduling, GPU requests and node allocatable resources before assuming a new replica can run. A second pod requesting one whole GPU will need spare allocatable GPU capacity or a deliberately configured sharing mechanism.

## Interview-ready incident summary

> I investigated poor throughput on an RTX 3070 Laptop GPU using controlled PyTorch matrix-multiplication workloads and NVIDIA telemetry. Artificial pacing reduced throughput to 3.96 operations/s versus 477.32 operations/s when continuously supplied with work, despite unchanged tensor memory allocation. A separate experiment showed 131.43 operations/s when repeatedly copying CPU tensors to the GPU versus 469.47 operations/s when reusing GPU-resident inputs. CUDA event timing showed matrix-multiplication kernels remained close to 2 ms, indicating substantial host/transfer-side overhead rather than a slow kernel. I documented an anomalous persistent memory-activity reading for follow-up instead of assuming memory bandwidth was saturated.

## Next steps / open questions

- Save reproducible versions of `gpu_experiment.py` and `transfer_bottleneck.py` alongside these notes and the raw monitoring logs.
- Time CPU-to-GPU copies independently; compare pageable and pinned-memory strategies where appropriate.
- Recheck unusual 100% memory-utilization reporting with an independent metric source, where available.
- Investigate whether the GPU hits power or thermal clock limits in longer-running, comparable experiments.
- Incorporate the findings into the integrated Kubernetes GPU/ML platform's performance troubleshooting runbook.

## Core lesson

GPU utilization measures sampled kernel activity, not peak compute efficiency, memory capacity consumption, or application quality. Low utilization can result from insufficient demand, CPU/data preparation, host-to-device transfers, synchronization, or small workloads; high utilization can coexist with inefficient kernels or throttling. Diagnose with correlated application and hardware measurements, isolate one bottleneck, make a controlled change, and judge success primarily by required throughput, latency, correctness, and resource efficiency.
