# Day 25 — PyTorch GPU Profiling

## Objective
Investigate CPU/GPU execution, CUDA kernel activity, memory allocation, and the performance impact of unnecessary synchronization.

## Tools
- Python
- PyTorch
- CUDA
- PyTorch Profiler
- Perfetto Trace Viewer

## Experiments
1. Profile ten matrix multiplication and ReLU operations.
2. Inspect CPU and GPU activity through an execution trace.
3. Benchmark 30 workload iterations with and without synchronization after each iteration.

## Profiling observations
- CUDA kernel execution was dominated by matrix multiplication.
- The matrix multiplication kernel recorded approximately 2.292 ms across ten calls.
- ReLU kernel execution totaled approximately 0.198 ms.
- The profiling trace showed CPU launch activity and GPU execution behavior.

## Benchmark results

| Configuration | Median wall-clock time |
|---|---:|
| Synchronize every iteration | 9.731 ms |
| Synchronize at the end | 9.151 ms |

**Observed result:** Removing repeated synchronization reduced median wall-clock execution time by approximately 6.0% in this experiment.

## Engineering conclusions
- CUDA operations execute asynchronously.
- Unsynchronized Python timers can underestimate completed GPU execution time.
- Unnecessary synchronization can introduce CPU stalls and GPU idle gaps.
- A profiler identifies expensive operations, but kernel duration alone does not establish whether a workload is compute-bound or memory-bound.
- Performance improvements should be validated through repeated measurements.

## Reproduction

```bash
python profile_matmul.py
python benchmark_sync.py
```

## Future investigation
- Compare different matrix sizes.
- Record median CUDA-event timings.
- Report variability across repeated benchmark trials.
- Explore CPU preprocessing overhead and batching.
