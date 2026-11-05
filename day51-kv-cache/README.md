# Day 51 — KV Cache and PagedAttention

**Track:** AI Infrastructure & ML Systems — 220-Day Roadmap  
**Phase:** 3 — Model Serving / Inference Infrastructure  
**Topic:** KV caching, memory sizing, and block-based allocation  
**Lab:** `kv_cache_sim.py`

## Overview

Large language models generate tokens autoregressively. During decoding, a transformer can reuse the **keys (K)** and **values (V)** calculated for previously processed tokens instead of recalculating them at every step. The **KV cache** speeds up decoding, but consumes memory as the number of cached tokens and active requests grows.

This lab uses a small Python simulation to compare two *simplified allocation strategies*:

- **Fixed allocation:** Reserve 128 token slots for each active request, regardless of its current length.
- **Paged allocation:** Allocate token slots in blocks of 16, adding blocks as necessary.

It illustrates why block-based allocation can reduce unused reserved capacity, a principle relevant to vLLM's **PagedAttention**. This is an allocation model, **not** a benchmark or implementation of vLLM or GPU attention.

## Learning objectives

- Explain how KV caching avoids recomputing historical keys and values during autoregressive decoding.
- Estimate KV-cache memory from model architecture, sequence length, cache precision, and concurrency.
- Explain why PagedAttention maps logical KV blocks to potentially noncontiguous physical blocks.
- Quantify reserved capacity and internal waste under fixed and block-based allocation.
- Recognize how KV-cache pressure can contribute to preemption, queueing, and increased inference latency.

## Repository contents

```text
day51-kv-cache/
├── kv_cache_sim.py     # Python allocation simulation
└── README.md           # Explanation, experiments, and findings
```

## Prerequisites and usage

- Python 3.8+ (standard library only; no GPU required)
- A terminal

```bash
cd ~/ai-infra/day51-kv-cache
python3 kv_cache_sim.py
```

The exercise uses these settings:

```python
BLOCK_SIZE = 16
FIXED_CAPACITY = 128

requests = {
    "A": 35,
    "B": 100,
    "C": 16,
    "D": 48,
}
```

The program reports tokens, blocks, allocated slots, unused slots, and paged-allocation efficiency. To reproduce the additional experiments, edit the request dictionary and rerun the script (or call the `analyze` function with the modified dictionary).

## How the simulation works

For a request containing `tokens` cached tokens:

```python
blocks = math.ceil(tokens / BLOCK_SIZE)
allocated_slots = blocks * BLOCK_SIZE
```

Across all active requests:

```text
Fixed slots         = number_of_requests × FIXED_CAPACITY
Paged slots         = Σ ceil(tokens_i / BLOCK_SIZE) × BLOCK_SIZE
Actual tokens       = Σ tokens_i
Fixed waste         = Fixed slots − Actual tokens
Paged waste         = Paged slots − Actual tokens
Paged efficiency    = Actual tokens / Paged slots × 100%
```

These quantities are **token-slot equivalents**, not direct measurements of GPU bytes. Real KV-cache memory also depends on the transformer architecture and data type.

## Experiments and observed results

### Experiment 1 — Initial requests

| Request | Cached tokens | Blocks (16 slots each) | Paged slots |
|---|---:|---:|---:|
| A | 35 | 3 | 48 |
| B | 100 | 7 | 112 |
| C | 16 | 1 | 16 |
| D | 48 | 3 | 48 |
| **Total** | **199** | **14** | **224** |

| Metric | Observed value |
|---|---:|
| Fixed allocation | 512 slots |
| Paged allocation | 224 slots |
| Fixed unused capacity | 313 slots |
| Paged unused capacity | 25 slots |
| Paged allocation efficiency | 88.84% |
| Reduction in reserved slots vs. fixed | **56.25%** |

**Interpretation:** With the same 199 cached tokens, the simplified paged scheme reserves 288 fewer token slots (512 − 224) than the fixed-capacity scheme.

### Experiment 2 — Request B completes

Remove request B:

```python
requests.pop("B")
analyze(requests)
```

| Metric | Observed value |
|---|---:|
| Active tokens (A + C + D) | 99 |
| Fixed allocation | 384 slots |
| Paged allocation | 112 slots |
| Fixed unused capacity | 285 slots |
| Paged unused capacity | 13 slots |
| Paged allocation efficiency | 88.39% |

**Interpretation:** When a request completes, the allocator can release its reserved KV-cache capacity. The remaining three requests require only 112 paged slots in this model.

### Experiment 3 — Request B grows to 120 tokens

Restore B with a longer sequence:

```python
requests["B"] = 120
analyze(requests)
```

| Metric | Observed value |
|---|---:|
| Active tokens | 219 |
| Fixed allocation | 512 slots |
| Paged allocation | 240 slots |
| Fixed unused capacity | 293 slots |
| Paged unused capacity | 21 slots |
| Paged allocation efficiency | 91.25% |

For request B alone, allocation changes from **7 blocks / 112 slots** at 100 tokens to **8 blocks / 128 slots** at 120 tokens. Block allocation grows in discrete steps, not one token slot at a time.

## KV-cache capacity planning

For a conventional full-attention transformer with equal cached sequence lengths, an approximate KV-cache tensor storage formula is:

\[
M_{\mathrm{KV}} = 2 \times L \times T \times H_{\mathrm{KV}} \times D \times S \times B
\]

Where:

- `L`: transformer layers
- `T`: cached tokens per request (including processed prompt and generated tokens)
- `H_KV`: KV heads per layer
- `D`: dimension per KV head
- `S`: bytes per cached element (e.g., 2 for FP16/BF16)
- `B`: simultaneous requests, assuming the same token count
- `2`: separate keys and values

**Worked example:** For 32 layers, 4,096 tokens, 8 KV heads, head dimension 128, BF16 (2 bytes), and one request:

```text
2 × 32 × 4096 × 8 × 128 × 2 × 1
= 536,870,912 bytes
= 512 MiB = 0.5 GiB
```

At 12 equally long concurrent requests, the KV tensor storage would be **6 GiB**, before allocation overhead and other GPU memory requirements. For varying request lengths, sum the per-request sizes. Some architectures use sliding-window or other nonstandard cache policies, so the formula should not be assumed universal.

## PagedAttention and continuous batching

**PagedAttention** organizes cached K/V states into logical blocks whose physical storage blocks need not be contiguous. A block table maps logical blocks to physical memory locations. This can reduce memory fragmentation and over-reservation, especially for requests with different sequence lengths.

**Continuous batching** dynamically schedules active requests, admitting new work and retiring finished requests between decoding steps. The two mechanisms are complementary:

| Mechanism | Primary concern |
|---|---|
| Continuous batching | Which requests execute on each scheduling iteration |
| PagedAttention | Where the requests' KV-cache blocks are stored and how they are accessed |

PagedAttention improves **allocation efficiency**; it does not remove the fundamental need to store K/V tensors for active contexts.

## Production troubleshooting notes

Symptoms of KV-cache pressure can include high KV-cache utilization, waiting requests, rising preemption counts, and deteriorating end-to-end latency—even when GPU compute utilization is below 100%.

A useful investigation should examine:

1. **KV-cache utilization:** Is the cache pool nearly exhausted?
2. **Preemption frequency:** Are requests being evicted and later recomputed?
3. **Queue depth / waiting requests:** Is admission into execution constrained?
4. **User-facing latency:** Check TTFT, ITL, and P95/P99 end-to-end latency.
5. **GPU memory and compute:** Distinguish VRAM capacity pressure from compute saturation.

Possible mitigations include limiting concurrent sequences, reducing actual context sizes, using supported lower-precision KV-cache storage, freeing VRAM via model-weight quantization, or scaling to more GPU capacity. Each has tradeoffs in quality, latency, throughput, or cost.

> **Important:** A CUDA GPU-memory allocation failure is not synonymous with Kubernetes `OOMKilled`, which generally refers to a container exceeding its system-memory cgroup limit. Likewise, vLLM RECOMPUTE preemption releases an affected request's cache allocations; it does not necessarily clear the entire server's KV cache.

See [vLLM optimization documentation](https://docs.vllm.ai/en/latest/configuration/optimization/) for current configuration guidance. Configuration flags, metric names, and preemption behavior may vary by vLLM version.

## Limitations

- The simulator counts reserved **token slots**, not VRAM bytes, attention FLOPs, or physical GPU allocations.
- The fixed strategy intentionally over-reserves 128 slots per request; it is an illustrative baseline rather than a faithful representation of every nonpaged implementation.
- It does not model block tables, sharing, prefix caching, memory fragmentation across physical allocations, eviction policy, attention kernels, or allocator metadata.
- No actual vLLM server, NVIDIA GPU benchmark, latency test, or throughput measurement was performed in this lab.
- Reported numerical results are the observed outputs of this Python simulation, not claims about production speedups.

## Key takeaways

KV caching trades additional memory for less redundant attention-related computation during decoding. Memory demand scales with sequence length and concurrency, making cache capacity a central operational constraint for LLM serving. PagedAttention manages that capacity in blocks, reducing unused reserved space, while continuous batching manages request execution. In production, diagnose bottlenecks using cache utilization, preemption, queueing, latency, and GPU memory metrics together.

## Further reading

- [vLLM project](https://github.com/vllm-project/vllm)
- [vLLM documentation](https://docs.vllm.ai/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (paper)](https://arxiv.org/abs/2309.06180)
