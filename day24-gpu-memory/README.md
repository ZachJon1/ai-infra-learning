# Day 24 -  GPU Memory Allocation and OOM Diagnosis

## Objectives

The goal of Day 24 was to understand how GPU memory is consumed by AI workloads and develop a systematic approach to diagnosing CUDA out-of-memory (OOM) failures.

By the end of the lab, I was able to:

- distinguish allocated, reserved, and free GPU memory
- explain why `nvidia-smi` and PyTorch can report different memory usage
- identify the major sources of GPU memory consumption
- diagnose multiple CUDA OOM patterns
- distinguish live-memory growth from normal PyTorch caching
- understand when allocator fragmentation may be relevant
- apply mitigation strategies based on the actual cause of an OOM

## GPU Memory Model

GPU memory consumption during model execution can include:

- model parameters
- gradients
- optimizer state
- activations
- temporary/workspace allocations
- CUDA context/runtime overhead

Training generally requires more GPU memory than inference because training must additionally maintain gradients, optimizer state, and activations required for backpropagation.

The important infrastructure question is not simply:

> How large is the model?

but:

> What is the peak GPU memory requirement of the workload?

## PyTorch Allocated vs Reserved Memory

PyTorch uses a CUDA caching allocator.

### Allocated memory

```python
torch.cuda.memory_allocated()
```

This represents memory currently occupied by live PyTorch tensors.

### Reserved memory

```python
torch.cuda.memory_reserved()
```

This represents memory controlled by PyTorch's caching allocator.

Conceptually:

```text
reserved memory
├── allocated to live tensors
└── cached/unallocated memory available for reuse
```

Therefore:

```text
reserved >= allocated
```

### Peak allocated memory

```python
torch.cuda.max_memory_allocated()
```

This records the highest allocated memory observed since the peak statistics were reset.

```python
torch.cuda.reset_peak_memory_stats()
```

Peak memory is particularly important because an application can fail during a temporary memory spike even when its normal memory usage appears safe.

## Experiment 1 — PyTorch Caching Allocator

Script:

```text
memory_test.py
```

Observed results:

```text
--- initial ---
allocated: 0.000 GB
reserved : 0.000 GB
peak     : 0.000 GB

--- after x ---
allocated: 0.373 GB
reserved : 0.373 GB
peak     : 0.373 GB

--- after y ---
allocated: 0.746 GB
reserved : 0.746 GB
peak     : 0.746 GB

--- after deleting y ---
allocated: 0.373 GB
reserved : 0.746 GB
peak     : 0.746 GB

--- after empty_cache ---
allocated: 0.373 GB
reserved : 0.373 GB
peak     : 0.746 GB
```

### Key finding

Deleting a tensor reduced allocated memory, but PyTorch initially kept the released memory reserved for reuse.

```python
del y
```

resulted in:

```text
allocated ↓
reserved  unchanged
```

Calling:

```python
torch.cuda.empty_cache()
```

released unused cached memory, causing reserved memory to decrease.

However, `empty_cache()` did not release memory associated with the still-live tensor `x`.

## Experiment 2 — Impossible Allocation OOM

Script:

```text
oom_test.py
```

GPU state:

```text
Total GPU memory : 7.65 GB
Free GPU memory  : 7.20 GB
Allocated        : 0.00 GB
Reserved         : 0.00 GB
```

Attempted allocation:

```text
7.92 GiB
```

Result:

```text
CUDA out of memory
```

The requested allocation was larger than both the available free memory and the total physical capacity of the GPU.

### Diagnosis

This was a straightforward capacity failure.

```text
request = 7.92 GiB
free    = 7.20 GiB
total   = 7.65 GiB
```

The allocation could not fit even on an otherwise empty GPU.

`torch.cuda.empty_cache()` would not solve this problem because there was no significant cached PyTorch memory to release.

## Experiment 3 — Existing Working Set + New Allocation

Script:

```text
live_tensor_oom.py
```

Observed state after the first allocation:

```text
free      : 3.25 GB
allocated : 3.97 GB
reserved  : 3.97 GB
```

The second allocation requested:

```text
3.61 GB
```

but only approximately:

```text
3.25 GB
```

was free.

The result was another CUDA OOM.

### Diagnosis

Unlike the first experiment, the requested tensor was not larger than the total GPU capacity.

Instead, an existing live allocation had already consumed approximately 3.97 GB.

```text
existing live tensor
        +
new allocation
        >
available GPU memory
```

This represents a common AI workload failure pattern where model weights or other persistent tensors are already resident on the GPU and a new batch or operation pushes the working set beyond capacity.

## Experiment 4 — Gradual Memory Growth

Script:

```text
memory_growth.py
```

The program intentionally retained every generated CUDA tensor:

```python
outputs.append(y)
```

Observed allocated memory:

```text
iteration 01 → 0.37 GB
iteration 05 → 1.87 GB
iteration 10 → 3.73 GB
iteration 15 → 5.60 GB
iteration 19 → 7.09 GB
iteration 20 → OOM
```

At failure:

```text
PyTorch allocated:          7.09 GiB
reserved but unallocated:      0 bytes
free:                       44.81 MiB
failed allocation:         382.00 MiB
```

### Diagnosis

Each individual tensor fit comfortably on the GPU.

The failure occurred because previous tensors remained referenced inside the Python list.

```text
tensor 1 ─┐
tensor 2 ─┤
tensor 3 ─┤
...       ├── outputs[]
tensor N ─┘
```

Because the references remained alive, PyTorch could not release their GPU memory.

This created continuous allocated-memory growth until the GPU was exhausted.

Possible fixes include:

```python
result = y.mean().item()
```

when only a scalar is needed, or:

```python
outputs.append(y.detach().cpu())
```

when results need to be retained but do not need to remain on the GPU.

## Inference Memory

For inference workloads, gradient tracking is normally unnecessary.

A typical pattern is:

```python
model.eval()

with torch.inference_mode():
    output = model(x)
```

`torch.inference_mode()` prevents autograd tracking and reduces unnecessary memory and compute overhead.

It does not eliminate all activations because intermediate tensors are still required during the forward pass.

## Memory Leak vs Caching

A critical troubleshooting distinction is:

### Allocated memory continuously increasing

```text
2 GB → 3 GB → 4 GB → 5 GB
```

This suggests that live GPU objects may be accumulating.

Possible causes include:

- retained output tensors
- retained computation graphs
- application caches
- growing queues
- unnecessary GPU-side state

### Reserved memory remains high while allocated memory falls

```text
allocated: 8 GB → 4 GB
reserved : 10 GB → 10 GB
```

This is not automatically a memory leak.

It can simply be normal behavior from PyTorch's caching allocator.

## Fragmentation

Allocator fragmentation can become relevant when PyTorch has substantial reserved-but-unallocated memory but is still unable to satisfy an allocation.

A useful warning sign is a large difference between:

```python
torch.cuda.memory_reserved()
```

and:

```python
torch.cuda.memory_allocated()
```

during an OOM.

Variable allocation sizes, such as dynamically changing inference batch sizes, can make allocator behavior more important.

Useful diagnostic commands include:

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
torch.cuda.memory_summary()
torch.cuda.memory_stats()
```

## PyTorch vs nvidia-smi

PyTorch allocator statistics do not necessarily equal GPU memory usage reported by:

```bash
nvidia-smi
```

For example:

```text
PyTorch allocated = 6 GB
PyTorch reserved  = 8 GB
nvidia-smi        = 14 GB
```

Additional GPU memory may be associated with:

- CUDA runtime/context
- other GPU processes
- non-PyTorch CUDA allocations
- libraries outside the PyTorch caching allocator

A large difference between `nvidia-smi` usage and PyTorch reserved memory should therefore trigger investigation of other GPU consumers.

## OOM Troubleshooting Workflow

When encountering:

```text
CUDA out of memory
```

first inspect:

```text
1. GPU total capacity
2. Current free GPU memory
3. Failed allocation size
4. PyTorch allocated memory
5. PyTorch reserved memory
6. Peak allocated memory
7. Other GPU processes
```

Then classify the failure.

### Capacity / working-set problem

Characteristics:

```text
allocated memory is high
reserved-but-unused memory is small
new allocation > remaining free memory
```

Possible mitigations:

- reduce batch size
- reduce sequence/input
