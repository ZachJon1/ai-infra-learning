# Day 30 — Multi-GPU Fundamentals

## Objective

Understand how multiple GPUs are organized and used for AI workloads, how GPUs communicate, why multi-GPU scaling is not automatically linear, and how to troubleshoot common multi-GPU infrastructure problems.

This lesson focuses on the infrastructure concepts required before moving into data parallelism, tensor/model parallelism, PyTorch Distributed, NCCL, and multi-node training.

---

## 1. Multiple GPU Workloads vs. a Multi-GPU Workload

Multiple GPUs can be used in two fundamentally different ways.

### Independent workloads

```text
Pod / Process A → GPU 0
Pod / Process B → GPU 1
Pod / Process C → GPU 2
Pod / Process D → GPU 3
```

Each workload operates independently and may not communicate with the others.

### Distributed multi-GPU workload

```text
                  Training Job
                       |
        +--------------+--------------+
        |              |              |
      GPU 0          GPU 1          GPU 2
        \              |              /
         +------ Communication -------+
```

One logical workload is distributed across multiple GPUs. This requires coordination and usually inter-GPU communication.

---

## 2. GPU Memory Is Normally Local

Multiple GPUs do not automatically behave like one GPU with a large shared VRAM pool.

For example:

```text
GPU 0: 48 GB
GPU 1: 48 GB
GPU 2: 48 GB
```

The machine physically contains 144 GB of GPU memory, but an application cannot automatically treat it as one contiguous 144 GB memory space.

A model larger than the VRAM of a single GPU must be explicitly partitioned or sharded across GPUs.

---

## 3. GPU Interconnects

GPU communication performance depends on how the GPUs are physically connected.

### PCIe

PCI Express is commonly used to connect GPUs to the host system and can also provide GPU-to-GPU communication paths.

### NVLink

NVLink provides higher-bandwidth GPU-to-GPU connectivity on supported NVIDIA systems.

### NVSwitch

NVSwitch provides a switching fabric allowing multiple NVLink-connected GPUs to communicate efficiently in larger GPU systems.

The important infrastructure question is not only:

> How many GPUs are installed?

but also:

> How are those GPUs connected?

Useful topology command:

```bash
nvidia-smi topo -m
```

---

## 4. Processes, Ranks, and World Size

A common distributed-training configuration uses one process per GPU.

```text
Rank 0 → GPU 0
Rank 1 → GPU 1
Rank 2 → GPU 2
Rank 3 → GPU 3
```

`world_size` represents the total number of participating distributed processes.

For example:

```text
world_size = 4
```

means four processes/ranks participate in the distributed job.

This often corresponds to four GPUs in a one-process-per-GPU configuration, but world size technically counts processes rather than GPUs.

---

## 5. Collective Communication

Distributed GPU workloads often need to exchange information.

Common collective operations include:

```text
Broadcast
Reduce
AllReduce
AllGather
ReduceScatter
```

### Reduce

Combine values from multiple ranks into one destination.

```text
Rank 0 ─┐
Rank 1 ─┤
Rank 2 ─┼── Reduce ──► one result
Rank 3 ─┘
```

### AllReduce

Combine values and return the result to all participating ranks.

```text
g0 ─┐
g1 ─┤
g2 ─┼── AllReduce ──► combined result → all ranks
g3 ─┘
```

AllReduce is particularly important for synchronizing gradients during distributed training.

---

## 6. Data Parallelism vs. Model Parallelism

### Data Parallelism

The model is replicated while the data is divided.

```text
GPU 0 → Full model + Data A
GPU 1 → Full model + Data B
GPU 2 → Full model + Data C
GPU 3 → Full model + Data D
```

Key idea:

```text
MODEL → replicated
DATA  → partitioned
```

### Model Parallelism

The model itself is partitioned across GPUs.

```text
GPU 0 → Model Part A
GPU 1 → Model Part B
GPU 2 → Model Part C
```

Key idea:

```text
MODEL → partitioned
```

This becomes necessary when a model or its working state cannot fit on a single GPU.

### Tensor Parallelism

Tensor parallelism divides individual tensors or operations within model layers across multiple GPUs.

Because GPUs frequently exchange partial computation results, tensor parallelism can be especially sensitive to inter-GPU bandwidth and topology.

---

## 7. Multi-GPU Scaling

Adding GPUs does not guarantee linear performance improvement.

### Speedup

```text
Speedup = single-GPU execution time / multi-GPU execution time
```

Example:

```text
