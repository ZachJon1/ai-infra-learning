# Day 31 — Data Parallelism with PyTorch DistributedDataParallel (DDP)

## Objective

Understand **synchronous data-parallel training**, implement a two-process PyTorch DDP job, verify consistent model parameters across ranks, and reproduce a **straggler-induced slowdown**.

This lab is part of an AI Infrastructure / ML Systems learning portfolio. It focuses on both **correctness** and **operational efficiency**.

## Architecture

```text
                       torchrun
                  (world size = 2)
                          |
              +-----------+-----------+
              |                       |
           Rank 0                  Rank 1
          CPU/Gloo                CPU/Gloo
        Full model copy         Full model copy
        Distinct samples        Distinct samples
              |                       |
         Forward/backward       Forward/backward
              |                       |
              +------ DDP ------------+
               gradient AllReduce
                          |
              Same gradient-based update
                          |
                 Same model parameters
```

> **Hardware scope:** This lab used **two CPU worker processes** with the **Gloo** communication backend on a single machine. It demonstrates DDP semantics and synchronization, **not** performance or scaling of multiple physical GPUs, NCCL, or a multi-node cluster.

## Concepts covered

- **Data parallelism:** Each worker maintains a complete model replica and trains on different samples.
- **Rank / world size:** Ranks identify worker processes; world size is the total number of participating processes.
- **`DistributedSampler`:** Partitions dataset indices among workers and reshuffles consistently between epochs via `set_epoch`.
- **DDP synchronization:** `loss.backward()` participates in DDP gradient synchronization. Workers apply corresponding parameter updates.
- **AllReduce:** A collective reduction whose output is available to all participating processes; gradient synchronization and reporting loss aggregates serve different purposes.
- **Stragglers:** A slow rank can make faster ranks wait and increase the total step/job duration.

With equally sized per-rank batches, the effective global batch size is:

```text
local batch size × world size × gradient accumulation steps
```

For this experiment, the per-step global batch size is **16 × 2 = 32** (without gradient accumulation).

## Prerequisites

- Python 3
- PyTorch with distributed support
- `torchrun` on `PATH`
- No GPU required for the CPU/Gloo version

If needed, activate the environment where PyTorch is already installed:

```bash
source ~/ai-infra-venv/bin/activate
```

Verify the environment:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
torchrun --help
python -m py_compile train_ddp.py
```

The exact PyTorch version was not recorded with these measurements; for a fully controlled reproduction, record `torch.__version__` and system information when rerunning.

## Training experiment

The script `train_ddp.py` trains a small linear regression model on a synthetic dataset containing 128 samples:

```text
y = 3x + 2
```

Configuration:

| Item | Value |
|---|---|
| Nodes | 1 |
| Processes | 2 |
| Backend | Gloo |
| Device | CPU |
| Model | `torch.nn.Linear(1, 1)` |
| Dataset | 128 synthetic samples |
| Local batch size | 16 |
| Steps per rank per epoch | 4 |
| Epochs | 10 |
| Optimizer | SGD, learning rate 0.15 |
| Expected ground-truth parameters | weight = 3, bias = 2 |

Run the training script:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 train_ddp.py --backend gloo
```

Observed data partitioning (first six sampler indices):

```text
Rank 0: [44, 121, 31, 71, 105, 15]
Rank 1: [94, 13, 116, 84, 14, 22]
```

The two ranks sampled different training examples. The final model was close to the target function:

| Metric | Measured |
|---|---:|
| Epoch 1 mean loss | 1.895290 |
| Epoch 10 mean loss | 0.000551 |
| Final weight | 2.9696571826934814 |
| Final bias | 2.0002129077911377 |
| Maximum parameter difference across ranks | **0.0** |

The **zero parameter difference** confirms that the model copies had identical parameters at the end of this successful run. It does not by itself provide a performance benchmark.

### Important distinction

- DDP's gradient synchronization occurs during backward propagation.
- The explicit `dist.all_reduce(total_loss)` in the script aggregates a **reporting metric** and is *not* the mechanism that makes the parameter updates consistent.
- The `dist.all_gather(...)` near the end collects parameters to compare the replicas.

## Break/fix experiment — slow worker

To reproduce a straggler, the training loop conditionally delays **Rank 1** for 200 ms immediately before `loss.backward()`:

```python
if rank == 1 and os.getenv("SIMULATE_STRAGGLER") == "1":
    time.sleep(0.2)
```

The script must also import `os` and `time`.

Run both cases from this folder:

```bash
# Baseline
time torchrun --standalone --nnodes=1 --nproc-per-node=2 train_ddp.py --backend gloo

# Rank 1 is 200 ms slower on each of its 40 training steps
time env SIMULATE_STRAGGLER=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 train_ddp.py --backend gloo
```

### Recorded measurements (October 8, 2026)

| Measurement | Baseline | Straggler |
|---|---:|---:|
| Shell `real` elapsed time | **3.852 s** | **12.295 s** |
| Rank 1 artificial delay | 0 ms/step | 200 ms/step |
| Final mean loss | 0.000551 | 0.000551 |
| Final weight | 2.9696571826934814 | 2.9696571826934814 |
| Final bias | 2.0002129077911377 | 2.0002129077911377 |
| Max parameter difference | 0.0 | 0.0 |

- **Added elapsed time:** 12.295 − 3.852 = **8.443 s**.
- **Observed slowdown factor:** 12.295 / 3.852 ≈ **3.19×**.
- **Artificial delay budget:** 40 steps × 0.2 s = **8.0 s** on Rank 1.

The measured increase is close to the injected delay. Despite the slowdown, both experiments produced the same reported loss and parameters. This illustrates that a distributed training job can be **numerically correct but operationally inefficient**.

These are **single-run end-to-end shell wall-clock measurements**, including process startup and shutdown. They are not statistically robust benchmark averages or isolated synchronization timings. Profiling individual ranks would be required to quantify their wait time.

Machine-readable results are saved in [`benchmark_results.csv`](benchmark_results.csv).

## Troubleshooting lessons

### Shell line continuation

A backslash must be the **last character on a Bash line** to continue the command. A one-line invocation avoids accidentally passing a malformed filename to `torchrun`.

### Python indentation

A `TabError: inconsistent use of tabs and spaces in indentation` occurred while inserting the simulation. The script was corrected using consistent four-space indentation and the experiment reran successfully.

Validate before launching multiple workers:

```bash
python -m py_compile train_ddp.py
python -m tabnanny train_ddp.py
```

### Distributed job hangs

When a production DDP job hangs, investigate **all ranks**, not just the apparent point of failure in one process. Potential causes include mismatched collectives, failed/stalled ranks, data-loading stalls, and communication problems. For a supported distributed environment, useful diagnostic settings include:

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun ...
# NCCL_DEBUG=INFO can provide additional output for NCCL-backed GPU runs.
```

Only enable NCCL-focused diagnostics when testing the NCCL backend, and inspect per-rank logs and profiler traces before attributing a slowdown to interconnect performance.

## Key takeaways

1. DDP replicates the model on every rank while partitioning training data.
2. Gradient synchronization makes corresponding replicas use consistent updates.
3. Training correctness and infrastructure performance are different properties.
4. A slow worker can delay faster workers at communication synchronization points.
5. Model replication alone cannot solve the problem of a model that exceeds any one GPU's memory; later lessons cover tensor/pipeline parallelism and parameter sharding.

## Next steps

- Day 32: **Model parallelism and tensor parallelism**.
- When suitable hardware is available: run multi-GPU DDP with **NCCL**, collect GPU/per-rank profiling data, and compare scaling against a single-GPU baseline.
- Later portfolio work: expose training telemetry and automate regression/troubleshooting checks.
