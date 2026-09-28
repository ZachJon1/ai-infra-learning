# Day 23 — GPU Monitoring and Troubleshooting with `nvidia-smi`

## Overview

Day 23 focuses on **GPU monitoring, performance interpretation, and troubleshooting using NVIDIA's `nvidia-smi` tooling**.

The objective of this lab was not simply to learn how to run `nvidia-smi`, but to understand how to interpret GPU telemetry in the context of AI infrastructure workloads.

The experiments explored:

- GPU utilization
- VRAM allocation
- GPU processes
- P-states
- SM and memory clocks
- power consumption
- temperature
- thermal slowdown
- software power caps
- process-level GPU ownership
- GPU starvation
- GPU memory exhaustion
- GPU performance bottlenecks

The lab also demonstrated why GPU troubleshooting requires correlating multiple metrics rather than interpreting a single number such as GPU utilization.

---

# Learning Objectives

By the end of this lab, I was able to:

1. Identify available NVIDIA GPUs.
2. Distinguish between NVIDIA driver CUDA compatibility and the installed CUDA Toolkit.
3. Monitor GPU utilization continuously.
4. Interpret GPU VRAM usage.
5. Distinguish VRAM allocation from GPU computation.
6. Identify GPU processes and map them to Linux PIDs.
7. Interpret GPU P-states.
8. Monitor SM clocks and memory clocks.
9. Monitor temperature and power consumption.
10. Identify software power limiting.
11. Identify software thermal slowdown.
12. Diagnose GPU starvation from upstream CPU bottlenecks.
13. Distinguish compute bottlenecks from memory-capacity bottlenecks.
14. Recognize that 100% GPU utilization does not necessarily mean optimal performance.
15. Build a basic GPU troubleshooting workflow.

---

# Test Environment

The experiments were performed on:

```text
GPU: NVIDIA GeForce RTX 3070 Laptop GPU
VRAM: 8192 MiB
Driver Version: 580.178.04
Driver-supported CUDA Version: 13.0
CUDA Toolkit Version: 13.0
NVCC Version: 13.0.88
Operating System: Linux
Framework: PyTorch
```

The GPU was also driving the graphical desktop, so some baseline VRAM consumption was associated with:

```text
Xorg
GNOME Shell
Desktop applications
```

---

# 1. Checking GPU Availability

The primary GPU inspection command is:

```bash
nvidia-smi
```

Example output from the system:

```text
NVIDIA-SMI 580.178.04
Driver Version: 580.178.04
CUDA Version: 13.0

GPU: NVIDIA GeForce RTX 3070 Laptop GPU
Memory: 8192 MiB
```

To list GPUs:

```bash
nvidia-smi --list-gpus
```

Observed:

```text
GPU 0: NVIDIA GeForce RTX 3070 Laptop GPU
```

---

# 2. Driver CUDA Version vs CUDA Toolkit Version

The CUDA version shown by:

```bash
nvidia-smi
```

should not automatically be interpreted as the installed CUDA Toolkit version.

`nvidia-smi` reports the CUDA API/runtime compatibility level supported by the currently installed NVIDIA driver.

The installed toolkit can be checked with:

```bash
nvcc --version
```

Observed:

```text
Cuda compilation tools, release 13.0, V13.0.88
```

In this environment:

```text
Driver-supported CUDA: 13.0
Installed CUDA Toolkit: 13.0
```

The two happened to match.

---

# 3. Querying Specific GPU Metrics

Instead of viewing the full `nvidia-smi` interface, selected fields can be requested directly.

```bash
nvidia-smi \
  --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
  --format=csv
```

Example idle result:

```text
NVIDIA GeForce RTX 3070 Laptop GPU
Driver: 580.178.04
Total Memory: 8192 MiB
Used Memory: ~439 MiB
Free Memory: ~7393 MiB
GPU Utilization: 0%
Temperature: 61 C
Power Draw: ~22 W
```

This represented an essentially idle GPU.

---

# 4. GPU Utilization vs VRAM Allocation

One of the most important concepts from this lab is:

```text
VRAM usage != GPU computation
```

A GPU can have a large amount of memory allocated while doing no computation.

For example:

```text
VRAM:      7 GB / 8 GB
GPU Util:  0%
```

This could occur if a model is loaded into GPU memory but is currently waiting for requests.

Likewise:

```text
VRAM:      2 GB / 8 GB
GPU Util:  100%
```

is also possible.

The workload may simply require relatively little memory while performing significant computation.

Therefore:

```text
GPU-Util
```

and:

```text
Memory-Usage
```

must be interpreted separately.

---

# 5. Continuous GPU Monitoring

A single `nvidia-smi` execution only gives a snapshot.

For continuous monitoring:

```bash
watch -n 1 nvidia-smi
```

This refreshes GPU status every second.

Another useful option is:

```bash
nvidia-smi dmon
```

This provides continuous device-level metrics including:

```text
Power
Temperature
SM utilization
Memory activity
Encoder utilization
Decoder utilization
```

A custom monitoring query can also be used:

```bash
nvidia-smi \
  --query-gpu=timestamp,pstate,utilization.gpu,memory.used,memory.free,clocks.sm,clocks.mem,temperature.gpu,power.draw \
  --format=csv \
  -l 1
```

The `-l 1` option collects metrics once every second.

---

# 6. Saving GPU Telemetry

GPU metrics can be written to a CSV file for later analysis.

Example:

```bash
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv \
  -l 1 > gpu-monitor.csv
```

This creates a basic GPU telemetry dataset that can later be visualized or analyzed.

This same idea scales into production monitoring systems such as:

```text
GPU
 ↓
DCGM Exporter
 ↓
Prometheus
 ↓
Grafana
 ↓
Alerts
```

---

# 7. PyTorch GPU Workload

A PyTorch matrix multiplication workload was used to generate significant GPU activity.

```python
import torch
import time

device = "cuda"

print(f"GPU: {torch.cuda.get_device_name(0)}")

a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

for _ in range(100):
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

time.sleep(10)
```

The final `sleep()` was intentionally added to keep the Python process alive after computation had stopped.

This made it possible to observe:

```text
Python process still running
VRAM still allocated
GPU utilization approximately 0%
```

This demonstrated that:

```text
allocated GPU memory
```

does not imply:

```text
active GPU computation
```

---

# 8. Observed GPU Behavior During Computation

During the workload, the GPU reached approximately:

```text
P-State:              P0
GPU Utilization:      100%
SM Clock:             1170–1260 MHz
Memory Clock:         7001 MHz
Temperature:          79–82 C
Power Draw:           ~124.7 W
VRAM Usage:           ~2054 MiB
```

During idle conditions after the workload:

```text
P-State:              P8
GPU Utilization:      0–3%
SM Clock:             210 MHz
Memory Clock:         405 MHz
Temperature:          ~71 C
Power Draw:           ~24 W
VRAM Usage:           ~350 MiB
```

This demonstrated the GPU's dynamic power-management behavior.

---

# 9. GPU Performance States

NVIDIA GPUs dynamically transition between performance states.

During this experiment:

```text
P0
```

was observed during heavy computation.

After the workload stopped:

```text
P8
```

was observed.

Conceptually:

```text
P0
 ↓
high-performance state

P8
 ↓
lower-power / idle-oriented state
```

P-state should not be interpreted as GPU utilization.

For example:

```text
P0
```

does not mean:

```text
100% GPU utilization
```

and it does not prove application efficiency.

---

# 10. GPU Utilization Does Not Equal Efficiency

A common mistake is to interpret:

```text
GPU-Util = 100%
```

as:

```text
The workload is using the GPU perfectly.
```

This is incorrect.

GPU utilization primarily tells us whether the GPU was busy during the sampling interval.

It does not directly tell us:

- CUDA core occupancy
- achieved FLOPS
- Tensor Core efficiency
- kernel efficiency
- memory efficiency
- achieved theoretical performance
- application throughput quality

Therefore:

```text
GPU utilization
```

must be combined with:

```text
clocks
power
temperature
memory usage
throughput
latency
CPU activity
```

---

# 11. GPU Clock Monitoring

The following query was used:

```bash
nvidia-smi \
  --query-gpu=pstate,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used \
  --format=csv
```

During heavy computation:

```text
GPU Util:      100%
SM Clock:      1260 MHz
Memory Clock:  7001 MHz
Temperature:   79 C
Power:         ~124.65 W
```

A later sample showed:

```text
GPU Util:      100%
SM Clock:      1170 MHz
Memory Clock:  7001 MHz
Temperature:   82 C
Power:         ~124.81 W
```

The GPU remained fully busy while the SM clock decreased.

This is an important observation:

```text
GPU Utilization
100% → 100%

SM Clock
1260 MHz → 1170 MHz
```

A workload may therefore remain fully busy while completing less work per second.

---

# 12. GPU Temperature

GPU temperature increased during sustained computation:

```text
79 C
 ↓
82 C
```

After the workload ended:

```text
~72 C
 ↓
~71 C
```

Temperature decreases gradually because the GPU hardware retains heat after computation stops.

Compute activity can stop immediately, but cooling requires time.

---

# 13. GPU Power Consumption

During the workload:

```text
Power Draw ≈ 125 W
```

At idle:

```text
Power Draw ≈ 24 W
```

This provides another indicator of active GPU computation.

However, power consumption should not be interpreted independently.

For example:

```text
GPU Util: 100%
Power: high
```

may indicate sustained compute activity.

Whereas:

```text
GPU Util: 0%
Power: low
```

typically indicates an idle workload.

---

# 14. Clock Event Reasons

One of the most important commands explored was:

```bash
nvidia-smi -q -d PERFORMANCE
```

During sustained computation, the system reported:

```text
Performance State: P3

Clocks Event Reasons:

Idle:                     Not Active
Applications Clock:       Not Active
SW Power Cap:             Active

HW Slowdown:              Not Active
HW Thermal Slowdown:      Not Active
HW Power Brake Slowdown:  Not Active

SW Thermal Slowdown:      Active
```

The two key observations were:

```text
SW Power Cap: Active
```

and:

```text
SW Thermal Slowdown: Active
```

This showed that software-managed power and thermal limits were influencing GPU clocks.

---

# 15. Thermal Throttling Diagnosis

Consider the observed behavior:

```text
GPU Utilization:   100% → 100%
Temperature:       79 C → 82 C
SM Clock:          1260 → 1170 MHz
```

By itself, a high temperature does not prove thermal throttling.

However, the additional result:

```text
SW Thermal Slowdown: Active
```

provided evidence that thermal management was actively constraining clocks.

The diagnostic pattern becomes:

```text
Temperature increases
        ↓
SM clock decreases
        ↓
Throughput may decrease
        ↓
SW Thermal Slowdown = Active
```

This is much stronger evidence than simply assuming:

```text
GPU is hot
therefore
GPU is throttling
```

---

# 16. Power-Limiting Diagnosis

During the same workload:

```text
SW Power Cap: Active
```

was observed.

The GPU was also drawing approximately:

```text
124–125 W
```

This indicated that software-controlled power management was limiting GPU clocks.

A useful pattern is:

```text
GPU utilization high
       +
power near configured limit
       +
SW Power Cap active
       +
clock constrained
```

which suggests a power-limited workload.

---

# 17. Performance Event Counters

`nvidia-smi -q -d PERFORMANCE` also reported accumulated counters.

Example:

```text
SW Power Capping
SW Thermal Slowdown
HW Thermal Slowdown
HW Power Braking
```

These counters are historical accumulated values.

They should not be interpreted as:

```text
the current workload has been throttled for exactly this duration
```

The distinction is:

```text
Active / Not Active
        ↓
current state

Counter
        ↓
historical accumulated activity
```

---

# 18. GPU Process Monitoring

GPU usage can be associated with individual processes.

The process table from `nvidia-smi` showed:

```text
Xorg
GNOME Shell
Python
```

GPU compute processes can be queried with:

```bash
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv
```

Example:

```text
PID:           14898
Process:       python3
GPU Memory:    632 MiB
```

---

# 19. Connecting GPU Processes to Linux

The GPU PID can then be investigated through Linux:

```bash
ps -p 14898 -o pid,user,etime,%cpu,%mem,cmd
```

Observed:

```text
PID:       14898
User:      zakaria
Elapsed:   ~1 minute
CPU:       ~4.3%
RAM:       ~1.8%
Command:   python3 -
```

This creates an important troubleshooting path:

```text
nvidia-smi
     ↓
GPU PID
     ↓
ps
     ↓
user
command
CPU usage
RAM usage
runtime
```

---

# 20. GPU Process Types

Typical NVIDIA process types include:

```text
G = graphics workload
C = compute workload
```

Examples from the system:

```text
Xorg          G
GNOME Shell   G
Python        C
```

A `G` workload represents a graphics context and should not simply be interpreted as AI compute.

---

# 21. GPU Starvation

One troubleshooting scenario explored was:

```text
GPU Utilization:    15%
VRAM:               6 / 8 GB
Power:              35 W
CPU:                100%
Request Queue:      increasing
```

The likely class of problem is:

```text
GPU starvation
```

The GPU may not be receiving work quickly enough.

Possible upstream causes include:

```text
CPU preprocessing
data loading
disk I/O
network I/O
CPU-to-GPU transfers
serialization
batch formation
insufficient workers
synchronization
```

The diagnostic chain becomes:

```text
CPU saturated
      ↓
preprocessing slow
      ↓
GPU waits
      ↓
GPU utilization low
      ↓
request queue grows
```

The correct first response is not necessarily:

```text
Buy a faster GPU
```

because the GPU itself may not be the bottleneck.

---

# 22. Healthy GPU Saturation

Another scenario:

```text
GPU Utilization:     99%
VRAM:                6 / 8 GB
Power:               120 / 125 W
Temperature:         74 C
SM Clock:            stable
Thermal Slowdown:    Not Active
Throughput:          stable
```

High utilization alone is not a problem.

If:

```text
throughput is stable
clocks are stable
temperature is acceptable
no throttling is active
```

then high GPU utilization may indicate that the accelerator is being used effectively.

---

# 23. GPU Memory Exhaustion

Another scenario:

```text
GPU Utilization: 45%
VRAM:            7.95 / 8 GB

CUDA out of memory.
Tried to allocate 512 MiB.
```

This is primarily a:

```text
GPU memory capacity problem
```

rather than a compute problem.

The low GPU-utilization number does not contradict the OOM.

These are separate resources:

```text
GPU Utilization
      ↓
compute activity

VRAM
      ↓
memory capacity
```

Possible mitigations include:

- reducing batch size
- FP16
- BF16
- quantization
- activation checkpointing
- reducing sequence length
- using a smaller model
- using a GPU with more VRAM
- distributing a model across multiple GPUs

Importantly:

```text
adding another GPU
```

does not automatically combine the GPUs' memory into one large address space.

The application must explicitly support workload/model distribution.

---

# 24. Busy GPU with Falling Throughput

Consider:

```text
Time       GPU Util    Temperature    SM Clock    Throughput

0 min      100%        68 C           1500 MHz    120/s
5 min      100%        77 C           1450 MHz    117/s
10 min     100%        84 C           1150 MHz     91/s
15 min     100%        87 C            900 MHz     70/s
```

And:

```text
SW Thermal Slowdown: Active
```

Although utilization stays at 100%, throughput falls because the GPU is operating at a lower clock frequency.

This demonstrates:

```text
busy
```

does not necessarily mean:

```text
fast
```

---

# 25. Troubleshooting Decision Tree

A useful high-level workflow is:

```text
                    Application slow
                           |
                           v
                      nvidia-smi
                           |
                +----------+----------+
                |                     |
          GPU utilization          GPU utilization
               low                     high
                |                       |
                v                       v
        Is GPU receiving work?      Check clocks
                |                   power / temp
       +--------+--------+               |
       |        |        |               |
      CPU      I/O    transfer         P-state
       |        |        |               |
       v        v        v               v
   preprocess storage CPU→GPU       Event reasons
```

VRAM should be checked independently:

```text
Memory usage
     |
     +------ normal
     |
     +------ nearly full
                |
                v
             OOM risk
```

Processes should also be inspected:

```text
nvidia-smi
     ↓
PID
     ↓
ps
     ↓
identify workload
```

---

# 26. Practical GPU Triage Workflow

When an AI application is slow:

## Step 1 — Verify the GPU

```bash
nvidia-smi
```

If the GPU is missing entirely, investigate:

```text
driver
kernel module
hardware
device permissions
container GPU exposure
```

---

## Step 2 — Check Utilization

Look at:

```text
GPU-Util
```

Low utilization while work is queued may indicate:

```text
CPU bottleneck
I/O bottleneck
data loading issue
transfer problem
poor batching
```

---

## Step 3 — Check VRAM

Look at:

```text
Memory-Usage
```

If near capacity, inspect application logs for:

```text
CUDA out of memory
```

---

## Step 4 — Check Clocks

Inspect:

```text
P-State
SM Clock
Memory Clock
```

Unexpected clock reduction can explain falling throughput.

---

## Step 5 — Check Power and Temperature

Inspect:

```text
Power Draw
Temperature
```

---

## Step 6 — Check Clock Event Reasons

```bash
nvidia-smi -q -d PERFORMANCE
```

Look for:

```text
SW Power Cap
SW Thermal Slowdown
HW Thermal Slowdown
HW Power Brake
Idle
```

---

## Step 7 — Identify Processes

```bash
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv
```

Then:

```bash
ps -p <PID> -o pid,user,etime,%cpu,%mem,cmd
```

Before terminating anything, determine:

```text
Who owns the process?
What workload is it?
How long has it been running?
Is it production?
Is it checkpointing?
Is it causing contention?
```

---

# 27. Key Troubleshooting Patterns

## Pattern A — Model loaded but idle

```text
VRAM:       HIGH
GPU Util:   LOW
Power:      LOW
```

Likely interpretation:

```text
model or tensors allocated
but little current computation
```

---

## Pattern B — Compute saturation

```text
GPU Util:   HIGH
Power:      HIGH
Clocks:     stable
Throughput: stable
```

Potentially healthy.

---

## Pattern C — Upstream starvation

```text
GPU Util:   LOW
CPU:        HIGH
Queue:      growing
```

Investigate:

```text
preprocessing
data loading
I/O
transfers
batching
parallelism
```

---

## Pattern D — Memory bottleneck

```text
VRAM:      nearly full
CUDA OOM:  present
```

Primary issue:

```text
GPU memory capacity
```

---

## Pattern E — Thermal constraint

```text
GPU Util:               HIGH
Temperature:            increasing
SM Clock:               decreasing
Throughput:             decreasing
SW Thermal Slowdown:    Active
```

---

## Pattern F — Power constraint

```text
GPU Util:       HIGH
Power:          near limit
SW Power Cap:   Active
SM Clock:       constrained
```

---

# 28. Interview-Level Takeaways

A few important questions that can be answered after this lab:

### Can a GPU have high VRAM usage but 0% utilization?

Yes.

Memory can remain allocated even when no GPU kernels are actively executing.

---

### Does 100% GPU utilization mean the GPU is optimally used?

No.

It only indicates that the GPU remained busy during the sampling period.

Performance could still be limited by:

```text
inefficient kernels
low clocks
power caps
thermal throttling
memory behavior
synchronization
```

---

### Can a CUDA OOM occur with low GPU utilization?

Yes.

OOM is caused by insufficient GPU memory capacity, not necessarily high compute activity.

---

### Can throughput fall while GPU utilization remains 100%?

Yes.

If clocks decrease due to power or thermal constraints, the GPU can remain continuously busy while executing fewer operations per second.

---

### Why inspect GPU PIDs?

Because identifying the process lets infrastructure engineers determine:

```text
who owns the workload
what application is running
how long it has run
whether it is expected
whether it is causing contention
```

---

# 29. Commands Learned

```bash
nvidia-smi
```

```bash
nvidia-smi --list-gpus
```

```bash
watch -n 1 nvidia-smi
```

```bash
nvidia-smi dmon
```

```bash
nvidia-smi pmon
```

```bash
nvidia-smi \
  --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
  --format=csv
```

```bash
nvidia-smi \
  --query-gpu=timestamp,pstate,utilization.gpu,memory.used,memory.free,clocks.sm,clocks.mem,temperature.gpu,power.draw \
  --format=csv \
  -l 1
```

```bash
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv
```

```bash
nvidia-smi -q -d PERFORMANCE
```

```bash
ps -p <PID> -o pid,user,etime,%cpu,%mem,cmd
```

```bash
nvcc --version
```

---

# 30. Repository Structure

```text
day23-gpu-monitoring/
├── README.md
├── gpu_workload.py
└── monitor_gpu.sh
```

`gpu_workload.py`

Generates a sustained PyTorch CUDA workload for observing:

```text
GPU utilization
VRAM allocation
power
temperature
clocks
P-state
```

`monitor_gpu.sh`

Continuously collects key GPU telemetry.

---

# Key Takeaway

GPU monitoring is fundamentally about **correlating multiple signals**.

A single metric rarely tells the whole story.

The core mental model from Day 23 is:

```text
                 GPU performance
                       |
        +--------------+--------------+
        |              |              |
      Compute        Memory         Operating
        |              |             State
        |              |              |
    GPU Util         VRAM       clocks / power /
                               temperature / P-state
        \              |              /
         \             |             /
          +------------+------------+
                       |
                  Processes
                       |
                       v
               Application behavior
```

The most important lesson is:

> High GPU utilization does not automatically mean high performance, high VRAM allocation does not mean active computation, and GPU performance problems should be diagnosed by correlating utilization, memory, clocks, power, temperature, process ownership, and application-level throughput.

---

# Next

## Day 24 — GPU Memory Allocation and OOM Diagnosis

The next lab will go deeper into:

- GPU memory allocation
- PyTorch memory behavior
- allocated vs reserved memory
- CUDA caching allocator
- fragmentation
- tensor memory calculations
- batch-size effects
- intentional OOM experiments
- diagnosing and mitigating CUDA OOM failures
