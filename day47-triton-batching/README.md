# Day 47 — Triton Dynamic Batching & Model Concurrency

**AI Infrastructure / ML Systems — 220-Day Learning Roadmap**  
**Focus:** NVIDIA Triton Inference Server · ONNX Runtime · Dynamic batching · Model instances · Inference benchmarking

## Project overview

This lab explores how [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) handles multiple independent inference requests. I extended the two-version ONNX model repository from Day 46 to accept a dynamic batch dimension, enabled Triton's server-side batching, and compared execution with one and two **CPU model instances**.

Three configurations were tested using the **same model and client workload**:

1. **No dynamic batching**, one CPU model instance.
2. **Dynamic batching**, one CPU model instance.
3. **Dynamic batching**, two CPU model instances.

**Main result:** Dynamic batching combined 1,600 single-sample requests into just 200 model executions (full batches of eight), but it **did not produce an eightfold throughput improvement**. For this very small CPU-backed model, differences in throughput and latency were modest and varied across runs.

> **Scope:** These are local, CPU-only functional and short-duration performance tests. They are **not GPU benchmarks** and do not establish production capacity or a statistically reliable optimal configuration.

## What this lab demonstrates

- Build and validate ONNX models with a **symbolic leading batch dimension**.
- Use Triton's `config.pbtxt` to configure batching and model-instance concurrency.
- Distinguish **client-side batching** from **server-side dynamic batching**.
- Verify batching from Triton's **inference count**, **execution count**, and **batch-size statistics**.
- Measure throughput, client-side mean/p95 latency, and Triton queue time.
- Compare configurations and discuss their latency/throughput trade-offs.

## Architecture

```text
16 concurrent Python HTTP clients
              |
              v
      Triton HTTP endpoint (:8000)
              |
              v
       Model: toy_classifier
       Version: 1 (tested)
              |
       [Dynamic batcher]           <-- present only in B and C
       max_batch_size = 8
       max_queue_delay = 2 ms
              |
      +-------+-------+
      |               |
  CPU instance 0   CPU instance 1  <-- second instance only in C
      |               |
      +-------+-------+
              |
           ONNX Runtime
              |
       Output: [batch, 2]

Triton metrics (:8002) and model statistics API
```

Both model versions (1 and 2) were loaded in the server. All benchmark requests explicitly targeted **version 1**.

## Model and repository

The toy classifier performs a matrix multiplication:

```text
INPUT [batch, 4]  x  W [4, 2]  ->  OUTPUT [batch, 2]
```

The ONNX graph uses `MatMul` and a symbolic batch dimension (`"batch"`). The two versions use the same matrix pattern with different weights: version 2 scales the version 1 weights by two.

| Model version | Example input | Expected output |
|---|---|---|
| 1 | `[1, 2, 3, 4]` | `[4, 6]` |
| 2 | `[1, 2, 3, 4]` | `[8, 12]` |

**Why the tensor shapes matter:** The ONNX graph must accept batched tensors. Setting `max_batch_size` in Triton alone cannot make a fixed-shape, single-sample model batch-compatible. Triton's `input.dims: [4]` and `output.dims: [2]` describe the **per-sample** shapes when `max_batch_size > 0`.

### Project layout

```text
day47-triton-batching/
├── README.md
├── build_models.py
├── test_dynamic_batching.py
├── model_repository/
│   └── toy_classifier/
│       ├── config.pbtxt            # Last lab state: batching disabled, 1 instance
│       ├── 1/
│       │   └── model.onnx
│       └── 2/
│           └── model.onnx
├── config.one-instance.pbtxt       # Dynamic batching ON, 1 CPU instance
├── config.two-instances.pbtxt      # Dynamic batching ON, 2 CPU instances
├── baseline-1.txt                  # Dynamic batching ON, 1 instance
├── baseline-2.txt
├── baseline-3.txt
├── two-instance-1.txt              # Dynamic batching ON, 2 instances
├── two-instance-2.txt
├── two-instance-3.txt
├── no-batching-1.txt               # Dynamic batching OFF, 1 instance
├── no-batching-2.txt
├── no-batching-3.txt
└── requirements-dev.txt            # Optional snapshot of the shared dev environment
```

This tree describes the experiment files. Ensure the named configuration and benchmark output files are included when publishing. `requirements-dev.txt` was created with `pip freeze` from a shared environment, so it may contain packages unrelated to this lab.

## Requirements

- Linux with Docker installed and the ability to pull the Triton server image.
- Python 3 for model generation and the benchmarking script.
- `numpy` and `onnx` for `build_models.py`.
- Triton image: `nvcr.io/nvidia/tritonserver:25.08-py3` (the server reported Triton **2.60.0**).
- **No GPU is required:** the tested model uses `KIND_CPU`.

```bash
# In an existing Python virtual environment:
python -m pip install numpy onnx

# Generate both ONNX versions
python build_models.py
```

The `test_dynamic_batching.py` client uses Python's standard library for HTTP requests, concurrency, and timing; no Triton Python client package is necessary for this particular test.

### Confirm the ONNX model's batch dimensions

```bash
python - <<'PY'
import onnx

for version in (1, 2):
    path = f"model_repository/toy_classifier/{version}/model.onnx"
    model = onnx.load(path)
    onnx.checker.check_model(model)
    for tensor in list(model.graph.input) + list(model.graph.output):
        dims = [d.dim_param or d.dim_value
                for d in tensor.type.tensor_type.shape.dim]
        print(f"Version {version}: {tensor.name} = {dims}")
PY
```

Expected shapes for both versions:

```text
INPUT  = ['batch', 4]
OUTPUT = ['batch', 2]
```

## Triton configurations

The **shared** parts of `config.pbtxt` are:

```protobuf
name: "toy_classifier"
backend: "onnxruntime"
max_batch_size: 8

input [
  { name: "INPUT" data_type: TYPE_FP32 dims: [4] }
]
output [
  { name: "OUTPUT" data_type: TYPE_FP32 dims: [2] }
]

version_policy: { latest: { num_versions: 2 } }
```

| Setting | A: No batching | B: Batched, 1 instance | C: Batched, 2 instances |
|---|---|---|---|
| `max_batch_size` | 8 | 8 | 8 |
| `dynamic_batching` | Absent | Enabled | Enabled |
| `max_queue_delay_microseconds` | — | 2000 | 2000 |
| `instance_group.count` | 1 | 1 | 2 |
| `instance_group.kind` | `KIND_CPU` | `KIND_CPU` | `KIND_CPU` |

For configurations B and C, add:

```protobuf
dynamic_batching {
  max_queue_delay_microseconds: 2000
}
```

For **one CPU instance** (A and B):

```protobuf
instance_group [
  { count: 1 kind: KIND_CPU }
]
```

For **two CPU instances** (C):

```protobuf
instance_group [
  { count: 2 kind: KIND_CPU }
]
```

**Important distinctions:**

- `max_batch_size: 8` sets the maximum **number of samples per execution**, not the number of model instances.
- Without `dynamic_batching`, the model can still accept a client-submitted batch, but Triton does not automatically combine the independent single-sample requests used in this lab.
- `max_queue_delay_microseconds: 2000` allows **up to** 2 ms of additional batching delay; it is not a mandatory 2 ms wait for every request.
- `instance_group.count: 2` creates **two instances for each loaded model version**. It does not require two GPUs, and it does not guarantee a speedup.

## Start the server and verify inference

From the directory containing `model_repository/`:

```bash
docker run -d --rm \
  --name day47-triton \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$PWD/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:25.08-py3 \
  tritonserver --model-repository=/models

# Confirm readiness (HTTP 200)
curl -i http://localhost:8000/v2/health/ready

# Verify both model versions are READY
docker logs --tail 80 day47-triton
```

A version 1 inference smoke test:

```bash
curl -sS -X POST \
  http://localhost:8000/v2/models/toy_classifier/versions/1/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [{
      "name": "INPUT", "shape": [1, 4],
      "datatype": "FP32", "data": [1.0, 2.0, 3.0, 4.0]
    }],
    "outputs": [{"name": "OUTPUT"}]
  }' | python -m json.tool
```

Observed response contained `"shape": [1, 2]` and `"data": [4.0, 6.0]`. Triton reported `READY` for both model versions and HTTP `200 OK` for the readiness endpoint.

Model inference statistics:

```bash
curl -sS http://localhost:8000/v2/models/toy_classifier/versions/1/stats \
  | python -m json.tool
```

## Benchmark procedure

The `test_dynamic_batching.py` script submits **independent HTTP requests**, each containing **one** input sample (`shape: [1, 4]`), using `ThreadPoolExecutor` and a barrier to start clients together.

Test settings:

```python
CONCURRENCY = 16
ROUNDS = 100
# 16 x 100 = 1,600 successful requests per run
```

The script measures elapsed client time, individual request latency, throughput, and p95 latency. It also takes **before/after snapshots of Triton's model-version statistics** to calculate how many inference samples and model executions occurred during each run. Each configuration was tested **three times**.

### Reproduce the three configurations

Configurations can be changed by copying a config file into the model repository and **restarting the container** (the lab used Triton's default model-control mode). Run the next commands from the project directory.

**B — Dynamic batching, one instance:**

```bash
cp config.one-instance.pbtxt model_repository/toy_classifier/config.pbtxt
# Stop the previous container, if running:
docker stop day47-triton
# Re-run the docker run command from "Start the server" above; verify readiness
for n in 1 2 3; do
  python test_dynamic_batching.py | tee "baseline-$n.txt"
done
```

**C — Dynamic batching, two instances:**

```bash
cp config.two-instances.pbtxt model_repository/toy_classifier/config.pbtxt
docker stop day47-triton
# Re-run the docker run command above; verify readiness
for n in 1 2 3; do
  python test_dynamic_batching.py | tee "two-instance-$n.txt"
done
```

**A — No dynamic batching, one instance:** Start with `config.one-instance.pbtxt`, remove the entire `dynamic_batching { ... }` block from a copy, and use that as `model_repository/toy_classifier/config.pbtxt`. Restart the container, then:

```bash
for n in 1 2 3; do
  python test_dynamic_batching.py | tee "no-batching-$n.txt"
done
```

> Re-run the container after each config change. The benchmark outputs below are the actual values obtained during this lab, not promised results on another machine.

## Measured results

Each repetition had 1,600 successful requests. All reported latency statistics below are **client-side** except for **Triton queue**.

| Configuration | Run | Throughput (req/s) | Mean latency (ms) | p95 latency (ms) | Mean Triton queue (ms) | Executions | Mean batch size |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — No batching, 1 instance | 1 | 2,400.40 | 4.20 | 7.20 | 0.03 | 1,600 | 1.00 |
| A — No batching, 1 instance | 2 | 2,738.75 | 3.62 | 5.24 | 0.02 | 1,600 | 1.00 |
| A — No batching, 1 instance | 3 | 2,983.80 | 3.31 | 4.74 | 0.02 | 1,600 | 1.00 |
| B — Batching, 1 instance | 1 | 2,491.65 | 4.20 | 5.98 | 0.56 | 200 | 8.00 |
| B — Batching, 1 instance | 2 | 2,648.86 | 3.96 | 5.45 | 0.52 | 200 | 8.00 |
| B — Batching, 1 instance | 3 | 2,689.85 | 3.89 | 5.48 | 0.54 | 200 | 8.00 |
| C — Batching, 2 instances | 1 | 2,338.40 | 4.50 | 6.88 | 0.60 | 210 | 7.62 |
| C — Batching, 2 instances | 2 | 2,815.92 | 3.71 | 5.07 | 0.51 | 200 | 8.00 |
| C — Batching, 2 instances | 3 | 2,945.54 | 3.56 | 4.79 | 0.47 | 200 | 8.00 |

### Median results (three repetitions each)

| Metric | A: No batching | B: Batching + 1 instance | C: Batching + 2 instances |
|---|---:|---:|---:|
| Throughput (req/s) | 2,738.75 | 2,648.86 | **2,815.92** |
| Mean client latency (ms) | **3.62** | 3.96 | 3.71 |
| Client p95 latency (ms) | 5.24 | 5.48 | **5.07** |
| Mean server queue (ms) | **0.02** | 0.54 | 0.51 |
| Average batch size | 1.00 | 8.00 | 8.00 |

*These are medians of per-run metrics, not a pooled p95 latency computed across all runs.*

### What the statistics mean

For configuration **B**:

```text
1,600 successful requests
1,600 inference samples
  200 model executions
    8 samples per model execution, on average
```

So dynamic batching reduced the **number of model executions** from 1,600 (A) to 200 (B): an **87.5% reduction**. It did **not** reduce total application work by 87.5%, nor improve throughput eightfold.

For configuration **C**, one repetition had 210 executions (mean batch size 7.62), rather than 200, because not every dispatched batch was full. Subsequent runs formed 200 full batches.

### Interpretation

1. **Batching was functionally verified.** In every B run, Triton combined separate single-sample requests into full batches of eight; in A, each request executed separately.
2. **Concurrency was functionally verified.** Server logs showed two CPU instances initialized **for each** of the two loaded model versions in C.
3. **More batching did not mean proportionately greater throughput.** This model's `MatMul` is extremely cheap. Request transport, Python scheduling, Triton scheduling, output handling, and queueing matter relative to compute time.
4. **The highest median throughput was C** (2,815.92 req/s), about 6.3% above B's median. But B's median throughput was about 3.3% **below** A's. Variation across repetitions means these results do not establish a consistently superior configuration.
5. **Queueing increased with batching.** Median mean queue duration was ~0.5 ms with dynamic batching versus ~0.02 ms without it. The 2 ms configured delay is a ceiling, not a fixed wait.
6. **Latency SLOs matter.** For a hypothetical *strict 5 ms p95* SLO, no configuration **consistently** met the target across runs; selecting the run with the largest throughput would be unjustified.

## Limitations and next experiments

- **CPU-backed toy model:** The model is not representative of GPU-heavy production inference. No GPU speedup or saturation claims should be derived from these results.
- **Short tests:** Each repetition lasted less than a second, making start-up effects, host contention, and Python client scheduling important. These are not steady-state capacity measurements.
- **Client and server shared a host:** HTTP/Python overhead can obscure small model-execution savings.
- **Only one concurrency setting:** Tests used 16 simultaneous clients; behavior at other load levels is unknown.
- **No resource profiling:** CPU usage, memory usage, hardware utilization, and tail latency under sustained load were not recorded.
- **CUDA warnings:** The Triton container logged CUDA driver/runtime warnings, but CPU inference remained healthy. GPU backend compatibility was not validated.

Suggested improvements for a future GPU-serving/load-testing lab:

1. Benchmark for longer durations with warmup and repeated, controlled load levels (e.g. 1, 4, 8, 16, 32 clients).
2. Compare batch sizes and max batching delays while observing the *actual* batch-size distribution.
3. Capture p50/p95/p99 latency, queue duration, execution duration, throughput, CPU/GPU utilization, and memory pressure.
4. Use a more substantial GPU-supported model and ensure the CUDA runtime/driver environment is functioning before claiming GPU results.
5. Choose settings that satisfy a measured service-level objective (SLO), not simply those yielding the fewest model executions.

## Troubleshooting notes

| Symptom | What to check |
|---|---|
| Model fails to load with batching enabled | Verify the ONNX graph has `[batch, 4]` input and `[batch, 2]` output, and the Triton config uses per-sample `dims`. |
| Model returns wrong shape | Verify the request includes `[1, 4]` (or another valid batched shape), the requested model version, and output dimensions. |
| Execution count equals request count | Confirm `dynamic_batching` is enabled, restart the server after config changes, and ensure clients send simultaneous compatible requests. |
| Extra model instances provide no speedup | Inspect CPU/GPU contention, queueing, actual batch sizes, and sustained utilization. |
| `pip` cannot import `onnx` | Activate the development virtual environment and install `onnx` there. |
| CUDA warnings during CPU experiments | Verify the model uses `KIND_CPU`; troubleshoot the GPU driver/runtime separately for future GPU tests. |

## Key takeaway

**Dynamic batching** joins compatible independent inference requests into fewer, larger executions; **model-instance concurrency** provides additional model execution instances. These are separate tuning controls. For a tiny CPU model, fewer executions did not translate into a proportional throughput increase, and queueing added latency. A production inference configuration should be selected using sustained workload measurements—especially p95/p99 latency, throughput, queue duration, instance utilization, memory pressure, and actual batch sizes—against a clearly defined SLO.

---

**Roadmap:** Day 47 of 220 · **Next:** Day 48 — vLLM Fundamentals.
