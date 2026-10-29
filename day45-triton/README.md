# Day 45 — NVIDIA Triton Inference Server

**AI Infrastructure & ML Systems — 220-Day Roadmap | Phase 3: Model Serving**

This lab deploys NVIDIA Triton Inference Server in Docker, serves a small CPU-backed model using Triton's Python backend, tests inference over HTTP, inspects Prometheus metrics, and diagnoses a deliberately malformed request.

> **Scope:** This is a CPU inference and serving-infrastructure lab. GPU-backed inference, dynamic batching, and production Kubernetes deployment are **not** demonstrated here.

## Learning objectives

- Understand Triton's HTTP/gRPC interfaces, model repository, scheduler, and backend responsibilities.
- Run a Triton container and load a versioned model.
- Verify server/model readiness separately from inference correctness.
- Send inference requests and inspect Prometheus counters.
- Distinguish client request errors from model-loading or server failures.

## Architecture

```text
Client (curl)
    |
    | HTTP :8000
    v
Triton Inference Server (Docker)
    |-- HTTP inference API :8000
    |-- gRPC inference API :8001
    |-- Prometheus metrics :8002
    |
    |-- Model repository: /models
    |      `-- add_one/1/model.py + config.pbtxt
    |
    `-- Python backend --> add_one model --> CPU
```

Triton's scheduler dispatches requests to a model instance, and the model backend implements the execution. In this example, `max_batch_size: 0` disables Triton-level batching.

## Prerequisites

- Linux machine with Docker installed and running.
- Network access to pull the Triton container image from NVIDIA NGC.
- `curl` and Python 3 for inspecting formatted JSON.
- Sufficient disk space for the Triton image.

**Environment observed in this lab:** Docker `29.9.2`; container image `nvcr.io/nvidia/tritonserver:26.08-py3`; Triton server `2.72.0` (as reported by server logs). Exact versions may differ on future reruns.

## Repository layout

```text
day45-triton/
├── model_repository/
│   └── add_one/
│       ├── config.pbtxt
│       └── 1/
│           └── model.py
├── evidence/                 # optional, generated locally
│   ├── triton-logs.txt
│   └── inference-metrics.txt
└── README.md
```

## 1. Create the model repository

From the root of `day45-triton/`:

```bash
mkdir -p model_repository/add_one/1
```

Create `model_repository/add_one/1/model.py`:

```python
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []

        for request in requests:
            x = pb_utils.get_input_tensor_by_name(
                request, "INPUT0"
            ).as_numpy()
            y = (x + 1).astype(np.float32)
            output = pb_utils.Tensor("OUTPUT0", y)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output])
            )

        return responses
```

Create `model_repository/add_one/config.pbtxt`:

```protobuf
name: "add_one"
backend: "python"
max_batch_size: 0

input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [4]
  }
]

output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [4]
  }
]

instance_group [
  {
    kind: KIND_CPU
    count: 1
  }
]
```

The model expects exactly four `FP32` values per request and returns each value incremented by one.

## 2. Start Triton

From the `day45-triton/` directory (the bind mount assumes the current directory contains `model_repository/`):

```bash
docker run --rm -d \
  --name day45-triton \
  --shm-size=1g \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$PWD/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:26.08-py3 \
  tritonserver --model-repository=/models
```

Check startup and readiness:

```bash
docker ps --filter name=day45-triton
docker logs day45-triton --tail 50
curl -i localhost:8000/v2/health/live
curl -i localhost:8000/v2/health/ready
curl -i localhost:8000/v2/models/add_one/ready
```

In the completed lab, Triton initialized its Python backend, loaded `add_one` version `1` with status `READY`, and returned **HTTP 200** for server and model readiness (the server liveness endpoint is included above for repeatable testing, but its response was not separately captured in the recorded lab).

The logs reported `model_control_mode: MODE_NONE` and `strict_readiness: 1`. Under `MODE_NONE`, changes to the model repository are not automatically picked up after startup.

## 3. Send an inference request

```bash
curl -sS -X POST \
  http://localhost:8000/v2/models/add_one/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "INPUT0",
        "shape": [4],
        "datatype": "FP32",
        "data": [1.0, 2.0, 3.0, 4.0]
      }
    ]
  }' | python3 -m json.tool
```

**Observed response:**

```json
{
  "model_name": "add_one",
  "model_version": "1",
  "outputs": [
    {
      "name": "OUTPUT0",
      "datatype": "FP32",
      "shape": [4],
      "data": [2.0, 3.0, 4.0, 5.0]
    }
  ]
}
```

This demonstrates functional inference, not just server availability.

## 4. Inspect inference metrics

```bash
curl -s localhost:8002/metrics \
  | grep -E '^nv_inference_(request_success|request_failure|count|exec_count)\{'
```

**Observed after one valid request and one invalid-shape request:**

```text
nv_inference_request_success{model="add_one",version="1"} 1
nv_inference_request_failure{model="add_one",reason="BACKEND",version="1"} 0
nv_inference_request_failure{model="add_one",reason="OTHER",version="1"} 0
nv_inference_request_failure{model="add_one",reason="CANCELED",version="1"} 0
nv_inference_request_failure{model="add_one",reason="REJECTED",version="1"} 0
nv_inference_count{model="add_one",version="1"} 1
nv_inference_exec_count{model="add_one",version="1"} 1
```

Metric interpretation:

| Metric | Meaning in this lab |
| --- | --- |
| `nv_inference_request_success` | Successful inference requests, here `1`. |
| `nv_inference_count` | Individual inferences executed, here `1` (the four numbers are a **single input tensor**, not four requests). |
| `nv_inference_exec_count` | Backend execution batches, here `1`. |
| `nv_inference_request_failure` | Per-model failure counters, all `0` in the captured output. |

**Interpretation caveat:** An HTTP 400 does not necessarily increment the per-model failure counters. The invalid-shape request in this experiment was rejected during request validation, before backend execution. HTTP status monitoring and Triton's model counters therefore answer different operational questions.

## 5. Break/fix exercise — invalid tensor shape

The model requires shape `[4]`. Deliberately send shape `[3]`:

```bash
curl -sS -i -X POST \
  http://localhost:8000/v2/models/add_one/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "INPUT0",
        "shape": [3],
        "datatype": "FP32",
        "data": [10.0, 20.0, 30.0]
      }
    ]
  }'
```

**Observed:** `HTTP/1.1 400 Bad Request` with an error indicating `Expected [4], got [3]`.

**Follow-up:** `curl -i localhost:8000/v2/models/add_one/ready` still returned **HTTP 200**. The model remained ready, and the inference execution count stayed at `1`.

**Root cause:** The client sent a tensor shape that violated the model interface. **Fix:** Correct the input to `[4]` (or redesign the model/configuration if variable-length input is required). Restarting Triton is not the appropriate first response to this client error.

### Troubleshooting decision guide

| Symptom | Check first |
| --- | --- |
| HTTP 400 on `/infer` | Request JSON, input names, datatypes, shapes, and model configuration. |
| Model `/ready` fails | Server logs, model loading status, repository paths, backend compatibility. |
| Server `/live` fails | Container/process state and startup logs. |
| Metrics do not reflect a rejected request | Whether failure occurred before model inference accounting; inspect HTTP responses and access-level telemetry. |
| GPU-related warnings during CPU run | Whether GPU access was enabled, driver/runtime compatibility, and whether the model actually needs CUDA. |

## 6. CUDA warnings observed

The Triton startup logs included:

```text
CUDA memory pool disabled
Unable to allocate pinned system memory ... CUDA driver version is insufficient for CUDA runtime version
CudaDriverHelper has not been initialized.
```

The model nevertheless loaded as `add_one_0_0 (CPU device 0)` and served requests successfully. This run **did not** use `--gpus all` and therefore did **not** validate GPU inference. A GPU-serving follow-up must separately check NVIDIA drivers, container GPU exposure, toolkit/runtime compatibility, and a GPU model execution.

## 7. Save lab evidence

```bash
mkdir -p evidence
docker logs day45-triton > evidence/triton-logs.txt 2>&1
curl -s localhost:8002/metrics > evidence/inference-metrics.txt
```

Review logs and metrics for machine-specific or sensitive information before committing them to a public repository. The observations above are from the completed Day 45 lab; future runs may produce different counts and startup messages.

## 8. Stop the container

```bash
docker stop day45-triton
```

The `--rm` option removes the container after it stops; files in the bind-mounted local model repository remain on disk.

## Results and next steps

**Completed:** Docker deployment, model load/readiness checks, correct HTTP inference, Prometheus metric inspection, and one controlled client-input failure with root-cause analysis.

**Not yet implemented:** GPU-backed Triton inference, version switching/live model control, dynamic batching, concurrency benchmarking, Kubernetes deployment, latency baselines, and a monitoring dashboard.

- **Day 46:** Explore model repositories, model versions, and model lifecycle management.
- **Day 47:** Configure batching and concurrency and compare request counts with backend execution counts.
- **Later portfolio integration:** Connect Triton to the existing FastAPI/Kubernetes GPU-ML service and collect real latency and throughput measurements.

## References

- [NVIDIA Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- [Triton Python backend repository](https://github.com/triton-inference-server/python_backend)
- [Triton metrics documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html)
