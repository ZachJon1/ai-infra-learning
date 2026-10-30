# Day 46 — NVIDIA Triton Model Repository

A reproducible hands-on lab for organizing, configuring, versioning, serving, and troubleshooting ONNX models with **NVIDIA Triton Inference Server**.

**Learning track:** AI Infrastructure & ML Systems — 220-Day Roadmap  
**Phase:** Model Serving / Inference Infrastructure  
**Day:** 46 of 220  
**Lab status:** Completed — both model versions served successfully; a backend failure was reproduced and recovered.

## Objectives

- Understand Triton's model repository directory conventions.
- Configure an ONNX model through `config.pbtxt`.
- Serve multiple versions of one model using a version policy.
- Check server and model-version readiness independently.
- Send version-specific inference requests over HTTP.
- Reproduce, diagnose, and repair a model-loading failure.

## Architecture

```text
                         HTTP client (curl)
                                 |
                                 v
                  Triton Inference Server (Docker)
                      HTTP API :8000
                      gRPC API :8001
                      Metrics  :8002
                                 |
                     model repository mount
                          /models (read-only)
                                 |
                   model_repository/toy_classifier/
                         |            |
                 config.pbtxt         +-- 1/model.onnx
                                      +-- 2/model.onnx
                                 |
                      ONNX Runtime backend
                           CPU instances
```

Both model versions share the same input/output interface but have different weight values. This lab uses **CPU inference** (`KIND_CPU`); it does **not** test GPU inference or batching performance.

## Repository layout

```text
day46-triton-repository/
├── README.md
├── build_models.py
├── request.json
├── model_repository/
│   └── toy_classifier/
│       ├── config.pbtxt
│       ├── 1/
│       │   └── model.onnx
│       └── 2/
│           └── model.onnx
└── .venv/                 # Local only; do not commit
```

The `1/` and `2/` directories are **numeric model versions**. The `config.pbtxt` file describes the model's backend, tensor signatures, execution placement, and version-selection policy.

## Environment used

| Component | Observed/tested value |
| --- | --- |
| Host | Ubuntu Linux |
| Docker | `29.9.2` |
| Triton container | `nvcr.io/nvidia/tritonserver:25.08-py3` |
| Triton server version (logs) | `2.60.0` |
| Model backend | `onnxruntime` |
| Execution device | CPU (`KIND_CPU`) |
| Model format | ONNX |
| API | HTTP / KServe v2 inference protocol |

The Triton image can emit CUDA/driver warnings when launched without GPU access. In this lab, CPU model loading and inference succeeded despite those warnings. GPU support and driver compatibility must be checked separately before running `KIND_GPU` models.

## Prerequisites

- Docker installed and running.
- Python 3 with `venv` support.
- Network access to pull the Triton container image (initial run).
- Available local ports `8000`, `8001`, and `8002`.

> **Note:** The Triton container is fairly large. A GPU is **not required** for this CPU-backed exercise.

## 1. Build the model repository

Start in the project directory:

```bash
mkdir -p ~/ai-infra/day46-triton-repository
cd ~/ai-infra/day46-triton-repository
mkdir -p model_repository/toy_classifier/{1,2}

python3 -m venv .venv
source .venv/bin/activate
pip install numpy onnx
```

Create `build_models.py`:

```python
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

base = Path("model_repository/toy_classifier")

for version in (1, 2):
    weights = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32) * version

    node = helper.make_node(
        "MatMul", inputs=["INPUT", "W"], outputs=["OUTPUT"]
    )

    graph = helper.make_graph(
        [node],
        f"toy_classifier_v{version}",
        [helper.make_tensor_value_info("INPUT", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("OUTPUT", TensorProto.FLOAT, [2])],
        [numpy_helper.from_array(weights, name="W")],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)

    destination = base / str(version) / "model.onnx"
    onnx.save(model, str(destination))
    print(f"Created {destination}")
```

Generate and inspect the two ONNX artifacts:

```bash
python build_models.py
find model_repository -type f | sort
```

Observed result:

```text
Created model_repository/toy_classifier/1/model.onnx
Created model_repository/toy_classifier/2/model.onnx

model_repository/toy_classifier/1/model.onnx
model_repository/toy_classifier/2/model.onnx
model_repository/toy_classifier/config.pbtxt
```

The final `config.pbtxt` appears after completing the next step.

## 2. Configure the Triton model

Create `model_repository/toy_classifier/config.pbtxt`:

```protobuf
name: "toy_classifier"
backend: "onnxruntime"

max_batch_size: 0

input [
  {
    name: "INPUT"
    data_type: TYPE_FP32
    dims: [4]
  }
]

output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [2]
  }
]

version_policy: {
  latest: {
    num_versions: 2
  }
}

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```

**Key settings:**

| Field | Meaning |
| --- | --- |
| `backend: "onnxruntime"` | Use the ONNX Runtime backend. |
| `max_batch_size: 0` | Disable Triton's batch dimension for this model. |
| `input` | Require four FP32 input values named `INPUT`. |
| `output` | Produce two FP32 output values named `OUTPUT`. |
| `latest: { num_versions: 2 }` | Make the two highest-numbered model versions eligible to load. |
| `instance_group` | Create one model execution instance on CPU. |

Both model artifacts must match the input/output names, datatypes, and shapes declared here.

## 3. Launch Triton in Docker

From `~/ai-infra/day46-triton-repository`, run:

```bash
docker run -d --rm \
  --name day46-triton \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$PWD/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:25.08-py3 \
  tritonserver --model-repository=/models
```

| Port | Purpose |
| --- | --- |
| `8000` | HTTP inference and management endpoints |
| `8001` | gRPC inference API |
| `8002` | Metrics endpoint |

The bind mount makes the host repository visible inside the container at `/models` in **read-only** mode. Stop or rename any previous container occupying these ports before launching.

Inspect the model status:

```bash
docker logs day46-triton --tail 60
```

**Observed model-status table:**

```text
+----------------+---------+--------+
| Model          | Version | Status |
+----------------+---------+--------+
| toy_classifier | 1       | READY  |
| toy_classifier | 2       | READY  |
+----------------+---------+--------+
```

## 4. Verify server and model readiness

```bash
curl -s -o /dev/null -w "Server: HTTP %{http_code}\n" \
  localhost:8000/v2/health/ready

curl -s -o /dev/null -w "Version 1: HTTP %{http_code}\n" \
  localhost:8000/v2/models/toy_classifier/versions/1/ready

curl -s -o /dev/null -w "Version 2: HTTP %{http_code}\n" \
  localhost:8000/v2/models/toy_classifier/versions/2/ready
```

**Observed result:**

```text
Server: HTTP 200
Version 1: HTTP 200
Version 2: HTTP 200
```

A running container or listening HTTP port does not by itself prove that a specific model version is ready.

## 5. Run version-specific inference

Create `request.json`:

```json
{
  "inputs": [
    {
      "name": "INPUT",
      "shape": [4],
      "datatype": "FP32",
      "data": [1.0, 2.0, 3.0, 4.0]
    }
  ]
}
```

Run inference against **version 1**:

```bash
curl -sS -X POST \
  http://localhost:8000/v2/models/toy_classifier/versions/1/infer \
  -H "Content-Type: application/json" \
  --data-binary @request.json | python -m json.tool
```

**Observed response:**

```json
{
  "model_name": "toy_classifier",
  "model_version": "1",
  "outputs": [
    {
      "name": "OUTPUT",
      "datatype": "FP32",
      "shape": [2],
      "data": [4.0, 6.0]
    }
  ]
}
```

Run inference against **version 2**:

```bash
curl -sS -X POST \
  http://localhost:8000/v2/models/toy_classifier/versions/2/infer \
  -H "Content-Type: application/json" \
  --data-binary @request.json | python -m json.tool
```

**Observed response:**

```json
{
  "model_name": "toy_classifier",
  "model_version": "2",
  "outputs": [
    {
      "name": "OUTPUT",
      "datatype": "FP32",
      "shape": [2],
      "data": [8.0, 12.0]
    }
  ]
}
```

### Why the outputs differ

Each ONNX model performs `MatMul` on the same four-value input. The model-generation script multiplies the base weight matrix by the version number, so version 2 produces exactly twice version 1's output.

| Version | Input | Observed output |
| --- | --- | --- |
| `1` | `[1, 2, 3, 4]` | `[4, 6]` |
| `2` | `[1, 2, 3, 4]` | `[8, 12]` |

This verifies **version-specific serving**; it is not a model-accuracy or performance comparison.

## 6. Break/fix experiment: invalid backend

**Purpose:** Demonstrate that Triton may remain running while its models are unavailable.

Back up the working configuration and stop the server:

```bash
cp model_repository/toy_classifier/config.pbtxt config.pbtxt.backup
docker stop day46-triton
```

Introduce a deliberate error:

```bash
sed -i 's/backend: "onnxruntime"/backend: "not_a_real_backend"/' \
  model_repository/toy_classifier/config.pbtxt

grep backend model_repository/toy_classifier/config.pbtxt
```

Launch Triton in diagnostic mode so it stays running despite model-loading errors:

```bash
docker run -d --rm \
  --name day46-triton \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$PWD/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:25.08-py3 \
  tritonserver \
  --model-repository=/models \
  --exit-on-error=false
```

Inspect the error and readiness:

```bash
docker logs day46-triton --tail 60

curl -s -o /dev/null -w "Server readiness: HTTP %{http_code}\n" \
  localhost:8000/v2/health/ready

curl -s -o /dev/null -w "Model readiness: HTTP %{http_code}\n" \
  localhost:8000/v2/models/toy_classifier/versions/1/ready
```

**Observed failure:**

```text
failed to load 'toy_classifier' version 1:
Invalid argument: unable to find backend library for backend 'not_a_real_backend'

failed to load 'toy_classifier' version 2:
Invalid argument: unable to find backend library for backend 'not_a_real_backend'

model version 1: UNAVAILABLE
model version 2: UNAVAILABLE

Server readiness: HTTP 400
Model readiness: HTTP 400
```

### Root cause and interpretation

- **Root cause:** `config.pbtxt` references a nonexistent backend, so neither model version can initialize.
- **`--exit-on-error=false`:** Allows the Triton process to continue after the model-loading failure.
- **Strict readiness:** With the logged `strict_readiness=1`, the server readiness endpoint reports a failure when the models are not ready. This **does not** terminate the process.
- **GPU warnings:** Unrelated to the intentionally invalid backend in this CPU exercise.

## 7. Restore service and verify recovery

Stop the failed server, restore the working configuration, and relaunch:

```bash
docker stop day46-triton

cp config.pbtxt.backup model_repository/toy_classifier/config.pbtxt
grep backend model_repository/toy_classifier/config.pbtxt

# Expected: backend: "onnxruntime"

docker run -d --rm \
  --name day46-triton \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$PWD/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:25.08-py3 \
  tritonserver --model-repository=/models
```

Recheck readiness:

```bash
curl -s -o /dev/null -w "Server: HTTP %{http_code}\n" \
  localhost:8000/v2/health/ready

curl -s -o /dev/null -w "Version 1: HTTP %{http_code}\n" \
  localhost:8000/v2/models/toy_classifier/versions/1/ready

curl -s -o /dev/null -w "Version 2: HTTP %{http_code}\n" \
  localhost:8000/v2/models/toy_classifier/versions/2/ready
```

**Observed after recovery:**

```text
Model version 1: READY
Model version 2: READY
Server: HTTP 200
Version 1: HTTP 200
Version 2: HTTP 200
```

The recovery was verified in the lab logs, not just inferred from the corrected file.

## 8. Operational takeaways

1. **Repository presence is not model readiness.** Confirm initialization through logs and readiness endpoints.
2. **Version policy governs availability.** `latest: { num_versions: 2 }` allows the two highest-numbered versions, assuming successful loading.
3. **`MODE_NONE` does not automatically reload model repository changes.** Triton must restart to discover updated models/configuration in the default model-control mode. Other model-control modes have different behavior.
4. **Process health and model readiness are different.** A container can be running while model readiness fails; Kubernetes readiness probes should reflect whether inference traffic can be served.
5. **`max_batch_size: 0` disables batching for this model.** Model execution instances are configured separately with `instance_group`.
6. **CPU validation does not establish GPU compatibility.** Validate GPU support with NVIDIA Container Toolkit, compatible host drivers, and a GPU-backed configuration in a separate exercise.

## 9. Troubleshooting quick reference

| Symptom | What to check |
| --- | --- |
| Model is `UNAVAILABLE` | `docker logs day46-triton --tail 60`; backend name and artifacts |
| HTTP readiness returns `400` | Model status table; required models and load failures |
| Port already allocated | Existing container/process on ports 8000–8002 |
| Model version is missing | Numeric version directory and `version_policy` |
| Repository update is ignored | `model_control_mode`; restart under `MODE_NONE` |
| CUDA/driver warning | Whether GPU was requested; NVIDIA driver/runtime compatibility if using GPU |

## 10. Cleanup

```bash
docker stop day46-triton
```

The `--rm` flag removes the stopped container, but **does not delete** the host `model_repository/` directory.

## Next step — Day 47

**Triton batching and concurrency:** experiment with `max_batch_size`, dynamic batching, `instance_group`, concurrent requests, and measured latency/throughput. Capture measured outcomes rather than assuming that batching always improves every metric.

## References

- [NVIDIA Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- [Triton model repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
- [Triton model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
- [Triton model management](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html)

---

*Lab evidence: two successful version-specific inference responses, a deliberate invalid-backend failure, two HTTP 400 readiness checks during the failure, and HTTP 200 checks for both versions after recovery. No load-testing or GPU performance benchmarks were performed in Day 46.*
