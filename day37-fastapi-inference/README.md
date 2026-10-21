# Day 37 — FastAPI/PyTorch Inference Service

## Overview

This project implements a lightweight machine learning inference service using **FastAPI, PyTorch, and Pydantic**. It is part of my 220-Day AI Infrastructure & ML Systems Engineering roadmap.

The objective is to understand how machine learning models can be exposed through HTTP APIs, how inference requests are validated and processed, and how model-serving applications handle initialization, health checks, readiness, and failures.

The service uses a small PyTorch linear regression model with predefined weights to demonstrate inference infrastructure without requiring model training or GPU resources.

## Learning Objectives

- Build and expose a machine learning model through a REST API using FastAPI.
- Validate inference requests and responses with Pydantic.
- Integrate a PyTorch model into an inference service.
- Load the model once during application startup using FastAPI lifespan management.
- Understand `model.eval()` and `torch.inference_mode()`.
- Implement separate health and readiness endpoints.
- Diagnose HTTP errors, model availability failures, and connection failures.
- Write automated API tests using pytest.
- Understand the relationship between worker processes, concurrency, memory usage, and inference performance.

## Architecture

The application follows a simple inference-serving pipeline:

```text
Client
   |
   | HTTP POST /predict
   v
Uvicorn (ASGI Server)
   |
   v
FastAPI Application
   |
   v
Pydantic Input Validation
   |
   v
PyTorch Model
   |
   | Inference
   v
Prediction Response
   |
   v
Client (JSON)
```

The PyTorch model is initialized during application startup and reused across inference requests within the application worker.

### Project Structure

```text
day37-fastapi-inference/
├── main.py
├── model.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── .gitignore
└── README.md
```

**File responsibilities:**

- `main.py`: FastAPI application, API endpoints, request validation, model initialization, and inference execution.
- `model.py`: PyTorch model definition.
- `tests/test_api.py`: Automated tests for application health, predictions, input validation, and model unavailability.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Files excluded from version control.

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| API framework | FastAPI | HTTP request handling and API routing |
| ASGI server | Uvicorn | Serve the FastAPI application |
| ML framework | PyTorch | Model execution and tensor computation |
| Validation | Pydantic | Request and response schemas |
| Testing | pytest, HTTPX | Automated API testing |
| Environment | Python 3.12, Ubuntu Linux | Local development |

## Installation and Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Install the CPU version of PyTorch:

```bash
python -m pip install torch \
  --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Start the inference server

```bash
uvicorn main:app --reload \
  --host 127.0.0.1 --port 8000
```

The API is available at:

`http://127.0.0.1:8000`

Interactive documentation:

`http://127.0.0.1:8000/docs`

The `--reload` option is intended for local development, not production deployment.

## PyTorch Model

The service uses a simple linear neural network implemented with `torch.nn.Linear(3, 1)`.

The model accepts three input features and produces one numerical prediction.

Its computation is:

\[
y = 0.2x_1 + 0.3x_2 + 0.5x_3 + 0.1
\]

For example:

```json
{
  "features": [1.0, 2.0, 3.0]
}
```

The model produces:

```json
{
  "prediction": 2.4,
  "model_version": "demo-linear-v1"
}
```

The weights are manually configured for reproducibility. The goal is to demonstrate ML-serving architecture, rather than train a predictive model.

### Model Initialization

The model is loaded during FastAPI's application lifespan and reused across subsequent inference requests.

Two important PyTorch operations are used:

- `model.eval()` enables evaluation behavior for layers such as Dropout and BatchNorm.
- `torch.inference_mode()` disables autograd tracking and reduces inference-related overhead.

These operations perform different functions and are used together during model serving.

## API Endpoints

### GET `/health`

Checks whether the HTTP application is responsive.

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy"
}
```

### GET `/ready`

Checks whether the PyTorch model is initialized and available for inference.

```bash
curl http://127.0.0.1:8000/ready
```

Successful response:

```json
{
  "status": "ready"
}
```

If the model is unavailable, the endpoint returns HTTP `503 Service Unavailable`.

### POST `/predict`

Accepts input features and executes model inference.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

Expected response:

```json
{
  "prediction": 2.4,
  "model_version": "demo-linear-v1"
}
```

### Request Validation

Pydantic enforces the expected input structure:

- Exactly three numerical features.
- Finite floating-point values.
- No unexpected request fields.

For example, supplying only two features results in HTTP `422 Unprocessable Entity`.

This prevents malformed requests from reaching the model execution stage.

## Health Checks and Readiness

The application separates service health from model readiness.

| Condition | `/health` | `/ready` |
|---|---|---|
| Application running, model loaded | 200 | 200 |
| Application running, model unavailable | 200 | 503 |
| Application process stopped | Connection failure | Connection failure |

This separation is important for Kubernetes deployments.

A readiness probe can prevent traffic from being routed to a pod whose model is unavailable, while liveness probes can identify conditions requiring container restarts.

A failed readiness probe alone does not automatically restart the container.

## Failure Injection and Troubleshooting

Controlled failure experiments were performed to understand model-serving reliability.

### Scenario 1 — Model Unavailable

The application supports a temporary failure-simulation setting:

```bash
SIMULATE_MODEL_UNAVAILABLE=1 \
uvicorn main:app --host 127.0.0.1 --port 8000
```

During the experiment:

- `/health` returned HTTP 200.
- `/ready` returned HTTP 503.
- `/predict` returned HTTP 503.

This demonstrated that an application can remain responsive while being unable to execute inference.

### Scenario 2 — Server Process Stopped

The Uvicorn server was terminated, and the health endpoint was tested again.

```bash
curl -i --max-time 3 \
  http://127.0.0.1:8000/health
```

The request failed because no application was accepting connections on port 8000.

The listening socket was inspected using:

```bash
ss -ltnp | grep ':8000'
```

This demonstrated the difference between an application-generated HTTP 503 response and a network connection failure.

### Scenario 3 — Recovery

After restoring normal operation:

- Health checks returned HTTP 200.
- Readiness checks returned HTTP 200.
- Inference requests successfully returned the expected prediction.

These experiments reinforce the importance of separating application health, model availability, and network connectivity when troubleshooting inference systems.

## Automated Testing

The application uses pytest and FastAPI's TestClient to test the API.

Run:

```bash
python -m pytest -v
```

### Test Results

| Test | Purpose | Result |
|---|---|---|
| `test_health` | Verify health endpoint | PASSED |
| `test_predict` | Verify model prediction | PASSED |
| `test_invalid_features` | Verify input validation | PASSED |
| `test_model_unavailable` | Verify unavailable-model handling | PASSED |

**Results from the Day 37 lab:**

```text
4 passed, 1 warning in 1.28s
```

The warning was caused by NumPy not being installed in the initial environment. All four tests passed, and NumPy can be installed to resolve that dependency warning.

## Preliminary Performance Observations

A few localhost HTTP request times were recorded using `curl`.

| Endpoint | Observed request time |
|---|---|
| `/health` | 0.75 ms |
| `/ready` | 0.82 ms |
| `/predict` | 1.62 ms |

Additional successful prediction requests took approximately 1.97 ms and 1.12 ms.

These values represent individual end-to-end localhost observations, not sustained performance benchmarks.

End-to-end request latency may include:

- HTTP request handling and routing
- Input validation
- Data preprocessing and tensor construction
- Model inference
- Response serialization
- Network communication

A low single-request latency does not establish the service's maximum requests per second.

Proper throughput benchmarking requires controlled concurrency, request-rate measurements, latency distributions, and resource monitoring.

## Engineering Lessons Learned

### 1. Separate API responsibilities from model execution

FastAPI handles HTTP requests and response delivery, while PyTorch performs the underlying inference computation.

This separation improves modularity and makes the application easier to maintain.

### 2. Model lifecycle management matters

Loading the model during application startup avoids repeatedly initializing model weights for every request.

However, independent worker processes generally maintain separate model instances, potentially increasing RAM or GPU memory usage.

### 3. Validation protects inference resources

Rejecting malformed requests before model execution avoids unnecessary computation and provides predictable client-facing errors.

### 4. Health does not imply readiness

An application can respond to HTTP requests while its ML model is unavailable.

Separating these checks is critical for reliable inference deployments.

### 5. Reliability requires failure testing

Testing only successful inference requests is insufficient. Model-serving systems should also be tested for invalid inputs, unavailable models, initialization failures, and connectivity problems.

### 6. Concurrency, latency, and throughput are different concepts

Increasing workers does not necessarily improve throughput, particularly when model execution is constrained by CPU, GPU, or memory resources.

System performance must be measured under representative workloads rather than inferred from isolated requests.

## Next Steps — Day 38

The next milestone is to containerize this inference service using Docker.

Planned activities include:

- Create a Dockerfile for the FastAPI/PyTorch application.
- Build and run the container locally.
- Configure container networking and runtime settings.
- Test inference requests against the containerized service.
- Investigate container startup and failure behavior.
- Prepare the application for eventual Kubernetes deployment.

The longer-term objective is to evolve this project into a production-style GPU-backed inference platform with batching, load testing, autoscaling, monitoring, and deployment automation.

---

**Roadmap:** AI Infrastructure & ML Systems Engineering — Day 37/220

**Project status:** Local CPU-backed inference service implemented, tested, and verified.

**Next milestone:** Dockerized inference serving.
