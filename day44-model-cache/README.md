# Day 44 — Model Loading, Caching & GPU Warmup

**Track:** AI Infrastructure & ML Systems — 220-Day Roadmap  
**Phase:** 3 — Model Serving / Inference Infrastructure  
**Stack:** Python, PyTorch, CUDA, FastAPI, Uvicorn  
**Status:** Local single-GPU experiments completed

## Overview

This lab investigates how model loading and caching affect inference startup, latency, concurrency, and resource use. It builds a small PyTorch model, measures the stages of loading it onto a GPU, and exposes it through FastAPI using **eager** and **lazy** initialization. The service caches the model within one worker process and uses a thread lock to prevent concurrent requests from initializing duplicate copies.

A configurable warmup experiment shows how running a few GPU inferences before accepting traffic changes first-request latency.

> **Scope:** This is an infrastructure and performance lab, not an accuracy benchmark. The model is randomly initialized, small (~16.07 MiB checkpoint), and used only to demonstrate serving behavior. The results below are measured on a local CUDA-enabled machine and should not be generalized to other GPUs or larger models.

## Learning objectives

- Separate checkpoint retrieval, CPU deserialization/reconstruction, CPU-to-GPU transfer, and first inference.
- Distinguish **disk caching** from an **in-memory/GPU-resident model**.
- Compare the startup and first-request costs of eager versus lazy loading.
- Coordinate concurrent lazy initialization with a per-process lock.
- Measure the tradeoff between GPU warmup time and first-request latency.
- Relate model readiness and cache lifecycle to Kubernetes deployment design.

## Architecture

```text
   Model checkpoint (model_weights.pt)
                 |
         read / deserialize
                 v
        CPU model + weights
                 |
           .to("cuda")
                 v
        GPU-resident model <-----------------------+
                 |                               |
         optional warmup                          |
                 |                               |
       FastAPI /predict requests                  |
                 |                               |
         ModelCache.get_model() ------------------+
                 |
         inference + response
```

**Eager mode:** Load the model during FastAPI lifespan startup; requests reuse it.  
**Lazy mode:** Delay loading until the first prediction request; subsequent requests reuse it.  
**Thread safety:** `threading.Lock` protects first-time initialization **within one worker process**.

## Project files

```text
day44-model-cache/
├── app.py                    # FastAPI service; eager/lazy loading and optional warmup
├── benchmark.py              # Checkpoint read / RAM / GPU / inference timings
├── concurrent_test.py        # Ten simultaneous HTTP requests
├── model_weights.pt          # Generated synthetic checkpoint (~16 MiB)
├── app_before_warmup.py      # Optional local backup of the pre-warmup service
└── README.md
```

`model_weights.pt` can be regenerated; committing a binary checkpoint is optional. For a clean repository, consider committing a small checkpoint-generation script instead.

## Requirements

- Linux or another environment supporting Python and PyTorch.
- Python virtual environment with `torch`, `fastapi`, and `uvicorn` installed.
- NVIDIA GPU with a functional CUDA-enabled PyTorch installation for GPU measurements. The code can fall back to CPU, but CPU results are not comparable with the GPU results below.

For a fresh environment, install the dependencies appropriate for your CUDA/PyTorch setup, and check device availability:

```bash
python3 -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
```

Run commands from the project directory:

```bash
cd ~/ai-infra/day44-model-cache
```

### Generate the synthetic checkpoint

The checkpoint consists of a 2,048 → 2,048 → 8 fully connected network with ReLU activation:

```bash
python3 - <<'PY'
import torch
from pathlib import Path

torch.manual_seed(44)
model = torch.nn.Sequential(
    torch.nn.Linear(2048, 2048),
    torch.nn.ReLU(),
    torch.nn.Linear(2048, 8),
)

path = Path("model_weights.pt")
torch.save(model.state_dict(), path)
print(f"Checkpoint: {path}")
print(f"Size: {path.stat().st_size / 1024**2:.2f} MiB")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
```

**Observed:** 16.07 MiB checkpoint; CUDA available: `True`.

## Experiment 1 — Model-loading pipeline

Run:

```bash
python3 benchmark.py
```

`benchmark.py` measures four stages with `time.perf_counter()`: reading a checkpoint with `torch.load(..., map_location="cpu", weights_only=True)`, reconstructing the model in RAM, moving it to the execution device, and executing the first inference. GPU synchronization is used around device transfer and inference measurements.

### Actual results

All times are milliseconds (ms):

| Stage | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Checkpoint read | 17.33 | 6.38 | 2.76 |
| RAM reconstruction | 26.17 | 24.68 | 26.16 |
| Device transfer | 9.77 | 3.28 | 3.24 |
| First inference | 66.53 | 0.22 | 0.19 |

**Interpretation:** Repeated checkpoint reads became faster, consistent with filesystem page caching; CPU reconstruction stayed near 25–26 ms. GPU transfer and inference were slower on the first iteration, consistent with one-time CUDA/library initialization overhead. This experiment does **not** prove an application-level model cache: it creates a fresh model in each iteration.

## Experiment 2 — Eager vs. lazy model loading

The FastAPI service supports:

- `LOAD_MODE=eager`: `ModelCache.get_model()` executes during app lifespan startup.
- `LOAD_MODE=lazy`: the first `/predict` call triggers initialization.
- `/status`: reports loading mode, device, whether the model is resident, load count, and loading time.
- `/predict`: runs a batch-of-one inference on a 2,048-element input, returning the 8-element output and timing fields.

### Eager startup

In terminal 1:

```bash
LOAD_MODE=eager WARMUP_RUNS=0 uvicorn app:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

In terminal 2:

```bash
curl -s http://127.0.0.1:8000/status

curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"scale": 1.0}'
```

Repeat the prediction request. The model should already be loaded before the first prediction; `load_count` should stay at `1`.

### Lazy startup

Stop the server with `Ctrl+C`, then restart:

```bash
LOAD_MODE=lazy uvicorn app:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Before requesting inference, `/status` should report `loaded: false`, `load_count: 0`. The first `/predict` loads the model; the next request reuses it.

### Actual eager/lazy results

These are the initial eager-versus-lazy measurements (before the separate warmup experiment):

| Metric | Eager | Lazy |
|---|---:|---:|
| Model load time | 128.84 ms | 126.16 ms (server log) |
| First request's loading wait | 0.001 ms | 126.961 ms |
| First inference | 57.938 ms | 48.024 ms |
| Second inference | 0.278 ms | 0.290 ms |
| Model load count after two requests | 1 | 1 |

**Conclusion:** Eager loading pays the loading cost during startup; lazy loading shifts it to the first request. Both approaches reused the GPU-resident instance after initialization. Eager loading **alone** did not remove the first GPU execution's cold-start overhead.

> `loading_wait_ms` and `inference_ms` are **instrumented server-side spans**, not full HTTP round-trip latency. They do not account for every component of request processing.

## Experiment 3 — Concurrent lazy initialization

Restart the server in lazy mode so the model is initially unloaded:

```bash
LOAD_MODE=lazy uvicorn app:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

From another terminal:

```bash
python3 concurrent_test.py
```

The test uses a `ThreadPoolExecutor` and `Barrier` to initiate **10 HTTP requests** near-simultaneously.

### Actual results

- Successful requests: **10/10**.
- `load_count`: **1** for every response and final `/status`.
- Single model initialization: **143.63 ms**.
- Per-request loading wait: approximately **74.92–144.71 ms**.
- Total HTTP request duration: approximately **313.55–317.44 ms**.

**Conclusion:** The per-process lock prevented duplicate initialization. Requests still waited for the model and encountered other sources of latency. The fact that all requests completed does **not** establish production-scale throughput capacity.

**Critical limitation:** A `threading.Lock` is **not shared across separate Uvicorn worker processes**. With `--workers 4`, each process has its own cache and can independently allocate model weights on the GPU. This can lead to duplicated VRAM usage and out-of-memory failures. The lab intentionally uses `--workers 1`.

## Experiment 4 — Eager loading with GPU warmup

The updated `app.py` supports `WARMUP_RUNS`, which runs that many GPU inference passes during lifespan startup after eager model loading. Compare a fresh server process with and without warmup:

```bash
# Baseline: no warmup
LOAD_MODE=eager WARMUP_RUNS=0 uvicorn app:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Stop the server before starting the warmed configuration:

```bash
# Five startup warmup passes
LOAD_MODE=eager WARMUP_RUNS=5 uvicorn app:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

For each process, run the same request three times:

```bash
for i in 1 2 3; do
  curl -s -X POST http://127.0.0.1:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"scale": 1.0}' |
  python3 -c 'import json,sys; r=json.load(sys.stdin); print("inference_ms:", r["inference_ms"], "loading_wait_ms:", r["loading_wait_ms"])'
done
```

### Actual results

| Metric | No warmup | Five warmup passes |
|---|---:|---:|
| Model loading | 141.13 ms | 138.81 ms |
| Warmup execution | 0 ms | 48.30 ms |
| First inference | 66.064 ms | 15.919 ms |
| Second inference | 0.575 ms | 0.745 ms |
| Third inference | 0.335 ms | 0.437 ms |

First-request inference time decreased from **66.064 ms to 15.919 ms**, or approximately **76%**, in this comparison. The model warmed at startup still had a slower first real inference than later requests. One possible reason is that the warmup path reuses an input tensor while `/predict` allocates a new input; additional measurements would be needed to isolate the cause.

**Tradeoff:** Warmup moves execution overhead to the startup phase. These results came from one fresh process for each setting, with subsequent requests; repeat across several clean starts before drawing general conclusions.

## Kubernetes implications

For **eager** model serving, startup and readiness probes should allow enough time for model retrieval, loading, and warmup, and only route traffic to replicas ready for their intended inference workload:

```yaml
startupProbe:
  httpGet:
    path: /ready
    port: 8000
  periodSeconds: 5
  failureThreshold: 36

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  periodSeconds: 5

livenessProbe:
  httpGet:
    path: /live
    port: 8000
  periodSeconds: 10
  failureThreshold: 3
```

This example permits roughly 180 seconds for startup checks; real thresholds should be sized using measured startup behavior and timeouts. `/live` and `/ready` refer to the **optional health-endpoint extension** discussed in the lab; confirm they are implemented in `app.py` before using this manifest.

**Lazy-loading caveat:** If `/ready` returns 503 until the model is resident, while the first real prediction request is the *only* trigger for loading, Kubernetes may never send traffic to the pod. Avoid this readiness deadlock either by defining readiness as the ability to accept and coordinate on-demand loading, or by independently preloading the model before reporting ready.

### Cache persistence and versioning

- A local disk checkpoint cache can avoid repeat downloads but does not automatically preserve process RAM or GPU allocations after restart.
- An `emptyDir` volume persists through container restarts within the same Pod but not Pod deletion/recreation.
- A suitably configured persistent volume or shared artifact store can preserve cached model files across supported pod lifecycles.
- Use **immutable model version IDs and checksums** to prevent stale artifacts from being served. Do not indiscriminately clear every cached model on every restart.

## Troubleshooting checklist

| Symptom | Investigation |
|---|---|
| Pod restarts before model becomes ready | Startup/liveness probe timing; application logs; actual loading and warmup duration |
| First request much slower than subsequent ones | Model load timing; CUDA initialization; warmup; GPU allocations; first-use kernels |
| Multiple concurrent requests load the model repeatedly | Locking around initialization; second `None` check; number of worker processes |
| GPU out-of-memory with multiple workers | Per-process duplicated weights; worker count; CUDA memory use; other GPU workloads |
| Model unexpectedly changes after deployment | Model version pinning; cache paths; artifact checksums; stale checkpoint reuse |
| Lazy pod never passes readiness | Check whether requests can reach it to trigger loading; adjust readiness contract or preload |
| Low throughput despite caching | Queue depth; GPU occupancy; batching; concurrency policy; backpressure; CPU bottlenecks |

## Limitations and next steps

**Measured in this lab:** local single-GPU loading stages, process-level model reuse, 10 concurrent requests, and five-pass warmup.  
**Not yet measured:** sustained throughput, full end-to-end latency percentiles (p50/p95/p99), GPU memory utilization by worker, cache performance across Kubernetes nodes, or model-version correctness under a deployment rollout.

Possible improvements:

1. Add end-to-end latency histograms and repeated fresh-process trials.
2. Benchmark warm versus cold concurrent traffic under a bounded request rate.
3. Pin model versions and validate artifact checksums before loading.
4. Document readiness semantics for eager and lazy deployment modes.
5. Compare the custom FastAPI service with NVIDIA Triton Inference Server (Day 45 onward).

## Key takeaway

> Model caching reduces repeated artifact retrieval and initialization, while eager loading and warmup move latency from request handling into startup. Locks prevent duplicate initialization within one process, but they do not share GPU-resident weights across multiple processes. Production serving requires coordinated model lifecycle management, controlled concurrency, readiness semantics, and version-safe caches.

---

**Author's note:** Benchmark values above are observations from the Day 44 local experiments. They are not vendor specifications or claims about production serving performance.
