# Day 38 — Dockerized Inference Server

**Track:** AI Infrastructure & ML Systems — 220-Day Roadmap  
**Phase:** Model Serving / Inference Infrastructure  
**Status:** Completed  
**Application source:** `~/ai-infra/day37-fastapi-inference`

## Objective

Package the Day 37 FastAPI/PyTorch inference API into a Docker image, verify its runtime and API contract, produce a leaner CPU-only image, and diagnose a deliberately introduced container networking failure.

## Architecture

```text
Client (curl)
  -> localhost:8000 / 8001 / 8002 (published host port)
  -> Docker container :8000
  -> Uvicorn + FastAPI
  -> PyTorch model
  -> JSON response
```

## Application contract

- `GET /health`: confirmed `200 OK`, body `{"status":"healthy"}`.
- `GET /ready`: confirmed `200 OK`, body `{"status":"ready"}` in the initial test.
- `POST /predict`: expects `{"features": [number, number, number]}`; response includes `prediction` and `model_version`.
- Invalid two-element feature array was rejected with `422 Unprocessable Entity`.
- A successful post-recovery request returned `{"prediction": 2.4, "model_version": "demo-linear-v1"}`.

## Build artifacts

- `Dockerfile` — original runtime image based on Python 3.11 slim, installing `requirements.txt`.
- `Dockerfile.cpu` — CPU-only PyTorch via `https://download.pytorch.org/whl/cpu`; installs minimal runtime dependencies.
- `requirements.txt` — `fastapi[standard]`, `torch`, `pytest`, `httpx`, `numpy`.
- `requirements.cpu.txt` — `fastapi`, `uvicorn`, `numpy` (CPU-only torch installed separately).
- `.dockerignore` — excludes local virtual environments, caches, test folders, Git metadata, and environment files.

The Dockerfile uses `WORKDIR /app`, exposes container port 8000, starts Uvicorn on `0.0.0.0:8000`, and runs as non-root `appuser` (verified in `docker inspect`).

## Observed results

| Measurement | Original `ai-inference:v1` | CPU-optimized `ai-inference:cpu` |
|---|---:|---:|
| Docker-reported image disk usage | 5.90 GB | 1.09 GB |
| Initial build elapsed time | 176.7 seconds | 39.2 seconds |
| Observed container memory | 308.7 MiB | 168 MiB |
| Observed idle CPU | 0.18% | 0.19% |
| Inference HTTP status | 200 | 200 |

- **Image-size reduction:** approximately **81.5%**.
- **Observed memory-use reduction:** approximately **45.6%**.
- Original PyTorch: **2.14.1+cu130** (CUDA build 13.0); optimized image installed from the CPU-only index.
- Times are separate builds with potentially different cached layers and download conditions; not a controlled build-speed benchmark.
- Memory and CPU readings are snapshots, not latency or throughput benchmarks.
- The Docker-reported 31.1 GiB memory *limit* should not be interpreted as a configured per-container memory cap; `docker inspect` in the failure lab showed `HostConfig.Memory: 0` for that container.

## Reproduction commands

```bash
cd ~/ai-infra/day37-fastapi-inference

docker build -t ai-inference:v1 .
docker build -f Dockerfile.cpu -t ai-inference:cpu .
docker images ai-inference

# On a clean host, choose host ports that are not in use.
docker run -d --name inference-server -p 127.0.0.1:8000:8000 ai-inference:v1
docker run -d --name inference-cpu -p 127.0.0.1:8001:8000 ai-inference:cpu

curl -i http://localhost:8001/health
curl -i http://localhost:8001/ready
curl -i -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[1.0,2.0,3.0]}'

# Deliberate port collision — expected to fail:
docker run -d --name inference-port-conflict \
  -p 127.0.0.1:8001:8000 ai-inference:cpu

docker ps -a
docker inspect inference-port-conflict
sudo ss -ltnp | grep ':8001'

# Recovery:
docker rm -f inference-port-conflict
docker run -d --name inference-recovered \
  -p 127.0.0.1:8002:8000 ai-inference:cpu
curl -i http://localhost:8002/health
curl -i http://localhost:8002/ready
curl -sS -X POST http://localhost:8002/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[1.0,2.0,3.0]}' | python3 -m json.tool
```

For repeated runs, stop and remove prior demo containers or use alternate names and host ports.

## Break/fix incident report

**Symptom:** A new `ai-inference:cpu` container couldn't publish `127.0.0.1:8001`.

**Error:** `Bind for 127.0.0.1:8001 failed: port is already allocated`.

**Evidence:** `docker ps -a` showed `inference-port-conflict` with status `Created`, while `inference-cpu` was already running with host port 8001. `docker inspect` reported `Running: false`, `OOMKilled: false`, `ExitCode: 128`, and an error specifying host-port binding. `sudo ss -ltnp` showed `docker-proxy` listening on `127.0.0.1:8001`.

**Root cause:** Docker host-port collision during container networking setup, before application startup; not a PyTorch initialization failure.

**Resolution:** Removed the failed container, then launched `inference-recovered` publishing port 8002 to container port 8000. Health and readiness endpoints returned HTTP 200. Prediction returned `2.4` for three numerical features using model version `demo-linear-v1`.

**Preventive guidance:** Keep an inventory of published host ports; use distinct host ports in local multi-container demos. In Kubernetes, pods have separate network identities and can listen on the same container port without publishing duplicate host ports. Use Services for stable access and load balancing.

## Lessons learned

1. An image is a reusable packaged artifact; a container is an instance. Image creation is not proof of operational readiness.
2. `0.0.0.0:8000` listens on all network interfaces *at port 8000*, not on all ports. `127.0.0.1` inside a container is that container's loopback interface.
3. Different Docker network namespaces allow identical *container* listening ports; publishing the same *host IP:port* twice causes a collision.
4. Build layer ordering improves caching; CPU-only PyTorch substantially reduced image disk usage for the CPU inference task.
5. HTTP 200 and valid JSON confirm a functioning API path, not model correctness. Validate checkpoints, preprocessing, parameters, and model quality separately.
6. Distinguish network configuration, container runtime startup, API readiness, and model behavior when diagnosing failures.

## Follow-on work

- Pin compatible runtime package versions and the base-image digest before calling the Dockerfile reproducible for production.
- Keep CPU and GPU images distinct and document their environment requirements.
- Add controlled load tests to measure p50/p95 latency and throughput rather than infer performance from image size.
- **Day 39:** GPU-backed inference: NVIDIA container runtime, device exposure, PyTorch CUDA selection, and CPU-vs-GPU correctness/performance checks.

## End-of-day retention paragraph

Docker packages our FastAPI/PyTorch inference application with its dependencies into a reusable image, while containers run isolated instances of that image. We verified model startup, readiness, valid predictions, and input rejection, then replaced an unnecessarily large CUDA-enabled image with a CPU-only variant, reducing measured disk usage from 5.90 GB to 1.09 GB. A deliberate host-port collision demonstrated that multiple containers can listen on the same internal port but cannot publish an identical host IP and port; the failure was diagnosed with `docker ps -a`, `docker inspect`, and `ss`, and recovery succeeded by publishing a different host port. Image size, runtime memory, API availability, and prediction correctness are distinct dimensions that must be measured and diagnosed separately.
