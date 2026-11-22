# Day 67 — Distributed Tracing for GPU-Backed LLM Inference

**AI Infrastructure & ML Systems · 220-Day Roadmap · Phase 4 (Observability)**  
**Lab date:** November 22, 2026 · **Environment:** Ubuntu + Docker Compose + NVIDIA GPU

> **Outcome:** Demonstrated end-to-end propagation of a W3C trace context from a Python client into Docker-hosted **vLLM 0.11.0**, through an **OpenTelemetry Collector 0.165.0**, and into **Jaeger 2.23.0**. A real Qwen2.5-0.5B-Instruct request appeared as **two services in a single trace**. We also injected and removed a known 600 ms client-side delay and compared measured workflow latency.

## Results at a glance

| Evidence | Verified observation |
|---|---|
| Local Jaeger + Collector | Both containers running; Jaeger UI returned HTTP 200; Collector gRPC `4317` and HTTP `4318` started |
| Single-process simulator | Three traces, five spans per trace; Collector logged **15 spans**; Jaeger showed 201, 512, 301 ms root durations |
| Real model serving | `Qwen/Qwen2.5-0.5B-Instruct` on one NVIDIA GPU, FP16, 2,048-token context and `--gpu-memory-utilization 0.65` |
| Docker image | Custom trace-enabled `day67-vllm-otel:v0.11.0`; OpenTelemetry verified available inside image |
| Real distributed tracing | `ai-infra-client` → `ai-infra-vllm`, **two services / two spans / depth two** in one Jaeger trace |
| First observed request | Client **1.3416 s**; vLLM `llm_request` **182.77 ms**; why the remaining time occurred is **not established** |
| Five later requests | **183, 165, 162, 166, 175 ms**; mean **170.2 ms**, median **166 ms**; **5/5 HTTP 200** |
| Controlled fault injection | Baseline **257 ms**, injected 600 ms delay **907 ms**, fixed **237 ms**; mean baseline **247 ms**, slowdown **660 ms** |
| Assessments | Section 1 **10/10**, Section 2 **10/10**, Section 4 **10/10**, Section 5 **10/10** (**40/40** total) |

**Limits:** These are local, sequential, small-sample measurements, not production p95/p99, throughput, or multi-GPU benchmarks. Jaeger comparison of the two *break/fix* traces was proposed but not yet supplied as a screenshot. The simulated traces are not GPU inference traces.

## Why this matters

Metrics answer *how often/how much*; logs answer *what was recorded*; traces show *which instrumented operations a request passed through, and when*. In a multi-service inference system, client latency may include preprocessing, transport, queueing, prefill, decode, and uninstrumented work. A high client duration does **not** by itself prove a GPU bottleneck.

### Architecture

```text
               Host Python client (ai-infra-client)
                 | W3C HTTP traceparent header
                 v
    Docker: day66-vllm / Qwen2.5-0.5B-Instruct
       [Day 66 network] + [Day 67 tracing network]
                 | OTLP/gRPC: otel-collector:4317
                 v
         OpenTelemetry Collector 0.165.0
            | receive → batch → export
                 | OTLP/HTTP: jaeger:4318
                 v
                 Jaeger 2.23.0
                 | Web UI: localhost:16686
```

In a verified Jaeger trace, the Python client's `client.chat_completion` is the parent of the vLLM `llm_request` span. We observed **two services**, not two GPUs. The server-side span does **not** describe every operation that may occur inside vLLM or the HTTP stack.

## Repository contents

```text
README.md                         this document
Day67_Distributed_Tracing_Notes.tex  fully editable, standalone LaTeX textbook-style notes
Day67_Distributed_Tracing_Notes.pdf  rendered notes and TikZ diagrams
docker-compose.yaml               Jaeger + Collector for standalone Day 67 lab
otel-collector.yaml               OTLP gRPC/HTTP receivers → batch → debug + Jaeger
Dockerfile.vllm-otel              extends vllm/vllm-openai:v0.11.0 with tracing packages
compose.day67-otel.yaml           override for the *existing* Day 66 Compose project
requirements-lab.txt             host-side Python tracing dependencies
trace_demo.py                     three synthetic request traces (no GPU)
trace_vllm_client.py              instrumented, context-propagating real HTTP request
trace_breakfix.py                 controlled --delay 0.6 / --delay 0 experiment
measurements.csv                  original observed latency measurements
evidence/                        screenshots of real Jaeger search and span details
```

> The files in this bundle describe a reproducible local demonstration. Running the commands below on another machine requires adapting project paths, model-cache mounts, available GPU resources, and possibly package versions. No production hardening is implied.

## Environment and prerequisites

- Docker Engine + Docker Compose plugin; Python 3 with `venv`; NVIDIA driver and GPU container access for vLLM; model cache already available or downloadable.
- Two separate Docker Compose projects in the verified workstation: `~/ai-infra/day67-distributed-tracing` for Collector/Jaeger, and `~/ai-infra/day66-Loki-Alloy-Grafana` for the existing vLLM stack.
- The **actual running vLLM image was 0.11.0**, although the host's separate `ai-infra-venv` had vLLM **0.33.0**. Always inspect the *container* version when configuring it.
- The original `compose.yaml` and `compose.vllm.yaml` are part of the Day 66 project, **not** duplicated in this bundle. The supplied `compose.day67-otel.yaml` intentionally relies on their existing GPU reservation and volume/port definitions.

## Reproduce the lab

### A. Start the standalone tracing backend

Copy this bundle's contents into a working directory such as `~/ai-infra/day67-distributed-tracing`, then:

```bash
cd ~/ai-infra/day67-distributed-tracing
docker compose config --quiet
docker compose up -d
docker compose ps
docker compose logs --tail=30 otel-collector
curl -sS -o /dev/null -w 'Jaeger HTTP %{http_code}\n' http://127.0.0.1:16686/
```

Expected: the Collector starts OTLP/gRPC on `4317` and OTLP/HTTP on `4318`; Jaeger UI returns `200`. Both published telemetry ports and Jaeger UI are bound to `127.0.0.1` for this local lab.

### B. Run the no-GPU trace demonstrator

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lab.txt
python trace_demo.py
docker compose logs --tail=50 otel-collector
```

Go to [Jaeger](http://localhost:16686/) and search for **`ai-infra-trace-demo`**. Expect **three traces, five spans each**, with `inference.request` as root, and `request.validate`, `queue.wait`, `model.generate.simulated`, `response.serialize` as children. Values are **simulated delays**, not model benchmarks.

### C. Build a vLLM image with tracing available

```bash
cd ~/ai-infra/day67-distributed-tracing
docker build -f Dockerfile.vllm-otel -t day67-vllm-otel:v0.11.0 .
docker run --rm --entrypoint python3 day67-vllm-otel:v0.11.0 -c '
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from vllm.tracing import is_otel_available
print("OpenTelemetry dependencies: OK")
print("vLLM tracing available:", is_otel_available())
'
```

Observed in the actual lab: `OpenTelemetry dependencies: OK` and `vLLM tracing available: True`.

### D. Attach tracing to the existing Day 66 vLLM project

**Important:** The following temporarily **recreates vLLM**. Do it only when it is acceptable to interrupt local inference. **Do not** `docker compose down` the Day 66 stack. Back up the original files and check the merged configuration first.

```bash
cd ~/ai-infra/day66-Loki-Alloy-Grafana
cp -p compose.yaml compose.yaml.pre-day67
cp -p compose.vllm.yaml compose.vllm.yaml.pre-day67
cp ~/ai-infra/day67-distributed-tracing/compose.day67-otel.yaml .

docker compose -p day66-loki \
  -f compose.yaml -f compose.vllm.yaml -f compose.day67-otel.yaml \
  config --quiet

docker compose -p day66-loki \
  -f compose.yaml -f compose.vllm.yaml -f compose.day67-otel.yaml \
  config --format json | python3 -c '
import json,sys
s=json.load(sys.stdin)["services"]["vllm"]
for k in ("image","command","networks","ports","volumes","deploy"):
    print(k, s.get(k))
'
```

Confirm the merged configuration retains **one NVIDIA GPU**, host port **127.0.0.1:8000**, the Hugging Face model cache, the original model/dtype/context/memory flags, and **both Docker networks**. If the network `day67-distributed-tracing_default` is not present, start the tracing stack first.

Recreate only vLLM:

```bash
docker compose -p day66-loki \
  -f compose.yaml -f compose.vllm.yaml -f compose.day67-otel.yaml \
  up -d --no-deps vllm

docker inspect day66-vllm --format '{{.Config.Image}}'
docker exec day66-vllm python3 -c '
import socket
s=socket.create_connection(("otel-collector",4317),timeout=5)
print("Collector reachable")
s.close()
'
```

Wait until model initialization finishes before interpreting connection resets as faults. Confirm model readiness with:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

**Observed settings:** `--model Qwen/Qwen2.5-0.5B-Instruct`, `--dtype float16`, `--max-model-len 2048`, `--gpu-memory-utilization 0.65`, `--otlp-traces-endpoint grpc://otel-collector:4317`. Detailed tracing was **not** enabled.

### E. Generate a real two-service trace

```bash
cd ~/ai-infra/day67-distributed-tracing
source .venv/bin/activate
python trace_vllm_client.py
```

The client creates a `SpanKind.CLIENT` span, injects W3C trace context into `traceparent`, sends a request to the actual `/v1/chat/completions` endpoint, and flushes. In Jaeger search for **`ai-infra-client`**, then inspect the trace. The verified trace had `client.chat_completion` → `ai-infra-vllm: llm_request`, depth **two** and **two spans**.

One verified trace ID was: `e97b70d80509c21b75e6c61ae38acad4`. Trace IDs are examples; each new run generates another ID.

### F. Run the controlled break/fix demonstration

```bash
cd ~/ai-infra/day67-distributed-tracing
python trace_breakfix.py --delay 0.6 | tee breakfix-slow.txt
python trace_breakfix.py --delay 0 | tee breakfix-fixed.txt
```

Jaeger should show `client.inference_workflow` → `client.preprocessing` and `client.http_request` → `ai-infra-vllm: llm_request`. **Compare** injected and fixed traces rather than inferring span timing solely from client output. The original run measured baseline **257 ms**, fault **907 ms**, fixed **237 ms**. The user did not provide Jaeger screenshots of these specific two traces, so the span-by-span difference remains to be checked visually.

Avoid running multiple concurrent GPU-backed vLLM servers on a one-GPU workstation; the Day 67 image is intended to replace the Day 66 service, not to launch another replica.

## Measured latency and interpretation

### First traced real request

| Instrumented value | Observed result |
|---|---:|
| Client elapsed time | 1,341.58 ms |
| vLLM `llm_request` duration | 183 ms (shown rounded in Jaeger) |
| `gen_ai.latency.e2e` | 182.773 ms |
| `gen_ai.latency.time_in_model_inference` | 181.503 ms |
| `gen_ai.latency.time_in_model_prefill` | 57.718 ms |
| `gen_ai.latency.time_in_model_decode` | 123.785 ms |
| `gen_ai.latency.time_in_queue` | 0.068 ms |
| `gen_ai.latency.time_to_first_token` | 59.366 ms |
| Prompt tokens / output tokens | 37 / 32 |

Within the model inference time, prefill accounted for approximately **31.8%** and decode for approximately **68.2%**. Do **not** add time-to-first-token as though it were a separate sequential stage. The initial **~1.16-second** gap between the client duration and observed engine span is **unattributed**; it may include work outside engine instrumentation. The trace alone does not establish what caused it.

### Repeated real requests

Five requests: **183, 165, 162, 166, 175 ms**. Mean **170.2 ms**; median **166 ms**; range **162–183 ms**. All returned HTTP 200. No p95/p99 or sustained throughput claim is made from five sequential trials.

### Controlled delay experiment

| Trial | Artificial delay | Measured total workflow |
|---|---:|---:|
| Baseline A | 0 ms | 257 ms |
| Injected fault | 600 ms | 907 ms |
| Baseline B | 0 ms | 237 ms |

Mean baseline `(257 + 237)/2 = 247 ms`; increase `907 − 247 = 660 ms`. The known injected delay accounts for **600 ms** of that change; the remaining **60 ms** cannot be assigned to a cause using these three measurements.

## Evidence snapshots

- [Synthetic trace search](evidence/jaeger_demo_search.png) — three simulated traces in Jaeger.
- [Synthetic span waterfall](evidence/jaeger_demo_timeline.png) — parent and child operations of a 512 ms simulated request.
- [Real two-service trace](evidence/jaeger_two_service_trace.png) — `ai-infra-client` and `ai-infra-vllm` in one trace.
- [vLLM timing attributes](evidence/jaeger_vllm_span_attributes.png) — token counts and measured prefill/decode/queue timings.

## Troubleshooting runbook

| Symptom | Check first | Verification |
|---|---|---|
| `vllm: command not found` on host | Active virtual environment; container vs host installation | `docker exec day66-vllm python3 -c 'from importlib.metadata import version; print(version("vllm"))'` |
| OpenTelemetry `ModuleNotFoundError` inside container | Missing packages in *container* image | Build Dockerfile; test `vllm.tracing.is_otel_available()` |
| No spans in Collector | Client SDK/exporter, endpoint, sampling and port mapping | `docker compose logs otel-collector` |
| Collector receives spans but Jaeger empty | Exporter endpoint `http://jaeger:4318`, search window, Jaeger readiness | Jaeger UI service search + Collector exporter error logs |
| vLLM cannot reach Collector | Container network vs `localhost` confusion | `docker exec day66-vllm` socket connect to `otel-collector:4317` |
| Connection reset just after vLLM recreation | Model still initializing, or genuine process failure | `docker logs day66-vllm`, then `/health` and `/v1/models` |
| Client and vLLM have different trace IDs | Missing/unpropagated `traceparent`, incompatible propagation/sampling | Print injected header; inspect trace IDs in Jaeger |
| Client latency much larger than server span | Uninstrumented work or overhead outside the visible span | Add measured spans and correlate metrics/logs; do not guess |

## Production design: what *not* to assume from the lab

1. **Sampling:** Head sampling decides early; tail sampling can preserve slow/error traces after observing more of their spans, but needs buffering and trace-aware routing across scaled Collectors.
2. **Delivery:** A Python SDK flush does not prove storage in Jaeger. The tracing backend may be unavailable; protect inference from blocking export, and monitor dropped spans.
3. **Privacy/security:** Do not put raw prompts, outputs, credentials, or identity-bearing data into span attributes by default. Local insecure OTLP and in-memory Jaeger are demonstration settings only.
4. **Overhead:** Tracing is not zero-cost. Compare controlled workloads with and without tracing, with instrumentation levels and traffic held comparable.
5. **Signal correlation:** Metrics (Prometheus), logs (Loki), and traces (Jaeger) answer different questions. Trace IDs in logs can join evidence, but this specific correlation has not been verified for the Day 66 Loki deployment.
6. **Scale:** This is one GPU and one model. It does not demonstrate multi-replica orchestration, production SLOs, GPU fleet efficiency, or a hardened tracing backend.

## Interview walkthrough (concise STAR)

- **Situation:** A working GPU-backed Qwen inference service had latency visible to the client, but request-level engine timing was not enough to explain the whole path.
- **Task:** Instrument end-to-end requests, propagate cross-service tracing context, and use evidence to distinguish model latency from client work.
- **Action:** Deployed Jaeger and an OpenTelemetry Collector; rebuilt the containerized vLLM 0.11.0 image with tracing dependencies; joined the Day 66 service to the Day 67 network; propagated `traceparent`; inspected timing attributes; injected a controlled 600 ms preprocessing delay.
- **Result:** Verified **two services in one trace**; measured five later requests at **170.2 ms mean**; the controlled delay raised workflow time from a **247 ms mean baseline** to **907 ms**. Did not claim a cause for uninstrumented first-request time.

## Learning outcomes and next lesson

- Explain trace ID, span ID, parent, context propagation, sampling, OTLP, Collector pipeline and Jaeger storage/visualization.
- Distinguish the measured critical path from sums of overlapping child-span durations.
- Debug missing traces, disconnected spans, Docker network mistakes, missing tracing dependencies and model warmup after recreation.
- Discuss Collector resilience, privacy, instrumentation overhead and production sampling.
- **Day 68:** Alerting; **Day 69:** SLI/SLO concepts; **Day 70:** Observability capstone.

### End-of-day concepts to retain

Distributed tracing follows an individual request through instrumented operations and services using a shared trace ID and propagated parent context; spans expose timing and relationships, while OpenTelemetry instruments and exports telemetry, Collectors receive/process/route it, and Jaeger helps inspect it. Our Docker-hosted Qwen2.5-0.5B vLLM service and Python client produced a verified two-service trace, and a deliberately introduced 600 ms preprocessing delay increased observed workflow latency, demonstrating why slow requests should be diagnosed by comparing client, transport, queue, prefill, decode and uninstrumented time rather than blaming GPU execution prematurely. In production, sampling, Collector retries/queues, privacy controls, trace-aware routing, and instrumentation overhead must be engineered deliberately, and a visible timing gap is not itself proof of root cause.

## Primary references

- [OpenTelemetry Collector configuration](https://opentelemetry.io/docs/collector/configuration/)
- [OpenTelemetry sampling concepts](https://opentelemetry.io/docs/concepts/sampling/)
- [OpenTelemetry Collector architecture](https://opentelemetry.io/docs/collector/architecture/)
- [Jaeger 2.23 getting started](https://www.jaegertracing.io/docs/2.23/getting-started/)
- [vLLM 0.11.0 observability configuration API](https://docs.vllm.ai/en/v0.11.0/api/vllm/config/observability.html)
- [vLLM 0.11.0 metrics and tracing design](https://docs.vllm.ai/en/v0.11.0/design/metrics.html)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)

**Publication:** These files are prepared for manual review and GitHub commit. No repository was changed or pushed by this document-generation step.
