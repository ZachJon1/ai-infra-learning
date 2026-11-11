# Day 56 — Monitoring fundamentals for AI infrastructure

This lab examined a **host-based vLLM inference service** from three viewpoints: the client, the application, and the Linux/GPU host. It used direct endpoint and command-line snapshots to practice interpreting evidence without claiming more than the measurements support.

## Setup and scope

| Component | Lab setup |
| --- | --- |
| Host | Ubuntu laptop; NVIDIA GeForce RTX 3070 Laptop GPU (8,192 MiB VRAM) |
| Serving application | vLLM 0.33.0, running directly on the host |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Endpoint | `http://127.0.0.1:8000` |
| Date | November 10, 2026 (host timestamps in CST) |

The host-based vLLM service in this exercise was separate from earlier Docker, FastAPI, and Minikube experiments. This lab did **not** deploy Prometheus, Grafana, alerts, or distributed traces.

## What was observed

1. With vLLM stopped after a system restart, `nvidia-smi` showed **633 MiB** device-wide GPU memory used, **16%** sampled GPU activity, and **61°C**. A process list was dominated by Brave processes; that short list alone could not establish GPU ownership, request history, or service health.
2. After vLLM started, `/health` returned **HTTP 200**, `/metrics` exposed `vllm:` metrics, and a chat completion returned **HTTP 200** with `Ready.`. The `request_success_total{finished_reason="stop"}` counter changed from **0 to 1**.
3. Ten short requests sent **sequentially** all returned HTTP 200. Their client latencies were **58.199, 19.693, 17.419, 19.682, 18.295, 17.130, 17.371, 17.417, 16.983, and 17.020 ms**. The shell loop took **0.289 s**; the stop-finish counter rose from **1 to 11**, and the end-to-end latency histogram count reached **11**.
4. A later running-service snapshot at **01:26:51 CST** showed **5,524 MiB** device-wide VRAM used, **23%** GPU activity, and **59°C**. `nvidia-smi --query-compute-apps` attributed **4,864 MiB** to `VLLM::EngineCore` (PID 55211). `free -h` showed **31 GiB** system RAM total, **16 GiB** available, and **0 B** swap used. A post-request metrics scrape showed **0** requests running, **0** waiting, and **0** KV-cache usage at that instant.
5. For a later single HTTP 200 request, vLLM's histogram count rose **13 → 14**, and its cumulative latency sum rose **0.3252289295 → 0.3825705051 seconds**. The matching client elapsed time was **60.765 ms**.

## Calculations and interpretation

| Question | Calculation | What it means |
| --- | --- | --- |
| How much did device-wide VRAM use rise? | `5,524 − 633 = 4,891 MiB` | Difference between two whole-device snapshots taken at different times. |
| How much GPU memory was listed for EngineCore? | `4,864 MiB` | Per-process attribution in the running snapshot; a different quantity from the 4,891 MiB device-wide change. |
| What was the short batch's observed request rate? | `10 / 0.289 ≈ 34.6 requests/s` | Rate for this ten-request sequential shell loop, not sustained capacity. |
| What were the client latency summaries? | Mean **21.921 ms**; median **17.418 ms** | The first request was noticeably slower; its cause was not isolated. |
| What was the server histogram increment for the final request? | `(0.3825705051 − 0.3252289295) / (14 − 13) ≈ 57.342 ms` | One request's measured vLLM end-to-end latency if no other request was recorded between scrapes. |
| How did that compare with the client? | `60.765 − 57.342 ≈ 3.423 ms` | Extra time outside the server metric's timing boundary; it cannot all be assigned to network delay. |

An idle `num_requests_waiting=0` or `kv_cache_usage_perc=0` snapshot describes the time of the scrape. It does not prove there was no waiting or cache use during earlier requests. Likewise, 10 successful requests establish an observed **0/10 HTTP failure fraction for this batch**, not long-term reliability.

## Reproduce the checks

Run these commands **on the host where vLLM is already listening on port 8000**. The model must be available under the served model name shown below.

```bash
# Host and GPU snapshots
date -Is
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
free -h

# Service health and current application metrics
curl -i http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/metrics |
  grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|request_success_total|e2e_request_latency_seconds_(count|sum))'

# Before/after latency experiment: record count and sum before one request.
curl -fsS http://127.0.0.1:8000/metrics |
  grep -E '^vllm:e2e_request_latency_seconds_(count|sum)'

curl -sS -o /dev/null \
  -w 'http=%{http_code} client_seconds=%{time_total}\n' \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Reply with one word: ready"}],"max_tokens":8,"temperature":0}'

# Record count and sum again, then subtract the earlier values.
curl -fsS http://127.0.0.1:8000/metrics |
  grep -E '^vllm:e2e_request_latency_seconds_(count|sum)'
```

Counters and histogram sums are cumulative for the current service process; compare **changes between two scrapes** and account for restarts. For a valid single-request difference, ensure no other client sends a request between the scrapes. `curl`'s client elapsed time and vLLM's server metric have different measurement boundaries.

## Monitoring concepts practiced

- **Monitoring versus observability:** collect known measurements over time, then use metrics, event logs, and, when enabled, traces to investigate unexpected behavior.
- **Four golden signals:** latency, traffic, errors, and saturation. GPU utilization alone does not answer whether requests succeeded or how long they took.
- **Infrastructure versus application signals:** device memory, process memory, system RAM, requests in progress, waiting queue, cache usage, and completion counters answer different questions.
- **Observation versus conclusion:** report the measured value and time first, then state what the measurement can support and what remains uncertain.

## Assessment and next step

Reported results: **Section 2: 10/10; Section 3: 10/10; Section 4: 9/10; Day 56 final assessment: 10/10.** No Section 1 score was reported. The Section 4 correction was **4,891 MiB device-wide increase** versus **4,864 MiB EngineCore allocation**.

**Day 57 handoff:** set up time-series collection of the vLLM metrics and use counter rates and history to answer questions that one-off `curl` scrapes cannot.

Full explanations and tables: [Day 56 student notes (PDF)](AI_Infra_Day56_Student_Notes.pdf) · [editable LaTeX](AI_Infra_Day56_Student_Notes.tex).

## References

- [Google SRE: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [OpenTelemetry: Observability primer](https://opentelemetry.io/docs/concepts/observability-primer/)
- [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [NVIDIA `nvidia-smi` documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
