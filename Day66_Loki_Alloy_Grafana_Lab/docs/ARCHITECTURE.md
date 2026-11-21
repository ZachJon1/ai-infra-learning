# Architecture and Operational Design — Day 66

## Scope

This is a single Ubuntu workstation running an NVIDIA GPU and Docker. The served model is Qwen2.5-0.5B-Instruct, with a local-only OpenAI-compatible HTTP API. The logging system is a separate Docker Compose service group. Other existing observability processes (Prometheus, Grafana, DCGM exporter) were observed to be running, but are *not* asserted to be connected to the new Loki data source.

## Data flow

```text
client -> localhost:8000 -> Docker vLLM container -> NVIDIA GPU
                               | stdout/stderr
                               v
                         Docker log driver
                               | Docker API (local socket)
                               v
                        Grafana Alloy
          discovery.docker -> discovery.relabel -> loki.source.docker
                               |
                         loki.write
                               |
                               v
                         Grafana Loki
                      label index + chunks
                               ^
                     Grafana Explore / LogQL
```

`compose.vllm.yaml` overlays the original Compose file. Alloy's effective configuration is `config.docker.alloy`, which **retains** the synthetic `lab.log` file source and adds Docker discovery restricted to `/day66-vllm`.

## Ports and trust boundaries

| Host endpoint | Inside container | Function | Exposure |
|---|---|---|---|
| 127.0.0.1:8000 | vLLM:8000 | Inference / health / metrics | Local |
| 127.0.0.1:13100 | Loki:3100 | Log query and ingestion API | Local |
| 127.0.0.1:13000 | Grafana:3000 | Explore / dashboards | Local |
| 127.0.0.1:12346 | Alloy:12345 | Collector diagnostics | Local |

The Docker socket is an **elevated local trust boundary**: the Alloy container can access a highly privileged daemon interface. A read-only bind does not make the socket API read-only. This is not a recommended multi-tenant production security design.

## Labels and metadata

The Docker source publishes `app=vllm`, `source=docker`, `job=ai-infra`, `environment=lab`. Automatic `service_name` and detected severity metadata may appear in query output. Stable labels describe a stream; per-request values such as request IDs should not become indexed labels. Plain-text vLLM stdout is not assumed to be JSON, and `| json` should be reserved for JSON-formatted log lines.

## Failure domains and instrumentation

| Failure | Expected observable symptom | Independent verification |
|---|---|---|
| vLLM API fails | `/health` non-200 or no response; requests fail | `docker ps`, `docker logs`, network probe |
| Alloy stops | New entries absent from Loki while Docker logs advance | Compare `docker logs` to Loki; `docker ps` |
| Loki unavailable | Collector write errors, Grafana query failures | `/ready`, Alloy logs, Loki logs |
| Grafana unavailable | UI down while Loki API may still work | Query Loki API directly |
| GPU OOM | Inference request errors or crashes may be recorded | `nvidia-smi`, DCGM, vLLM logs, container state |
| Docker log rotation | Old outage entries no longer in local logs | Examine log rotation and retention settings |

## Incident record: confirmed vs inferred

1. **HTTP 404 experiment:** `day66-intentional-404` was generated deliberately; its access log was visible in Loki under an INFO severity. This proves that a 4xx status need not be an ERROR-level log.
2. **Collector outage:** stopping `day66-loki-alloy-1` did not stop `day66-vllm`. Docker still held `day66-alloy-outage-test` in its output.
3. **Recovery:** after Alloy restarted, Grafana displayed both the outage-test and recovered-test log lines. This verifies successful backfill for that controlled interruption.
4. **Limitation:** screenshots did not show a separately executed *during-outage* Loki query; avoid claiming that specific observation. This incident does not prove recovery across longer outages or log rotation.

## Production hardening backlog

- Replace unauthenticated Loki with controlled access, TLS, proper tenancy, and least-privileged collection.
- Integrate Loki, Prometheus and DCGM sources with one reviewed Grafana operations dashboard.
- Define storage retention; alert on collector ingestion lag, dropped log entries, and storage exhaustion.
- Separate audit/access/inference logs, add request correlation without high-cardinality indexed labels.
- Validate failure recovery under retention pressure and make incident evidence reproducible.
- Consider Kubernetes node log collection and security boundaries for a multi-node deployment.
