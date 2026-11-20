# Incident report — Inference gateway pointed to incorrect backend port

**Date of test:** 2026-11-19 (America/Chicago); corresponding log timestamps were on 2026-11-20 UTC.  
**Impact scope:** Deliberate fault injection on an isolated learning gateway (port 8002); production services were not modified.  
**Status:** End-to-end functional recovery **verified**. Final recovered structured-log record **not fully verified** because the pasted output ends mid-line.

## Symptoms

- Gateway `GET /health` on `127.0.0.1:8002` returned HTTP 200.
- Actual vLLM `GET /health` on `127.0.0.1:8000` returned HTTP 200.
- An inference request sent through the gateway returned **HTTP 502** with `{"detail":"Inference backend unavailable"}`.
- The gateway was configured to contact `127.0.0.1:8999`, with no server listening there in the intended test setup.

## Key evidence (transcribed from terminal output)

| Field | Failure | Recovery |
|---|---|---|
| Gateway upstream URL | `127.0.0.1:8999` | `127.0.0.1:8000` |
| Client HTTP status | 502 | 200 |
| Client latency | 29.794 ms | 70.697 ms |
| Gateway latency | 28.85 ms | Not verified from complete log |
| Gateway error class | `ConnectError` | No client-facing error |
| Model output | None | `Ready.` |
| Prompt/completion tokens | N/A | 35 / 3 |

**Correlated failure request ID:** `fed0b99a-4c07-47cd-8782-1541ee65dee9`.

```json
{"timestamp":"2026-11-20T03:56:35.976537+00:00","level":"ERROR","service":"inference-gateway","event":"upstream_connection_failed","request_id":"fed0b99a-4c07-47cd-8782-1541ee65dee9","error_type":"ConnectError"}
{"timestamp":"2026-11-20T03:56:35.976732+00:00","level":"ERROR","service":"inference-gateway","event":"http_request_completed","request_id":"fed0b99a-4c07-47cd-8782-1541ee65dee9","method":"POST","path":"/v1/chat/completions","status_code":502,"duration_ms":28.85}
```

The log excerpts are normalized in whitespace for readability. They represent observed records, not new benchmark runs.

## Diagnosis

The healthy gateway tried to contact the **wrong local port**. A connection exception was raised before an upstream HTTP response could be obtained. The gateway caught that exception and returned HTTP 502. This is not evidence of a GPU/driver fault or invalid model alias.

This differs from the earlier test with `nonexistent-model`: that request received **HTTP 404** and a `NotFoundError` response from a reachable vLLM server, with a structured `upstream_http_error` event.

## Remediation

1. Stopped only the isolated gateway on port 8002.
2. Restarted that gateway with `VLLM_URL=http://127.0.0.1:8000/v1/chat/completions`.
3. Used `tee -a breakfix.log` to retain the pre-repair evidence.
4. Re-sent the same valid model inference request.

## Verification

- HTTP response: **200**.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Assistant content: **`Ready.`**.
- Client response time: **0.070697 seconds**.
- **Outstanding:** Check `tail -n 1 breakfix.log | python3 -m json.tool` on the local host. The user-pasted log line was cut off after the `path` field, so the final stored JSON event's completeness and `duration_ms` have not yet been independently confirmed.

## Preventive controls (proposed, not implemented)

- Validate backend URL and model alias during startup or deployment.
- Maintain distinct liveness and dependency-aware readiness endpoints with bounded timeouts.
- Log gateway request IDs and forward correlation context to downstream services.
- Centralize logs with security-aware retention policies; use high-cardinality IDs as searchable log fields rather than unrestricted Loki index labels.
- Run automated smoke tests that verify both health and a minimal inference request after a configuration change.

## Interview-ready summary

A healthy gateway and healthy model server do not imply that their configured route works. By correlating structured connection errors with 502 request logs and comparing the configured upstream address against the actual serving port, I isolated a configuration fault and restored end-to-end inference without restarting the GPU backend.
