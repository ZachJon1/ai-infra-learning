# Day 61 — Observed terminal evidence and remaining validation

All entries labelled **observed** are grounded in the terminal output the learner shared. We did not remotely execute these commands or inspect the learner's local filesystem.

## Observed — healthy first request

- `GET http://127.0.0.1:8001/metrics` before inference: `app_inference_request_duration_seconds_count 0`, `_sum 0`, in-flight Gauge `0`.
- Model ID returned by vLLM: `Qwen/Qwen2.5-0.5B-Instruct`.
- First `POST :8001/v1/chat/completions` returned `HTTP 200`.
- After request: `app_inference_requests_total{status_code="200"} 1.0`, in-flight Gauge `0.0`, histogram `_count 1.0`, `_sum 0.15389483899980405` seconds.
- The histogram bucket `le="0.1"` was `0` and `le="0.25"` was `1` (consistent with a 153.9 ms observation).

## Observed — Prometheus scrape and ten-request smoke test

- Isolated Prometheus base URL used: `http://127.0.0.1:9091`.
- `up{job="day61-gateway"}` was `1` for instance `127.0.0.1:8001`.
- Ten sequential requests returned HTTP 200 with these **client-side** durations (seconds):

```text
0.126503
0.112698
0.110379
0.127945
0.124285
0.122681
0.110411
0.111094
0.110220
0.110736
```

- Mean client-side duration: `0.1166952 s`; minimum `0.110220 s`; maximum `0.127945 s`.
- `sum(app_inference_requests_total)` returned `11`.
- `sum(rate(app_inference_requests_total[5m]))` returned `0.03389830508474576 requests/s` (about `2.0339/min`).

## Observed — deliberate failures

- `up{job="day61-unreachable"}` returned `0` for instance `127.0.0.1:18798`.
- `up{job="day61-gateway"}` returned `1` for instance `127.0.0.1:8001`.
- `docker kill --signal=HUP day61-prometheus` returned `day61-prometheus`, confirming that the signal was sent, **not** that a particular YAML change was loaded.
- A separate Uvicorn gateway process bound to `127.0.0.1:8002`, configured to forward to `127.0.0.1:18799`, logged a POST with HTTP `502 Bad Gateway`.
- The same faulty gateway logged `GET /metrics` with HTTP `200 OK`.
- That faulty gateway was subsequently stopped with `Ctrl+C`.

## Not yet observed — follow-up evidence

1. The exact series `app_inference_requests_total{status_code="502"}` for the faulty process was not pasted before its shutdown. Re-run the faulty process and capture the `:8002/metrics` endpoint **before** stopping it.
2. A HUP reload was signalled, but no active-targets response verifying that `day61-unreachable` was removed from the YAML configuration was provided.
3. No measured p95/p99 inference latency, token throughput, concurrent-load capacity, Grafana screenshots or long-running SLO data has been supplied yet.

## Evidence capture template

```bash
# Rerun faulty gateway, issue one request, THEN capture metrics while it runs:
curl -fsS http://127.0.0.1:8002/metrics | grep 'app_inference_requests_total'

# Confirm active targets after removing unreachable job and reloading:
curl -fsS http://127.0.0.1:9091/api/v1/targets | \
  python3 -c 'import json,sys; d=json.load(sys.stdin); [print(t.get("labels",{}).get("job"), t.get("health")) for t in d["data"]["activeTargets"]]'
```
