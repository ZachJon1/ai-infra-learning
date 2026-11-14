# Day 60 — Observability Incident Runbook

## Incident A: Grafana says “No data” for every panel

1. Check Grafana `curl -fsS http://127.0.0.1:3000/api/health`.
2. Check Prometheus `curl -fsS http://127.0.0.1:9090/-/ready`.
3. In Grafana, confirm Prometheus data-source URL (`http://127.0.0.1:9090` for *this* host-networked setup) and use **Save & test**.
4. Query `up` in Grafana Explore; compare with Prometheus UI/API.
5. Inspect dashboard time range and Query inspector. Do not immediately restart every service.

## Incident B: VRAM panel works, GPU utilization panel is blank

1. Check exporter: `curl -fsS http://127.0.0.1:9400/metrics | grep '^DCGM_FI_DEV_GPU_UTIL'`.
2. Query `DCGM_FI_DEV_GPU_UTIL{job="dcgm-exporter"}` in Prometheus.
3. In Grafana verify exact query spelling, Range mode, time range, legends, and transformations/overrides that might hide a series.
4. The correct GPU utilization legend is `GPU {{gpu}}` on this setup.
5. Use **Absolute** threshold mode for an illustrative warning line at **80%**, not relative/Percentage mode.

## Incident C: `up{job="vllm"}=1` but latency is blank

1. Confirm the inference API is reachable: `curl -fsS http://127.0.0.1:8000/health`.
2. Check histogram count: `curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:e2e_request_latency_seconds_count'`.
3. Produce a small, authorized test workload and wait at least one Prometheus scrape interval.
4. Query `rate(vllm:e2e_request_latency_seconds_count[5m])` and inspect time range. Zero completions can leave a zero denominator in a mean-latency expression.
5. Avoid equating a live scrape endpoint with a successful or timely inference endpoint.

## Incident D: p95 differs from individual client latencies

1. Confirm whether you are comparing **server-side** histogram latency to **client-side** wall-clock latency.
2. Compare the same time range and same request population; bucket interpolation can be coarse.
3. Inspect the histogram's available `le` buckets and total sample count.
4. Use repeated, controlled tests before drawing conclusions about tail latency.

## Incident E: After a laptop reboot, metrics targets are DOWN

1. `docker ps -a`; check Prometheus and Grafana Compose projects.
2. If required: `minikube start -p gpu-lab` for the Kubernetes lab profile.
3. Restore DCGM exporter and vLLM using their *previously tested* Day 59 and model-serving configurations.
4. Query `sum(up)` and individual `up{job="..."}` labels.
5. Confirm restart policies, stable persistent volumes, and avoid treating old `Exited (0)` containers as application crashes.

## Interview scenarios (practice explaining aloud)

1. **Target UP, user latency bad.** Which measurements would you correlate, and why does scrape health not prove service health?
2. **GPU memory stays at 5+ GiB, utilization drops near zero.** What hypothesis fits a loaded, idle model? What additional measurements distinguish idle from memory leak?
3. **p95 suddenly increases during a concurrent load test.** How would you distinguish queueing, compute saturation, memory pressure, and application errors?
4. **Grafana breaks after a reboot.** Explain persistent storage, restart policy, service ordering, and how you would make recovery reproducible.

### Evidence standard

Log observed commands, exact errors, diagnosis, repair, and verification. Avoid asserting root causes or hardware efficiency without supporting measurements.
