# Day 64 Grafana panel queries (actual existing job names)

| Panel | PromQL | Unit |
|---|---|---|
| GPU utilization | `DCGM_FI_DEV_GPU_UTIL{job="dcgm-exporter"}` | percent (0-100) |
| GPU VRAM used | `DCGM_FI_DEV_FB_USED{job="dcgm-exporter"} / 1024` | GiB (exported metric in MiB) |
| GPU temperature | `DCGM_FI_DEV_GPU_TEMP{job="dcgm-exporter"}` | Celsius |
| GPU power draw | `DCGM_FI_DEV_POWER_USAGE{job="dcgm-exporter"}` | watts |
| Completed req/sec | `sum(rate(vllm:e2e_request_latency_seconds_count{job="vllm"}[2m]))` | req/s |
| Mean server latency | `sum(rate(vllm:e2e_request_latency_seconds_sum{job="vllm"}[2m])) / sum(rate(vllm:e2e_request_latency_seconds_count{job="vllm"}[2m]))` | seconds |
| Waiting requests | `sum(vllm:num_requests_waiting{job="vllm"})` | requests |
| Optional: GPU KV cache usage | `100 * max(vllm:kv_cache_usage_perc{job="vllm"})` | percent |
| Optional: generation token rate | `sum(rate(vllm:generation_tokens_total{job="vllm"}[2m]))` | tokens/s, only if this exact metric is exposed |
| Optional: server p95 | `histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{job="vllm"}[5m])))` | seconds, only if histogram buckets exported |

**Diagnostics:** `up{job=~"prometheus|dcgm-exporter|vllm"}` should be 1 for each job. `up=1` only proves Prometheus can scrape a target, not that a particular metric is present. Counter-based rates can be absent or volatile before sufficient samples arrive.

**Important:** The 2-minute PromQL rolling window is not the same estimator as the 90.62-second benchmark's overall client-side mean/p95. GPU utilization is a sampling-interval metric, not a percentage of VRAM occupancy. On the tested RTX 3070, advanced DCGM profiling `DCGM_FI_PROF_*` fields were unavailable.
