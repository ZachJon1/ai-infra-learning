# Day 64 operational runbook

## Service dependency chain

GPU & vLLM -> DCGM exporter and `/metrics` -> Prometheus -> Grafana.
Prometheus and Grafana are already running in Docker **host-network** mode.

## Known good environment

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8 GiB, NVIDIA driver 580.196.07.
- DCGM exporter image: `nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4`, container `day64-dcgm-exporter`, host port 9400.
- vLLM 0.33.0, model `Qwen/Qwen2.5-0.5B-Instruct`, API port 8000.
- Existing `day57-prometheus`: host port 9090, config bind `/home/zakaria/ai-infra/day57-prometheus/prometheus.yml` -> `/etc/prometheus/prometheus.yml`.
- Existing `day60-grafana`: host port 3000.
- Existing jobs: `prometheus`, `vllm`, `dcgm-exporter` (15s scrapes). No new jobs or rename required.

## Triage sequence

1. Confirm services:
   ```bash
   docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
   ss -lnt | grep -E ':(8000|9400|9090|3000)\b'
   ```
2. Confirm raw exporters and server metrics:
   ```bash
   curl -fsS http://127.0.0.1:9400/metrics | grep -E '^DCGM_FI_DEV_(GPU_UTIL|FB_USED|GPU_TEMP|POWER_USAGE)' | head
   curl -fsS http://127.0.0.1:8000/metrics | grep -E '^vllm:' | head
   ```
3. Check Prometheus scrape targets (actual job labels):
   ```bash
   curl -fsSG --data-urlencode 'query=up{job=~"prometheus|vllm|dcgm-exporter"}' \
       http://127.0.0.1:9090/api/v1/query | python3 -m json.tool
   ```
4. If `up=1` but a Grafana panel is blank, check the panel's exact PromQL metric name, `job` label, data source UID, time range, dashboard variables, and scrape interval.
5. If `up=0`, inspect target errors and the exporter/server's logs. `docker logs --tail 80 day64-dcgm-exporter` is helpful.
6. If DCGM reports `Not collecting DCP metrics`, expect some unsupported profiling counters on this GeForce GPU; core GPU metrics were **successfully emitted** in this lab. Do not misdiagnose this warning as the whole exporter failing.
7. If vLLM has stopped, reactivate the *existing* environment (`activate-ai` was available in this session) and start the small model using the command below. Avoid duplicate vLLM instances on the 8 GiB GPU.

## Restart vLLM (user's established environment)

```bash
activate-ai
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype auto \
  --max-model-len 2048 --gpu-memory-utilization 0.60
```

## Is GPU performance throttled?

Do **not** conclude throttling from 83 C or power reducing from ~125 W to ~55 W alone. Log concurrent utilization, GPU SM/memory clocks, temperature, power, and supported throttle/performance-limit reasons *during* a repeatable load:

```bash
nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE
nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw \
  --format=csv -l 2
```

Check whether the exact fields are supported by the installed driver. If GPU temperature is uncomfortably high or stability is affected, stop the workload and let hardware cool. Laptop cooling, device power policy, and P-states can affect measurements.

## Interpreting queue and throughput

- `GPU_UTIL=100%`, `waiting=0`: busy device; **not sufficient** to prove request saturation.
- Increasing concurrency + flat throughput + increasing queue + rising p95: strong indication of a capacity limit, but locate the bottleneck before choosing hardware scaling.
- VRAM high while GPU idle: often reserved weights/KV cache; check model and cache use before calling it a leak.
- `up=1`, Grafana No data: investigate metric naming/datasource/labels/time window, not necessarily service availability.

## Data integrity

The supplied `evidence/gpu-samples.csv` is transcribed from the user's session and the preserved `evidence/load-test.txt` contains exact reported summary statistics. Reconstructed dashboard JSON is a template, **not** the export of the live dashboard. Preserve the actual Grafana export separately if needed.
