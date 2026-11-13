#!/usr/bin/env bash
set -euo pipefail
printf '\n[1/5] GPU host detection\n'
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv || true
printf '\n[2/5] Containers\n'
docker ps --filter name=day59-dcgm-exporter --filter name=day57-prometheus
printf '\n[3/5] DCGM endpoint\n'
curl -fsS http://127.0.0.1:9400/metrics | grep -E '^DCGM_FI_DEV_(GPU_UTIL|FB_USED|GPU_TEMP|POWER_USAGE)\{' || true
printf '\n[4/5] Prometheus target\n'
curl -fsSG http://127.0.0.1:9090/api/v1/query --data-urlencode 'query=up{job="dcgm-exporter"}' | python3 -m json.tool
printf '\n[5/5] Config validation\n'
docker exec day57-prometheus promtool check config /etc/prometheus/prometheus.yml
