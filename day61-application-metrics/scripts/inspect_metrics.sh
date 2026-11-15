#!/usr/bin/env bash
set -euo pipefail
PROM="${PROM:-http://127.0.0.1:9091}"
echo '--- Local gateway metrics ---'
curl -fsS http://127.0.0.1:8001/metrics | grep -E '^app_inference_' | head -30
echo '--- Prometheus target ---'
curl -fsSG "$PROM/api/v1/query" --data-urlencode 'query=up{job="day61-gateway"}' | python3 -m json.tool
echo '--- Cumulative requests ---'
curl -fsSG "$PROM/api/v1/query" --data-urlencode 'query=sum(app_inference_requests_total)' | python3 -m json.tool
echo '--- 5-minute request rate ---'
curl -fsSG "$PROM/api/v1/query" --data-urlencode 'query=sum(rate(app_inference_requests_total[5m]))' | python3 -m json.tool
