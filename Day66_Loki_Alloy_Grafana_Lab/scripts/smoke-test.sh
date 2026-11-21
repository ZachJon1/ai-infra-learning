#!/usr/bin/env bash
set -euo pipefail
BASE="http://127.0.0.1"
printf '%s\n' '== Day 66 health checks =='
curl -fsS "$BASE:13100/ready"
printf '\n'
curl -fsS "$BASE:8000/health"
printf '\nvLLM health OK\n'
printf '%s\n' '== Real Docker vLLM logs in Loki (recent hour) =='
curl -fsSG "$BASE:13100/loki/api/v1/query_range" \
 --data-urlencode 'query={app="vllm",source="docker"}' \
 --data-urlencode 'since=1h' | python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get("data",{}).get("result",[]); print("Loki status:", d.get("status"), "streams:",len(r), "entries:",sum(len(x.get("values",[])) for x in r)); sys.exit(0 if d.get("status")=="success" and r else 1)'
