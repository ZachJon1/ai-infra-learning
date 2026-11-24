#!/usr/bin/env bash

PROM="http://localhost:9090"
WINDOW="${1:-30m}"

query() {
    printf '\n%s\n' "$1"

    curl -fsSG "$PROM/api/v1/query" \
        --data-urlencode "query=$2" |
    python3 -c '
import json, sys

data = json.load(sys.stdin)
results = data["data"]["result"]

if not results:
    print("No data")
else:
    for result in results:
        print("Value:", result["value"][1])
'
}

query "HTTP availability (%)" \
"100 * sum(increase(http_requests_total{job=\"vllm\",handler=\"/v1/chat/completions\",method=\"POST\",status=\"2xx\"}[$WINDOW])) / sum(increase(http_requests_total{job=\"vllm\",handler=\"/v1/chat/completions\",method=\"POST\",status=~\"2xx|5xx\"}[$WINDOW]))"

for item in \
    "TTFT:vllm:time_to_first_token_seconds:0.5:0.01" \
    "E2E:vllm:e2e_request_latency_seconds:2.5:0.05"
do
    IFS=: read -r NAME PREFIX METRIC LE BUDGET <<< "$item"
    METRIC="$PREFIX:$METRIC"

    GOOD="sum(increase(${METRIC}_bucket{job=\"vllm\",le=\"$LE\"}[$WINDOW]))"
    TOTAL="sum(increase(${METRIC}_count{job=\"vllm\"}[$WINDOW]))"
    SUM="sum(increase(${METRIC}_sum{job=\"vllm\"}[$WINDOW]))"

    query "$NAME observation count" "$TOTAL"

    query "$NAME compliance (%)" \
        "100 * ($GOOD / $TOTAL)"

    query "$NAME average latency (ms)" \
        "1000 * ($SUM / $TOTAL)"

    query "$NAME error budget consumed (%)" \
        "100 * (1 - ($GOOD / $TOTAL)) / $BUDGET"

    query "$NAME burn rate" \
        "(1 - ($GOOD / $TOTAL)) / $BUDGET"

done
