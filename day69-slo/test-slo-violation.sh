#!/usr/bin/env bash

PROM="http://localhost:9090"
WINDOW="${1:-30m}"

GOOD='sum(increase(vllm:time_to_first_token_seconds_bucket{job="vllm",le="0.005"}['"$WINDOW"']))'

TOTAL='sum(increase(vllm:time_to_first_token_seconds_count{job="vllm"}['"$WINDOW"']))'

SLI="($GOOD / $TOTAL)"

query() {
    printf '\n%s\n' "$1"

    curl -fsSG "$PROM/api/v1/query" \
      --data-urlencode "query=$2" |
    python3 -c '
import json, sys
results = json.load(sys.stdin)["data"]["result"]
for r in results:
    print("Value:", r["value"][1])
if not results:
    print("No data")
'
}

query "Total observations" "$TOTAL"

query "TTFT compliance within 5 ms (%)" \
    "100 * $SLI"

query "Bad event percentage" \
    "100 * (1 - $SLI)"

query "Error budget consumed (%)" \
    "100 * (1 - $SLI) / 0.01"

query "Remaining error budget (%)" \
    "100 * (1 - (1 - $SLI) / 0.01)"

query "Burn rate" \
    "(1 - $SLI) / 0.01"

