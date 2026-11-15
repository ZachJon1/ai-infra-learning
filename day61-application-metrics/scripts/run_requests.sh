#!/usr/bin/env bash
set -euo pipefail
MODEL="$(curl -fsS http://127.0.0.1:8000/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')"
for i in $(seq 1 10); do
  curl -sS -o /dev/null -w "Request ${i}: HTTP %{http_code}, duration=%{time_total}s\n" \
    http://127.0.0.1:8001/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is GPU inference?\"}],\"max_tokens\":30,\"stream\":false}"
done
