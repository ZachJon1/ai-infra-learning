#!/usr/bin/env bash

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
URL="http://127.0.0.1:8000/v1/chat/completions"
REQUESTS="${1:-30}"

for ((i=1; i<=REQUESTS; i++)); do

    printf "Request %02d: " "$i"

    curl -sS \
      --max-time 30 \
      -o /dev/null \
      -w 'http=%{http_code} latency=%{time_total}s\n' \
      "$URL" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL\",
        \"messages\": [
          {
            \"role\": \"user\",
            \"content\": \"Explain Kubernetes briefly. Example $i\"
          }
        ],
        \"max_tokens\": 32,
        \"temperature\": 0
      }"

    sleep 1

done
