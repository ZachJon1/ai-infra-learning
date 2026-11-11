#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
REQUESTS="${REQUESTS:-20}"
DELAY="${DELAY:-2}"

if ! [[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REQUESTS must be a positive integer" >&2
  exit 2
fi

success=0
for ((i=1; i<=REQUESTS; i++)); do
  payload=$(printf '{"model":"%s","messages":[{"role":"user","content":"Explain GPU memory in one sentence."}],"max_tokens":32,"temperature":0}' "$MODEL")
  status=$(curl -sS -o /dev/null -w '%{http_code}' \
    "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data "$payload") || status="curl_error"
  echo "Request $i: HTTP $status"
  [[ "$status" == "200" ]] && ((success+=1)) || true
  sleep "$DELAY"
done
printf 'Successful: %d/%d\n' "$success" "$REQUESTS"
[[ "$success" -eq "$REQUESTS" ]]
