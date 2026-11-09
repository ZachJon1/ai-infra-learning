#!/usr/bin/env python3
"""Minimal OpenAI-compatible vLLM local smoke test (Python stdlib)."""
import json
import time
import urllib.request

base = "http://127.0.0.1:8000"
with urllib.request.urlopen(base + "/v1/models", timeout=10) as resp:
    model = json.load(resp)["data"][0]["id"]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Explain Kubernetes GPU scheduling in two sentences."}],
    "temperature": 0,
    "max_tokens": 100,
    "stream": False,
}
request = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
start = time.perf_counter()
with urllib.request.urlopen(request, timeout=60) as resp:
    response = json.load(resp)
print("Model:", model)
print("Latency (s):", round(time.perf_counter() - start, 3))
print("Usage:", response.get("usage", {}))
print("Answer:", response["choices"][0]["message"]["content"])
