import csv
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N = 10

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "Explain what a GPU does in one sentence."
        }
    ],
    "max_tokens": 32,
    "temperature": 0,
    "stream": False
}

rows = []

for i in range(1, N + 1):
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            data = json.load(response)

        latency_ms = (time.perf_counter() - start) * 1000
        usage = data.get("usage", {})
        finish = data["choices"][0]["finish_reason"]

        rows.append({
            "request": i,
            "status": status,
            "latency_ms": round(latency_ms, 3),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "finish_reason": finish
        })

        print(f"Request {i:02d}: {latency_ms:.3f} ms")

    except Exception as exc:
        print(f"Request {i:02d} failed: {exc}")
        raise

output = Path("day62_baseline.csv")

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

latencies = sorted(r["latency_ms"] for r in rows)

def percentile(p):
    # Nearest-rank percentile, consistent with Section 2
    return latencies[math.ceil(p * len(latencies)) - 1]

print("\n=== Baseline latency summary ===")
print(f"Requests:  {len(latencies)}")
print(f"Mean:      {statistics.mean(latencies):.3f} ms")
print(f"Median:    {statistics.median(latencies):.3f} ms")
print(f"p95:       {percentile(0.95):.3f} ms")
print(f"p99:       {percentile(0.99):.3f} ms")
print(f"Minimum:   {min(latencies):.3f} ms")
print(f"Maximum:   {max(latencies):.3f} ms")
print(f"\nSaved: {output.resolve()}")
