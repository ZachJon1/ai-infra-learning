import argparse
import csv
import json
import math
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--requests", type=int, default=20)
parser.add_argument("--concurrency", type=int, default=2)
args = parser.parse_args()

if args.requests < 1 or args.concurrency < 1:
    parser.error("requests and concurrency must be positive")

payload = {
    "model": MODEL,
    "messages": [{
        "role": "user",
        "content": "Explain what a GPU does in one sentence."
    }],
    "max_tokens": 32,
    "temperature": 0,
    "stream": False
}

def send_request(i):
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            data = json.load(resp)

        elapsed = (time.perf_counter() - start) * 1000

        return {
            "request": i,
            "status": status,
            "latency_ms": round(elapsed, 3),
            "completion_tokens": data["usage"]["completion_tokens"],
            "error": ""
        }

    except Exception as exc:
        return {
            "request": i,
            "status": 0,
            "latency_ms": round(
                (time.perf_counter() - start) * 1000, 3
            ),
            "completion_tokens": 0,
            "error": str(exc)
        }

start = time.perf_counter()

with ThreadPoolExecutor(
    max_workers=args.concurrency
) as pool:
    futures = [
        pool.submit(send_request, i)
        for i in range(1, args.requests + 1)
    ]
    rows = [f.result() for f in as_completed(futures)]

wall_seconds = time.perf_counter() - start
rows.sort(key=lambda r: r["request"])

filename = (
    f"day62_concurrency_{args.concurrency}.csv"
)

with open(filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(
        f"Request {row['request']:02d}: "
        f"{row['latency_ms']:.3f} ms "
        f"status={row['status']}"
    )

successes = [
    r["latency_ms"]
    for r in rows
    if r["status"] == 200
]
successes.sort()

print("\n=== CONCURRENT BENCHMARK ===")
print(f"Concurrency:  {args.concurrency}")
print(f"Sent:         {args.requests}")
print(f"Successful:   {len(successes)}")
print(f"Failed:       {args.requests-len(successes)}")
print(f"Wall time:    {wall_seconds:.3f} s")
print(
    f"Throughput:   "
    f"{len(successes)/wall_seconds:.3f} req/s"
)

if successes:
    def percentile(p):
        index = math.ceil(p * len(successes)) - 1
        return successes[index]

    print(f"Mean:         {statistics.mean(successes):.3f} ms")
    print(f"Median:       {statistics.median(successes):.3f} ms")
    print(f"p95:          {percentile(0.95):.3f} ms")
    print(f"p99:          {percentile(0.99):.3f} ms")
    print(f"Min:          {min(successes):.3f} ms")
    print(f"Max:          {max(successes):.3f} ms")

print(f"\nSaved: {filename}")
