import argparse
import csv
import json
import math
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser()

parser.add_argument(
    "--url",
    default="http://127.0.0.1:8002/predict"
)
parser.add_argument("--requests", type=int, default=300)
parser.add_argument("--output", default="gpu_baseline.csv")

args = parser.parse_args()

payload = json.dumps({
    "features": [1.0, 2.0, 3.0]
}).encode()

CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]
WARMUP_REQUESTS = 20


def send_request():
    request = Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    try:
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read())
            if "prediction_value" not in data:
                raise ValueError("Invalid prediction response")

        elapsed = (time.perf_counter() - start) * 1000
        return True, elapsed

    except Exception:
        return False, (time.perf_counter() - start) * 1000


def percentile(values, p):
    values = sorted(values)
    index = max(0, math.ceil(len(values) * p / 100) - 1)
    return values[index]


def benchmark(concurrency):
    print(f"Testing concurrency={concurrency}...")

    # Warm up before each measurement
    for _ in range(WARMUP_REQUESTS):
        send_request()

    latencies = []
    errors = 0

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(send_request)
            for _ in range(args.requests)
        ]

        for future in as_completed(futures):
            success, latency = future.result()

            if success:
                latencies.append(latency)
            else:
                errors += 1

    duration = time.perf_counter() - start

    if not latencies:
        raise RuntimeError("No successful requests")

    result = {
        "concurrency": concurrency,
        "throughput_rps": round(len(latencies) / duration, 2),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "errors": errors,
        "error_rate_pct": round(100 * errors / args.requests, 2)
    }

    return result


results = []

for concurrency in CONCURRENCY_LEVELS:
    result = benchmark(concurrency)
    results.append(result)
    print(result)
    print()


with open(args.output, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {args.output}")