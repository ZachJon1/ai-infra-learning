
import json
import math
import statistics
import time
from urllib.request import Request, urlopen

SERVERS = {
    "CPU": "http://127.0.0.1:8003/predict",
    "GPU": "http://127.0.0.1:8002/predict",
}

WARMUP = 20
REQUESTS = 300

payload = json.dumps({
    "features": [1.0, 2.0, 3.0]
}).encode("utf-8")


def send_request(url):
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urlopen(request, timeout=10) as response:
        result = json.load(response)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    assert math.isclose(
        result["prediction_value"],
        2.4,
        abs_tol=1e-5,
    )

    return elapsed_ms


def benchmark(name, url):
    print(f"\nBenchmarking {name}...")

    for _ in range(WARMUP):
        send_request(url)

    latencies = []

    start = time.perf_counter()

    for _ in range(REQUESTS):
        latencies.append(send_request(url))

    duration = time.perf_counter() - start

    latencies.sort()

    p50 = statistics.median(latencies)
    p95_index = math.ceil(
        0.95 * len(latencies)
    ) - 1
    p95 = latencies[p95_index]

    throughput = REQUESTS / duration

    print(f"Requests: {REQUESTS}")
    print(f"p50 latency: {p50:.3f} ms")
    print(f"p95 latency: {p95:.3f} ms")
    print(f"Throughput: {throughput:.2f} req/s")


for name, url in SERVERS.items():
    benchmark(name, url)
