
import json
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

BASE = "http://localhost:8000"
MODEL = "toy_classifier"
VERSION = "1"

CONCURRENCY = 16
ROUNDS = 100

INFER_URL = f"{BASE}/v2/models/{MODEL}/versions/{VERSION}/infer"
STATS_URL = f"{BASE}/v2/models/{MODEL}/versions/{VERSION}/stats"

payload = {
    "inputs": [{
        "name": "INPUT",
        "shape": [1, 4],
        "datatype": "FP32",
        "data": [1.0, 2.0, 3.0, 4.0]
    }],
    "outputs": [{"name": "OUTPUT"}]
}


def get_stats():
    with urlopen(STATS_URL, timeout=10) as response:
        return json.load(response)["model_stats"][0]


def send_request(barrier):
    barrier.wait(timeout=15)

    request = Request(
        INFER_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    with urlopen(request, timeout=15) as response:
        result = json.load(response)

    latency_ms = (time.perf_counter() - start) * 1000

    assert result["outputs"][0]["data"] == [4.0, 6.0]
    return latency_ms


before = get_stats()
latencies = []

start = time.perf_counter()

with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    for _ in range(ROUNDS):
        barrier = threading.Barrier(CONCURRENCY)
        futures = [
            executor.submit(send_request, barrier)
            for _ in range(CONCURRENCY)
        ]
        latencies.extend(f.result() for f in futures)

elapsed = time.perf_counter() - start
after = get_stats()

inferences = after["inference_count"] - before["inference_count"]
executions = after["execution_count"] - before["execution_count"]

old_batches = {
    item["batch_size"]: item["compute_infer"]["count"]
    for item in before["batch_stats"]
}

print("\n=== Dynamic Batching Results ===")
print(f"Successful requests: {len(latencies)}")
print(f"Inference count:     {inferences}")
print(f"Execution count:     {executions}")
print(f"Average batch size:  {inferences / executions:.2f}")
print(f"Throughput:          {len(latencies) / elapsed:.2f} req/s")
print(f"Mean client latency: {statistics.mean(latencies):.2f} ms")

ordered = sorted(latencies)
p95_index = math.ceil(0.95 * len(ordered)) - 1
print(f"p95 client latency:  {ordered[p95_index]:.2f} ms")

queue_before = before["inference_stats"]["queue"]
queue_after = after["inference_stats"]["queue"]

queue_count = queue_after["count"] - queue_before["count"]
queue_ns = queue_after["ns"] - queue_before["ns"]

if queue_count:
    print(f"Mean Triton queue:   {queue_ns / queue_count / 1e6:.2f} ms")

print("\nObserved batch sizes:")
for item in after["batch_stats"]:
    size = item["batch_size"]
    count = item["compute_infer"]["count"] - old_batches.get(size, 0)
    if count > 0:
        print(f"  Batch size {size}: {count} executions")
