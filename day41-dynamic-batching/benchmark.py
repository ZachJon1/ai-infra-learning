
import math
import time

import torch
from torch import nn

from dynamic_batcher import DynamicBatcher


def percentile(values, p):
    values = sorted(values)
    index = max(0, math.ceil(p * len(values)) - 1)
    return values[index]


def benchmark(
    model,
    name,
    max_batch,
    max_wait,
    inputs,
    interval_ms=0,
):
    batcher = DynamicBatcher(
        model,
        max_batch_size=max_batch,
        max_wait_ms=max_wait,
        verbose=False,
    )

    futures = []
    start = time.perf_counter()

    for i, tensor in enumerate(inputs):
        if interval_ms > 0:
            target = start + i * interval_ms / 1000
            remaining = target - time.perf_counter()

            if remaining > 0:
                time.sleep(remaining)

        futures.append(batcher.submit(tensor))

    for future in futures:
        future.result(timeout=60)

    elapsed = time.perf_counter() - start

    latencies = batcher.request_latency_ms
    queue_times = batcher.queue_wait_ms
    sizes = batcher.batch_sizes

    print(f"\n{name}")
    print(f"Requests:           {len(inputs)}")
    print(f"Batches:            {len(sizes)}")
    print(f"Average batch size: {sum(sizes)/len(sizes):.2f}")
    print(f"Throughput:         {len(inputs)/elapsed:.1f} req/s")
    print(f"p50 latency:        {percentile(latencies, 0.50):.3f} ms")
    print(f"p95 latency:        {percentile(latencies, 0.95):.3f} ms")
    print(f"p95 queue wait:     {percentile(queue_times, 0.95):.3f} ms")


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).to(device).eval()

print(f"Device: {device}")

# GPU/model warmup
with torch.inference_mode():
    for batch_size in [1, 8]:
        sample = torch.randn(
            batch_size, 128, device=device
        )

        for _ in range(30):
            model(sample)

if device.type == "cuda":
    torch.cuda.synchronize(device)


configs = [
    ("A: No batching", 1, 0),
    ("B: Dynamic 5ms", 8, 5),
    ("C: Dynamic 20ms", 8, 20),
]


# Experiment 1: Burst traffic
print("\n===== BURST TRAFFIC =====")

burst_inputs = [
    torch.randn(128) for _ in range(512)
]

for name, size, delay in configs:
    benchmark(
        model,
        name,
        size,
        delay,
        burst_inputs,
    )


# Experiment 2: Paced traffic
print("\n===== PACED TRAFFIC: 500 req/s =====")

paced_inputs = [
    torch.randn(128) for _ in range(128)
]

for name, size, delay in configs:
    benchmark(
        model,
        name,
        size,
        delay,
        paced_inputs,
        interval_ms=2,
    )
