import time
import math
import csv
import statistics
import torch
from torch import nn

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Model and input initialization
setup_start = time.perf_counter()

model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024)
).to(device).eval()

x = torch.randn(32, 1024, device=device)

if device.type == "cuda":
    torch.cuda.synchronize()

setup_ms = (time.perf_counter() - setup_start) * 1000


def measure():
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000


# First inference
first_ms = measure()

# Warmup phase
warmup = [measure() for _ in range(5)]

# Steady-state measurements
samples = [measure() for _ in range(100)]
ordered = sorted(samples)

def percentile(p):
    index = math.ceil(p / 100 * len(ordered)) - 1
    return ordered[index]

print(f"Device: {device}")
print(f"Model/input setup: {setup_ms:.3f} ms")
print(f"First inference: {first_ms:.3f} ms")
print(f"Warm p50: {statistics.median(samples):.3f} ms")
print(f"Warm p95: {percentile(95):.3f} ms")
print(f"Warm p99: {percentile(99):.3f} ms")

with open("latencies.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["phase", "latency_ms"])
    writer.writerow(["first", first_ms])
    writer.writerows(("warmup", v) for v in warmup)
    writer.writerows(("steady", v) for v in samples)