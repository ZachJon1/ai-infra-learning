
import time
import torch
import torch.nn as nn

assert torch.cuda.is_available(), "CUDA GPU required"

device = "cuda"

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(32, 10)
).to(device).eval()

batch_sizes = [1, 4, 8, 16]
iterations = 100

print(
    f"{'Batch':<8} {'Latency(ms)':<15} "
    f"{'Throughput/s':<17} {'Peak VRAM(MiB)':<15}"
)

with torch.inference_mode():
    for bs in batch_sizes:
        inputs = torch.randn(
            bs, 3, 224, 224, device=device
        )

        # Warmup
        for _ in range(20):
            outputs = model(inputs)

        torch.cuda.synchronize()
        del outputs

        torch.cuda.reset_peak_memory_stats()

        timings = []

        for _ in range(iterations):
            start = time.perf_counter()
            outputs = model(inputs)
            torch.cuda.synchronize()

            timings.append(time.perf_counter() - start)

        avg_latency_ms = (
            sum(timings) / iterations * 1000
        )

        throughput = (
            bs * iterations / sum(timings)
        )

        peak_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        print(
            f"{bs:<8} {avg_latency_ms:<15.3f} "
            f"{throughput:<17.2f} {peak_vram:<15.2f}"
        )

        del inputs, outputs
