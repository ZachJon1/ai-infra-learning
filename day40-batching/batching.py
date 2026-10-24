
import time
import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

# Example inference model
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(32, 10)
).to(device).eval()

# Eight inputs, each with shape [3, 224, 224]
inputs = torch.randn(8, 3, 224, 224, device=device)

def sequential_inference():
    return torch.cat(
        [model(inputs[i:i+1]) for i in range(8)],
        dim=0
    )

def batched_inference():
    return model(inputs)

def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()

def benchmark(fn, iterations):
    synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        fn()

    synchronize()
    return time.perf_counter() - start

iterations = 50 if device.type == "cuda" else 10

with torch.inference_mode():
    # Warm up both execution paths
    for _ in range(10):
        sequential_inference()
        batched_inference()

    # Verify output consistency
    individual = sequential_inference()
    batched = batched_inference()

    print("Output shape:", batched.shape)
    print(
        "Outputs close:",
        torch.allclose(individual, batched,
                       rtol=1e-4, atol=1e-5)
    )

    # Measure processing time
    seq_time = benchmark(sequential_inference, iterations)
    batch_time = benchmark(batched_inference, iterations)

count = 8 * iterations

print(f"Sequential throughput: {count / seq_time:.2f} inputs/s")
print(f"Batched throughput:    {count / batch_time:.2f} inputs/s")
print(f"Speedup:               {seq_time / batch_time:.2f}x")
print(f"Sequential total time: {seq_time:.4f} s")
print(f"Batched total time:    {batch_time:.4f} s")
