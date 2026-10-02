import time
import statistics
import torch

assert torch.cuda.is_available(), "CUDA GPU required"

device = "cuda"
N = 1024
STEPS = 30
TRIALS = 7

print("GPU:", torch.cuda.get_device_name(0))

a = torch.randn(N, N, device=device)
b = torch.randn(N, N, device=device)


@torch.inference_mode()
def benchmark(force_sync=False):

    # Clear outstanding GPU work
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_event.record()

    for _ in range(STEPS):
        output = torch.relu(a @ b)

        # Deliberately inefficient behavior
        if force_sync:
            torch.cuda.synchronize()

    end_event.record()
    end_event.synchronize()

    wall_ms = (time.perf_counter() - wall_start) * 1000
    gpu_ms = start_event.elapsed_time(end_event)

    return wall_ms, gpu_ms, output


# Warm up the GPU
with torch.inference_mode():
    for _ in range(10):
        _ = torch.relu(a @ b)

torch.cuda.synchronize()

results = {
    "forced_sync": {"wall": [], "event": []},
    "no_sync": {"wall": [], "event": []},
}

# Alternate execution order to reduce order bias
for trial in range(TRIALS):
    cases = ["forced_sync", "no_sync"]

    if trial % 2:
        cases.reverse()

    for case in cases:
        wall, gpu, output = benchmark(
            force_sync=(case == "forced_sync")
        )

        results[case]["wall"].append(wall)
        results[case]["event"].append(gpu)


for case, measurements in results.items():
    wall_median = statistics.median(measurements["wall"])
    event_median = statistics.median(measurements["event"])

    print(f"\n{case}")
    print(f"Median wall time: {wall_median:.3f} ms")
    print(f"Median CUDA event time: {event_median:.3f} ms")
    print(f"Mean wall time: {statistics.mean(measurements['wall']):.3f} ms")


forced = statistics.median(results["forced_sync"]["wall"])
optimized = statistics.median(results["no_sync"]["wall"])

print(f"\nWall-time ratio (forced/no_sync): {forced / optimized:.2f}x")
