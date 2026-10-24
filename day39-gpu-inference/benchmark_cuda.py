
import statistics
import time
import torch

from model import TinyRegressionModel

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")

N = 300
WARMUP = 100

cpu_model = TinyRegressionModel().eval()
gpu_model = TinyRegressionModel().to("cuda:0").eval()

x_cpu = torch.tensor(
    [[1.0, 2.0, 3.0]],
    dtype=torch.float32,
)
x_gpu = x_cpu.to("cuda:0")

cpu_times = []
gpu_end_to_end = []
gpu_device_times = []

with torch.inference_mode():

    # Warm up both models
    for _ in range(WARMUP):
        cpu_model(x_cpu)
        gpu_model(x_gpu)

    torch.cuda.synchronize()

    for _ in range(N):

        # CPU execution
        t0 = time.perf_counter()
        cpu_model(x_cpu).item()
        cpu_times.append(
            (time.perf_counter() - t0) * 1000
        )

        # GPU including transfers and synchronization
        t0 = time.perf_counter()
        gpu_model(x_cpu.to("cuda:0")).item()
        gpu_end_to_end.append(
            (time.perf_counter() - t0) * 1000
        )

        # GPU execution region only
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        gpu_model(x_gpu)
        end.record()

        end.synchronize()

        gpu_device_times.append(
            start.elapsed_time(end)
        )

print(f"GPU: {torch.cuda.get_device_name(0)}")

for name, times in [
    ("CPU model + scalar", cpu_times),
    ("GPU model + transfers", gpu_end_to_end),
    ("GPU device execution region", gpu_device_times),
]:
    print(
        f"{name}: "
        f"median={statistics.median(times):.4f} ms"
    )
