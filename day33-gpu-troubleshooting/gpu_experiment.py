import sys
import time
import torch

mode = sys.argv[1] if len(sys.argv) > 1 else "paced"

if mode not in ("paced", "continuous"):
    raise ValueError("Mode must be paced or continuous")

device = "cuda"
N = 2048
duration = 20

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Experiment: {mode}", flush=True)

a = torch.randn(N, N, device=device)
b = torch.randn(N, N, device=device)
c = torch.empty(N, N, device=device)

# Warm up CUDA and matrix multiplication
for _ in range(10):
    torch.mm(a, b, out=c)

torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

operations = 0
total_cuda_ms = 0.0

start = time.perf_counter()

while time.perf_counter() - start < duration:
    start_event.record()

    torch.mm(a, b, out=c)

    end_event.record()
    end_event.synchronize()

    total_cuda_ms += start_event.elapsed_time(end_event)
    operations += 1

    if mode == "paced":
        time.sleep(0.25)

elapsed = time.perf_counter() - start

print(f"Operations: {operations}")
print(f"Wall time: {elapsed:.2f} s")
print(f"Operations/sec: {operations / elapsed:.2f}")
print(f"Average CUDA event time: {total_cuda_ms / operations:.3f} ms")
print(f"CUDA event time total: {total_cuda_ms / 1000:.2f} s")
print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB")
