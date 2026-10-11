import sys
import time
import torch

mode = sys.argv[1] if len(sys.argv) > 1 else "copy-each"

if mode not in ("copy-each", "resident"):
    raise ValueError("Choose copy-each or resident")

assert torch.cuda.is_available()

torch.manual_seed(33)

N = 2048
DURATION = 12

# Create data in CPU memory.
a_cpu = torch.randn(N, N)
b_cpu = torch.randn(N, N)

# Allocate GPU buffers.
a_gpu = a_cpu.to("cuda")
b_gpu = b_cpu.to("cuda")
out = torch.empty_like(a_gpu)

# Warm up GPU computation.
for _ in range(10):
    torch.mm(a_gpu, b_gpu, out=out)

torch.cuda.synchronize()

count = 0
total_kernel_ms = 0.0

start = time.perf_counter()

while time.perf_counter() - start < DURATION:

    if mode == "copy-each":
        # Deliberate inefficiency:
        # Recopy the SAME tensors on every iteration.
        a_gpu.copy_(a_cpu)
        b_gpu.copy_(b_cpu)

    kernel_start = torch.cuda.Event(enable_timing=True)
    kernel_end = torch.cuda.Event(enable_timing=True)

    kernel_start.record()
    torch.mm(a_gpu, b_gpu, out=out)
    kernel_end.record()

    kernel_end.synchronize()

    total_kernel_ms += kernel_start.elapsed_time(kernel_end)
    count += 1

elapsed = time.perf_counter() - start

print(f"\nMode: {mode}")
print(f"Operations: {count}")
print(f"Wall time: {elapsed:.2f} s")
print(f"Throughput: {count / elapsed:.2f} ops/sec")
print(f"Average kernel time: {total_kernel_ms / count:.3f} ms")
print(f"Kernel event time total: {total_kernel_ms / 1000:.2f} s")
print(
    f"Peak PyTorch VRAM allocation: "
    f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB"
)
