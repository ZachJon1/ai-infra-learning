import time
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is unavailable")

device = torch.device("cuda")
duration = 90

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Running GPU workload for {duration} seconds")

torch.set_grad_enabled(False)

# Modest allocations for an 8 GB GPU
a = torch.randn(2048, 2048, device=device)
b = torch.randn(2048, 2048, device=device)

start = time.monotonic()
operations = 0

while time.monotonic() - start < duration:
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    operations += 1

elapsed = time.monotonic() - start

print(f"Completed: {operations} matrix multiplications")
print(f"Elapsed: {elapsed:.2f} seconds")
print(f"Operations/second: {operations / elapsed:.2f}")
