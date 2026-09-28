cat > gpu_workload.py <<'PY'
import torch
import time

device = "cuda"

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU not available")

print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nAllocating matrices...")
a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

print("Starting GPU workload...")

for _ in range(100):
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

print("GPU computation complete.")
print("Holding allocations for 10 seconds...")
time.sleep(10)

print("Finished.")
PY
