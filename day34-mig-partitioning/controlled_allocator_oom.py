
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = 0

# Artificially limit this process's PyTorch CUDA
# caching allocator to 10% of total GPU memory.
# This is NOT a MIG partition or hardware quota.
torch.cuda.set_per_process_memory_fraction(0.10, device)

total = torch.cuda.get_device_properties(device).total_memory

print(f"GPU: {torch.cuda.get_device_name(device)}")
print(f"Physical GPU memory: {total / 2**20:.0f} MiB")
print(f"Artificial allocator cap: {total * 0.10 / 2**20:.0f} MiB")

# First allocation: expected to succeed.
first = torch.empty(
    256 * 1024 * 1024,
    dtype=torch.uint8,
    device="cuda"
)

print("\nFirst allocation: SUCCESS")
print(f"Tensor memory allocated: {torch.cuda.memory_allocated()/2**20:.0f} MiB")

# Second allocation: total requested tensor memory
# would exceed our artificial allocator cap.
try:
    second = torch.empty(
        768 * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda"
    )
    print("Second allocation: SUCCESS (unexpected)")

except torch.OutOfMemoryError as exc:
    print("\nSecond allocation: EXPECTED CUDA OOM")
    print(str(exc).splitlines()[0])

print("\nProgram remained alive after handling the exception.")
