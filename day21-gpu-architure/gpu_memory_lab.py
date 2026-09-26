import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU not available")

print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("PyTorch CUDA build:", torch.version.cuda)

properties = torch.cuda.get_device_properties(0)
print("CUDA-visible VRAM GiB:", properties.total_memory / 1024**3)

# FP32 tensor
x = torch.zeros((5000, 5000), device="cuda")

print("\nFP32")
print("Elements:", x.numel())
print("Bytes/element:", x.element_size())
print("Tensor MiB:", x.numel() * x.element_size() / 1024**2)

# FP16 tensor
y = torch.zeros(
    (5000, 5000),
    dtype=torch.float16,
    device="cuda",
)

print("\nFP16")
print("Elements:", y.numel())
print("Bytes/element:", y.element_size())
print("Tensor MiB:", y.numel() * y.element_size() / 1024**2)

print("\nPyTorch allocator")
print("Allocated MiB:", torch.cuda.memory_allocated() / 1024**2)
print("Reserved MiB:", torch.cuda.memory_reserved() / 1024**2)

del x
del y

print("\nAfter deleting tensors")
print("Allocated MiB:", torch.cuda.memory_allocated() / 1024**2)
print("Reserved MiB:", torch.cuda.memory_reserved() / 1024**2)

torch.cuda.empty_cache()

print("\nAfter empty_cache()")
print("Allocated MiB:", torch.cuda.memory_allocated() / 1024**2)
print("Reserved MiB:", torch.cuda.memory_reserved() / 1024**2)
