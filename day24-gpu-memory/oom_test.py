import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

device = torch.device("cuda")

def gb(x):
    return x / 1024**3

free, total = torch.cuda.mem_get_info()

print(f"Total GPU memory : {gb(total):.2f} GB")
print(f"Free GPU memory  : {gb(free):.2f} GB")
print(
    f"Allocated        : "
    f"{gb(torch.cuda.memory_allocated()):.2f} GB"
)
print(
    f"Reserved         : "
    f"{gb(torch.cuda.memory_reserved()):.2f} GB"
)

# Request about 110% of currently free GPU memory.
requested_bytes = int(free * 1.10)

# FP32 = 4 bytes per element
num_elements = requested_bytes // 4

print(
    f"\nAttempting allocation of approximately "
    f"{gb(requested_bytes):.2f} GB..."
)

try:
    x = torch.empty(
        num_elements,
        dtype=torch.float32,
        device=device
    )

except torch.OutOfMemoryError as e:
    print("\n===== CUDA OOM DETECTED =====")
    print(e)

    print("\n===== MEMORY AFTER FAILURE =====")
    print(
        f"allocated: "
        f"{gb(torch.cuda.memory_allocated()):.2f} GB"
    )
    print(
        f"reserved : "
        f"{gb(torch.cuda.memory_reserved()):.2f} GB"
    )
