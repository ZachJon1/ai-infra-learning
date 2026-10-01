import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

def gb(x):
    return x / 1024**3

def show(label):
    free, total = torch.cuda.mem_get_info()

    print(f"\n--- {label} ---")
    print(f"free      : {gb(free):.2f} GB")
    print(
        f"allocated : "
        f"{gb(torch.cuda.memory_allocated()):.2f} GB"
    )
    print(
        f"reserved  : "
        f"{gb(torch.cuda.memory_reserved()):.2f} GB"
    )

free, total = torch.cuda.mem_get_info()

show("initial")

# Consume about 55% of currently free VRAM.
first_bytes = int(free * 0.55)
first_elements = first_bytes // 4

x = torch.empty(
    first_elements,
    dtype=torch.float32,
    device="cuda"
)

show("after first live tensor")

# Now request another 50% of the ORIGINAL free VRAM.
second_bytes = int(free * 0.50)
second_elements = second_bytes // 4

print(
    f"\nAttempting second allocation: "
    f"{gb(second_bytes):.2f} GB"
)

try:
    y = torch.empty(
        second_elements,
        dtype=torch.float32,
        device="cuda"
    )

except torch.OutOfMemoryError as e:
    print("\n===== OOM =====")
    print(e)
    show("after failed second allocation")
