import torch
import time

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

device = torch.device("cuda")

def gb(x):
    return x / 1024**3

def show_memory(label):
    print(f"\n--- {label} ---")
    print(
        f"allocated: {gb(torch.cuda.memory_allocated()):.3f} GB"
    )
    print(
        f"reserved : {gb(torch.cuda.memory_reserved()):.3f} GB"
    )
    print(
        f"peak     : {gb(torch.cuda.max_memory_allocated()):.3f} GB"
    )

torch.cuda.reset_peak_memory_stats()

show_memory("initial")

x = torch.randn(
    10000,
    10000,
    device=device,
    dtype=torch.float32
)

show_memory("after x")

y = torch.randn(
    10000,
    10000,
    device=device,
    dtype=torch.float32
)

show_memory("after y")

del y

show_memory("after deleting y")

torch.cuda.empty_cache()

show_memory("after empty_cache")

time.sleep(10)
