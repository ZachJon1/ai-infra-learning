import torch
from pathlib import Path
from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
)

# Select execution device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Matrix size
N = 1024 if device == "cuda" else 512

a = torch.randn(N, N, device=device)
b = torch.randn(N, N, device=device)

def workload():
    return torch.relu(a @ b)

# Warm up before profiling
for _ in range(5):
    output = workload()

if device == "cuda":
    torch.cuda.synchronize()

# Collect profiler activities
activities = [ProfilerActivity.CPU]

if device == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
) as prof:

    with record_function("inference_workload"):
        for _ in range(10):
            output = workload()

        if device == "cuda":
            torch.cuda.synchronize()

# Display profiling results
sort_key = (
    "self_cuda_time_total"
    if device == "cuda"
    else "self_cpu_time_total"
)

print(
    prof.key_averages().table(
        sort_by=sort_key,
        row_limit=12,
    )
)

# Export trace for visualization
trace_path = Path("day25_trace.json")
prof.export_chrome_trace(str(trace_path))

print(f"Trace saved to {trace_path.resolve()}")
