
import torch
import torch.nn as nn

# Require GPU execution for this experiment
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))

# A small example inference model
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

model = model.to(device)
model.eval()

# Simulate one incoming request
inputs = torch.randn(1, 128)

print("Original input device:", inputs.device)

# Transfer request data to GPU
inputs = inputs.to(device)

with torch.inference_mode():
    outputs = model(inputs)

print("Model device:", next(model.parameters()).device)
print("Input device:", inputs.device)
print("Output device:", outputs.device)
print("Output shape:", outputs.shape)

print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated(0) / 1024**2, 2),
    "MiB"
)
