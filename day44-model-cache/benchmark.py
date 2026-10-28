import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 8)
    )

def sync():
    if device == "cuda":
        torch.cuda.synchronize()

print(f"Running on: {device}")

for run in range(1, 4):
    # Stage 1: Read checkpoint
    start = time.perf_counter()
    weights = torch.load(
        "model_weights.pt",
        map_location="cpu",
        weights_only=True
    )
    disk_time = time.perf_counter() - start

    # Stage 2: Reconstruct model in RAM
    start = time.perf_counter()
    model = create_model()
    model.load_state_dict(weights)
    model.eval()
    ram_time = time.perf_counter() - start

    # Stage 3: Transfer to device
    sync()
    start = time.perf_counter()
    model = model.to(device)
    sync()
    device_time = time.perf_counter() - start

    # Stage 4: First inference
    x = torch.ones(1, 2048, device=device)

    sync()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model(x)

    sync()
    inference_time = time.perf_counter() - start

    print(f"\nRun {run}")
    print(f"Checkpoint read: {disk_time*1000:.2f} ms")
    print(f"RAM reconstruction: {ram_time*1000:.2f} ms")
    print(f"Device transfer: {device_time*1000:.2f} ms")
    print(f"First inference: {inference_time*1000:.2f} ms")

    del model, weights, x, output
