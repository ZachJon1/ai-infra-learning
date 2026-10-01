import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

device = torch.device("cuda")

def gb(x):
    return x / 1024**3

def show(i):
    free, total = torch.cuda.mem_get_info()

    print(
        f"iteration={i:02d} | "
        f"allocated={gb(torch.cuda.memory_allocated()):.2f} GB | "
        f"reserved={gb(torch.cuda.memory_reserved()):.2f} GB | "
        f"free={gb(free):.2f} GB"
    )

outputs = []

show(0)

for i in range(1, 30):
    try:
        # ~0.37 GiB FP32 tensor
        y = torch.randn(
            10000,
            10000,
            dtype=torch.float32,
            device=device
        )

        # Intentionally retain the GPU tensor
        outputs.append(y)

        show(i)

    except torch.OutOfMemoryError as e:
        print("\n===== OOM =====")
        print(e)

        print(
            f"\nNumber of retained tensors: "
            f"{len(outputs)}"
        )

        show(i)

        break
