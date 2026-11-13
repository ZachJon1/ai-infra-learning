"""Day 59 controlled CUDA matrix-multiplication workload.

Reconstructed from the guided lab methodology; the user's exact local
script was not exported into this repository. Requires CUDA-enabled PyTorch.
"""
import argparse
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--size", type=int, default=2048)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Running GPU workload for {args.seconds:.0f} seconds", flush=True)
    torch.set_grad_enabled(False)
    a = torch.randn(args.size, args.size, device=device)
    b = torch.randn(args.size, args.size, device=device)
    start = time.monotonic()
    operations = 0
    try:
        while time.monotonic() - start < args.seconds:
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            operations += 1
    except KeyboardInterrupt:
        print("Workload interrupted by user.")
    elapsed = time.monotonic() - start
    print(f"Completed: {operations} matrix multiplications")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Operations/second: {operations / elapsed:.2f}")


if __name__ == "__main__":
    main()
