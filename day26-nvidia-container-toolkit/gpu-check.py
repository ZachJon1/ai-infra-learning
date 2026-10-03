"""Check CUDA visibility and optionally execute a small GPU workload."""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("visible", "hidden"), required=True)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()

    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    print("CUDA available:", available)
    print("GPUs:", count)

    expected = args.expect == "visible"
    if available != expected:
        raise RuntimeError(f"Expected CUDA {args.expect}; availability was {available}")
    if args.count is not None and count != args.count:
        raise RuntimeError(f"Expected {args.count} GPUs; detected {count}")
    if not expected:
        print("GPU-hiding check passed")
        return

    print("GPU:", torch.cuda.get_device_name(0))
    a = torch.randn(1024, 1024, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    print("Result device:", b.device)
    print("GPU computation successful!")


if __name__ == "__main__":
    main()
