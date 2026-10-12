
import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--memory-mib", type=int, default=512)
parser.add_argument("--duration", type=int, default=40)
args = parser.parse_args()

device = torch.device("cuda:0")

# Allocate and initialize approximately 512 MiB of tensor data.
elements = args.memory_mib * 1024 * 1024 // 4
reserved = torch.ones(
    elements, dtype=torch.float32, device=device
)

# Repeated GPU computation.
a = torch.randn(1536, 1536, device=device)
b = torch.randn(1536, 1536, device=device)
c = torch.empty_like(a)

torch.cuda.synchronize()

allocated = torch.cuda.memory_allocated() / (1024**2)

print(f"{args.name}: started")
print(f"{args.name}: tensor memory allocated = {allocated:.1f} MiB")

start = time.perf_counter()
iterations = 0

while time.perf_counter() - start < args.duration:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    iterations += 1

elapsed = time.perf_counter() - start

print(f"{args.name}: elapsed = {elapsed:.2f} s")
print(f"{args.name}: iterations = {iterations}")
print(f"{args.name}: iterations/s = {iterations/elapsed:.2f}")
