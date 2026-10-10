
import torch

torch.manual_seed(32)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Input: 2 samples, 4 input features
X = torch.randn(2, 4, device=device)

# Weights: 4 input features, 8 output features
W = torch.randn(4, 8, device=device)

# Full, unpartitioned computation
Y_full = X @ W

print("\nFull output shape:", tuple(Y_full.shape))

# ---------------------------------------
# 1. COLUMN PARALLELISM
# ---------------------------------------

# Divide output features into two partitions
W0, W1 = torch.chunk(W, 2, dim=1)

# Independently compute output partitions
Y0 = X @ W0
Y1 = X @ W1

# Reconstruct complete output
Y_column = torch.cat([Y0, Y1], dim=1)

print("\nCOLUMN PARALLELISM")
print("Weight partition shapes:", W0.shape, W1.shape)
print("Partial output shapes:", Y0.shape, Y1.shape)
print("Matches full output:",
      torch.allclose(Y_column, Y_full, atol=1e-5))

# ---------------------------------------
# 2. ROW PARALLELISM
# ---------------------------------------

# Divide the input and the weight matrix
X0, X1 = torch.chunk(X, 2, dim=1)
R0, R1 = torch.chunk(W, 2, dim=0)

# Compute partial contributions
P0 = X0 @ R0
P1 = X1 @ R1

# Reconstruct the complete output
Y_row = P0 + P1

print("\nROW PARALLELISM")
print("Weight partition shapes:", R0.shape, R1.shape)
print("Partial output shapes:", P0.shape, P1.shape)
print("Matches full output:",
      torch.allclose(Y_row, Y_full, atol=1e-5))

# Verify both implementations
assert torch.allclose(Y_column, Y_full, atol=1e-5)
assert torch.allclose(Y_row, Y_full, atol=1e-5)

print("\nAll tensor parallelism checks passed!")


# Two consecutive tensor-parallel linear layers

# X shape: (2, 4)
# W shape: (4, 8) from previous experiment

W_next = torch.randn(8, 4, device=device)

# Reference result using full matrices
full_result = (X @ W) @ W_next

# First layer: column parallel
first_shards = torch.chunk(W, 2, dim=1)
hidden_parts = [X @ shard for shard in first_shards]

# Second layer: row parallel
second_shards = torch.chunk(W_next, 2, dim=0)

# Each GPU can compute using its local hidden partition
partial_results = [
    hidden_parts[i] @ second_shards[i]
    for i in range(2)
]

# Sum contributions at the end
parallel_result = sum(partial_results)

print("Full result shape:", full_result.shape)
print("Parallel result shape:", parallel_result.shape)
print("Results match:",
      torch.allclose(
          full_result, parallel_result,
          atol=1e-5, rtol=1e-5
      ))
