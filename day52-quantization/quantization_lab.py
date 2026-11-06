
import torch

torch.manual_seed(52)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Simulate one neural-network weight matrix
W = torch.randn(2048, 2048, device=device, dtype=torch.float16)
x = torch.randn(32, 2048, device=device, dtype=torch.float16)

reference = x @ W


def quantize(w, bits):
    qmax = 2 ** (bits - 1) - 1

    # Symmetric, per-tensor quantization
    scale = w.float().abs().max().item() / qmax
    scale = max(scale, 1e-8)

    q = torch.round(w.float() / scale)
    q = q.clamp(-qmax, qmax).to(torch.int8)

    # Dequantize for ordinary FP16 matrix multiplication
    restored = (q.float() * scale).to(torch.float16)

    return q, restored


q8, w8 = quantize(W, 8)
q4, w4 = quantize(W, 4)

# Pack two signed 4-bit values into one byte
nibbles = (q4.flatten().to(torch.int16) & 0x0F).to(torch.uint8)
packed4 = nibbles[0::2] | (nibbles[1::2] << 4)


def relative_error(prediction, target):
    return (
        (prediction.float() - target.float()).norm()
        / target.float().norm()
    ).item()


output8 = x @ w8
output4 = x @ w4

print("\nWeight storage (excluding scaling metadata)")
print(f"FP16: {W.numel() * W.element_size() / 2**20:.2f} MiB")
print(f"INT8: {q8.numel() * q8.element_size() / 2**20:.2f} MiB")
print(f"INT4: {packed4.numel() / 2**20:.2f} MiB")

print("\nRelative output errors")
print(f"INT8: {relative_error(output8, reference):.6f}")
print(f"INT4: {relative_error(output4, reference):.6f}")
