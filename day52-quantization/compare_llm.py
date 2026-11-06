
import argparse
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["fp16", "nf4"], required=True)
args = parser.parse_args()

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MIB = 1024 ** 2

assert torch.cuda.is_available(), "CUDA GPU required"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

config = None
if args.mode == "nf4":
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

before = torch.cuda.memory_allocated()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    quantization_config=config,
)
model.eval()

torch.cuda.synchronize()
after = torch.cuda.memory_allocated()

print(f"\nPrecision: {args.mode}")
print(f"Model footprint: {model.get_memory_footprint()/MIB:.2f} MiB")
print(f"GPU allocated increase: {(after-before)/MIB:.2f} MiB")

messages = [
    {
        "role": "user",
        "content": "Explain why quantization reduces GPU memory usage."
    }
]
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

# Handle either BatchEncoding or tensor output
input_ids = (
    encoded["input_ids"]
    if hasattr(encoded, "keys")
    else encoded
)

inputs = input_ids.to("cuda")

# Warm up the model
with torch.inference_mode():
    model.generate(
        inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

times = []
generated = None

for _ in range(3):
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)

new_tokens = generated.shape[-1] - inputs.shape[-1]
avg_time = sum(times) / len(times)

print(f"Average generation time: {avg_time:.3f} s")
print(f"Generated tokens: {new_tokens}")
print(f"Approx. throughput: {new_tokens/avg_time:.2f} tokens/s")
print(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated()/MIB:.2f} MiB")

print("\nModel response:")
print(tokenizer.decode(generated[0][inputs.shape[-1]:], skip_special_tokens=True))
