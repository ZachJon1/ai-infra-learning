# Day 52 — LLM Quantization: Memory, Numerical Error, and Inference Performance

**AI Infrastructure & ML Systems — 220-Day Roadmap**  
**Phase 3: Model Serving / Inference Infrastructure**  
**Topic:** Quantization concepts and practical tradeoffs

## Overview

This lab explores **quantization** as a technique for reducing the memory requirements of machine-learning models. It begins with a controlled PyTorch matrix experiment, then compares **FP16** inference against **NF4 (4-bit)** inference using a real language model.

The main result is an instructive tradeoff: **NF4 reduced the measured model memory footprint by 54.3%, but decreased single-request generation throughput by 30.1%** on the tested configuration. Lower precision does not automatically mean lower latency.

> **Scope:** These are measurements from one local GPU and a small model. They are not universal FP16-versus-4-bit performance claims.

## Learning objectives

- Explain the differences among FP32, FP16, BF16, FP8, INT8, and INT4.
- Distinguish a numerical **format** (such as INT4 or NF4) from a quantization **method** (such as GPTQ or AWQ).
- Quantify the memory savings and numerical errors associated with reduced precision.
- Compare a real model's FP16 and 4-bit memory footprints, latency, and token throughput.
- Identify how GPU kernel support, dequantization overhead, KV-cache memory, and model quality influence deployment decisions.

## Files

```text
 day52-quantization/
 ├── quantization_lab.py    # Simulated FP16 / INT8 / INT4 matrix experiment
 ├── compare_llm.py        # FP16 vs bitsandbytes NF4 model experiment
 └── README.md             # This document
```

## Environment

- **OS:** Linux
- **Python:** 3.11 (observed in the test environment)
- **Accelerator:** CUDA-enabled NVIDIA laptop GPU (RTX 3070 Laptop GPU, 8 GB VRAM, as identified during the study)
- **Libraries:** PyTorch, Transformers, Accelerate, bitsandbytes
- **LLM:** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Quantized loading:** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`

Exact library, CUDA, and driver versions were **not captured** for the recorded runs. To record them for a future reproducible benchmark:

```bash
python --version
python -m pip freeze > requirements-lock.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
nvidia-smi
```

## Setup and execution

Create an isolated environment (or reuse the existing `~/day52-quant-venv`):

```bash
cd ~/ai-infra/day52-quantization
python3 -m venv ~/day52-quant-venv
source ~/day52-quant-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers accelerate bitsandbytes
```

Run the numerical experiment:

```bash
python quantization_lab.py
```

Run each LLM variant in a **separate process** to avoid having the previous model resident in GPU memory:

```bash
python compare_llm.py --mode fp16
python compare_llm.py --mode nf4
```

The Hugging Face model may be downloaded on the first run; a token is not normally required to download this public model.

### Transformers compatibility note

In the installed Transformers version, `tokenizer.apply_chat_template(...)` returned a `BatchEncoding` object. Passing that object positionally to `model.generate(...)` triggered an `AttributeError` when generation attempted to access `.shape`. The solution was to extract the actual input IDs:

```python
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
inputs = input_ids.to("cuda")
```

This was an **input-format compatibility error**, not a GPU out-of-memory or quantization failure.

---

## Experiment 1 — Simulated numerical quantization

**Procedure:** Generate a random `2048 × 2048` FP16 weight matrix and a `32 × 2048` FP16 input tensor. Apply simple symmetric **per-tensor** INT8 and INT4 quantization, dequantize the weights to FP16, and compare matrix-multiplication outputs with the original FP16 baseline. For the storage calculation, the INT4 values are packed two per byte.

### Measured weight storage

| Representation | Storage | Reduction vs FP16 |
|---|---:|---:|
| FP16 | 8.00 MiB | Baseline |
| INT8 | 4.00 MiB | 50% |
| Packed INT4 | 2.00 MiB | 75% |

### Measured relative output error

| Representation | Relative output error | Percentage equivalent |
|---|---:|---:|
| INT8 | 0.012211 | 1.2211% |
| INT4 | 0.221487 | 22.1487% |

The metric was the relative Frobenius-norm error between the quantized and reference matrix outputs:

\[
\text{Relative error} = \frac{\lVert Y_{\mathrm{quantized}} - Y_{\mathrm{FP16}} \rVert_F}{\lVert Y_{\mathrm{FP16}} \rVert_F}
\]

**Interpretation:** INT4 achieved the largest storage reduction but introduced much greater numerical error in this simple experiment (approximately **18.1×** the INT8 relative error). This is a matrix-output error, **not** a measure of real-world LLM task accuracy. The calculation did not use an optimized INT4 inference kernel: the quantized weights were restored to FP16 for multiplication.

---

## Experiment 2 — Real LLM: FP16 vs NF4

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Prompt:** `Explain why quantization reduces GPU memory usage.`  
**Decoding:** Greedy (`do_sample=False`)  
**Warmup:** One generation of up to 32 new tokens  
**Timed runs:** Three generations, each limited to 64 new tokens  
**Timing:** Wall-clock time with CUDA synchronization before and after each run  
**Comparison:** FP16 versus bitsandbytes NF4 using FP16 compute

### Recorded results

| Metric | FP16 | NF4 (4-bit) | Observation |
|---|---:|---:|---|
| Model memory footprint | 942.29 MiB | 430.42 MiB | **54.3% lower** with NF4 |
| Increase in PyTorch GPU allocated memory after loading | 942.32 MiB | 451.86 MiB | Lower with NF4 |
| Peak PyTorch allocated memory after warmup/reset | 953.10 MiB | 469.48 MiB | Lower with NF4 |
| Average generation time | 1.162 s | 1.663 s | **43.1% longer** with NF4 |
| Generated tokens | 64 | 64 | Same number reported |
| Approximate generation throughput | 55.08 tokens/s | 38.48 tokens/s | **30.1% lower** with NF4 |

Calculations:

- **Memory footprint reduction:** `(942.29 - 430.42) / 942.29 × 100 ≈ 54.3%`
- **Throughput reduction:** `(55.08 - 38.48) / 55.08 × 100 ≈ 30.1%`
- **Increase in generation time:** `(1.663 - 1.162) / 1.162 × 100 ≈ 43.1%`

### Interpretation

NF4 substantially reduced model-resident memory, but it **did not speed up generation** in this test. Plausible explanations include:

1. **Dequantization/computation overhead** from operating on 4-bit weights.
2. **Kernel efficiency:** the FP16 path may be more efficient for the available GPU and software stack.
3. **Small model size:** reduced weight traffic may not outweigh overhead for a ~0.5B-parameter model at batch size one.

These are **hypotheses**, not profiled diagnoses. Confirming the bottleneck would require GPU profiling, more repetitions, different prompt/output lengths, and concurrency experiments.

### Output quality observations

Both variants answered the quantization prompt. The NF4 response included awkward or questionable statements about numeric ranges, while the FP16 response was more coherent in this single example. **One truncated, 64-token response is insufficient to conclude that NF4 meaningfully harms model quality.** A fair comparison needs a multi-prompt evaluation set with consistent decoding and task-specific metrics.

## Deployment considerations

Quantization decisions should account for more than model-weight storage:

- **Total VRAM:** weights, quantization metadata, KV cache, activations, temporary buffers, and runtime overhead.
- **GPU support:** compute capability and availability of optimized kernels for the chosen quantization format and implementation.
- **Latency and throughput:** measure with the actual model, prompts, sequence lengths, and concurrent request patterns.
- **Quality:** test representative tasks for accuracy, consistency, and unacceptable output degradation.
- **Concurrency and context length:** smaller weight footprints may free capacity for KV-cache growth or additional requests, but this must be verified.

For example, a **7B-parameter** model has a theoretical weight footprint of approximately **14 GB** at FP16 versus **3.5 GB** at 4 bits (decimal units, ignoring metadata). A 12 GB GPU might accommodate its quantized weights, but fitting the entire inference workload still depends on KV-cache capacity and other allocations.

## Limitations

- Results were collected on **one GPU** using a **single small model** and **one prompt**.
- No controlled power, clock-frequency, temperature, or competing-process measurements were recorded.
- The LLM benchmark used only **three timed generations per mode**, not statistically robust repetitions.
- Throughput is approximated from generated tokens divided by average generation time; it is **not** a multi-client requests-per-second benchmark.
- The measured peak memory is **PyTorch allocated memory after warmup**, not device-wide peak VRAM reported by `nvidia-smi`.
- INT4 matrix storage in Experiment 1 excludes scale metadata, and its numerical comparison is **not** optimized low-bit inference.
- NF4 is **not GPTQ or AWQ**; results for one quantization implementation do not automatically generalize to another.
- Model quality was inspected qualitatively, with no standardized evaluation dataset.

## Key takeaway

> **Quantization primarily provides a model-memory optimization; whether it also improves latency or throughput depends on hardware support, kernels, model scale, and workload.** In this measured local experiment, NF4 cut model footprint by **54.3%** but reduced single-request generation throughput by **30.1%**. Select the quantization strategy using **measured VRAM, latency, throughput, and output quality**, not bit width alone.

## Research papers — GPTQ and AWQ

These papers provide the theoretical and algorithmic background for the weight-quantization techniques discussed in this lesson. **Neither GPTQ nor AWQ was benchmarked in the experiments above**; the real-LLM comparison used bitsandbytes NF4.

1. **GPTQ — *Accurate Post-Training Quantization for Generative Pre-trained Transformers***  
   Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh (2022).  
   **Paper:** [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)  
   **Focus:** Weight-only, post-training quantization that uses approximate second-order information to compensate for quantization error.

2. **AWQ — *Activation-aware Weight Quantization for LLM Compression and Acceleration***  
   Ji Lin et al. (MLSys 2024).  
   **Paper:** [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) · [MLSys 2024 proceedings](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)  
   **Focus:** Uses activation statistics to identify salient weight channels and optimize scaling to limit quantization error.

## Next steps

Day 53 extends this work into **inference load testing**: compare concurrency levels and request rates, record p50/p95/p99 latency and throughput, study KV-cache pressure, and identify saturation and backpressure behavior. Future work could also compare GPTQ, AWQ, and other GPU-supported quantization methods using the same controlled workload.
