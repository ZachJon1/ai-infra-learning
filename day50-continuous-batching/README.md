# Day 50 — Continuous Batching in vLLM

**Track:** AI Infrastructure & ML Systems — 220-Day Roadmap  
**Phase:** 3 — Model Serving / Inference Infrastructure  
**Lab date:** November 3, 2026  
**Model:** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)  
**Status:** Two benchmark comparisons completed; root-cause investigation pending

## Overview

This lab explores **continuous batching**, the technique used by LLM inference engines such as vLLM to revise the set of running requests at generation-iteration boundaries. Rather than requiring every request in a fixed batch to finish, a scheduler can remove completed sequences and admit waiting requests when capacity permits.

We compared two values of vLLM's per-iteration token budget (`--max-num-batched-tokens=256` and `1024`) on the same small language model. Both a short-prompt and a long-prompt workload were tested. **The larger token budget did not improve performance in these runs.** The effect was particularly pronounced in the long-prompt workload, where the 1024-token configuration delivered **30.3% less output throughput** and much higher tail latency than the 256-token configuration.

> **Scope:** These are *observations from single benchmark runs*, not proof that a 256-token budget is universally faster. The tests did not yet collect a complete server-side scheduling trace or repeated-run statistics.

## Learning objectives

- Distinguish **static**, **dynamic**, and **continuous** batching.
- Explain the **prefill** and **decode** phases of autoregressive inference.
- Understand how **chunked prefill** and iteration-level scheduling share a token budget.
- Interpret `max_num_seqs`, `max_num_batched_tokens`, and KV-cache capacity constraints.
- Compare **TTFT**, **ITL**, **TPOT**, throughput, and tail latency.
- Use **service-level objectives (SLOs)** rather than raw GPU utilization alone to guide configuration decisions.

## Key concepts

| Concept | Meaning |
| --- | --- |
| Static batching | Processes a fixed group of requests together; newly arrived requests generally wait for the batch to finish. |
| Dynamic batching | Forms batches from requests gathered over a time window or according to other batching constraints. |
| Continuous batching | Updates the active request set at successive generation/scheduling iterations. |
| Prefill | Processes prompt tokens and builds KV-cache state; often compute-intensive. |
| Decode | Generates output autoregressively using the existing KV cache; often memory-bandwidth-sensitive, depending on batch size and hardware. |
| Chunked prefill | Divides prompt processing across scheduling iterations instead of requiring one uninterrupted prefill. |
| TTFT | Time to first token: request arrival to first output token. |
| ITL | Inter-token latency: delay between consecutive output tokens. |
| TPOT | Time per output token after the first; often computed as an average per request. |
| KV cache | Stored attention key/value tensors for prior tokens; grows with active sequence context and constrains concurrency. |

**Important distinction:** A large prefill may increase **ITL** for requests already streaming tokens; waiting or delayed prefill can increase **TTFT** for new requests.

## Environment and configurations

The experiment was run locally with the vLLM CLI and OpenAI-compatible serving API. The following settings are documented from the lab commands and reported benchmark output:

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| API endpoint | `http://127.0.0.1:8000` |
| `max_model_len` | `2048` |
| `max_num_seqs` | `4` |
| `gpu_memory_utilization` | `0.80` |
| Compared `max_num_batched_tokens` | `256`, `1024` |
| Benchmark backend | `openai` |
| Client `max_concurrency` | `8` |
| Request rate | `inf` (offered load as fast as concurrency permits) |
| Output stopping | `--ignore-eos` |

**Not captured in these results:** GPU model, GPU memory size, driver/CUDA versions, vLLM/PyTorch versions, GPU temperature/power traces, vLLM KV-cache usage, and preemption counts. 

### Environment check

```bash
python --version
vllm --version
nvidia-smi
```

If vLLM is unavailable in the existing Python environment, use an isolated environment compatible with the host GPU, drivers, Python, and vLLM release. For example, after checking compatibility:

```bash
cd ~/ai-infra/day50-continuous-batching
uv venv --python 3.12 --seed --managed-python .venv-vllm
source .venv-vllm/bin/activate
uv pip install vllm --torch-backend=auto
vllm --version
```

> vLLM/CUDA installation details can change. Consult the [official vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/) and [serving CLI reference](https://docs.vllm.ai/en/latest/cli/serve/) if your system differs.

## Reproducing the benchmarks

Run **one server configuration at a time**, stopping it before starting the other configuration. Keep the model, hardware, and all other settings unchanged.

### 1. Start the server — 256-token budget

**Terminal 1**:

```bash
cd ~/ai-infra/day50-continuous-batching
source .venv-vllm/bin/activate

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.80
```

Check the server is ready:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

### 2. Experiment A — Short prompts

**Terminal 2** (while the server is running):

```bash
mkdir -p results

vllm bench serve \
  --backend openai \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name random \
  --input-len 128 \
  --output-len 128 \
  --num-prompts 24 \
  --max-concurrency 8 \
  --request-rate inf \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --result-dir results \
  --result-filename baseline.json
```

The original runs reported 24/24 successful requests for both configurations.

### 3. Experiment B — Longer prompts

With the 256-token server still running:

```bash
vllm bench serve \
  --backend openai \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name random \
  --input-len 768 \
  --output-len 256 \
  --num-prompts 48 \
  --max-concurrency 8 \
  --request-rate inf \
  --ignore-eos \
  --seed 42 \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --result-dir results \
  --result-filename long-prompt-256.json
```

### 4. Repeat with the 1024-token budget

Stop the server in Terminal 1 (`Ctrl+C`) and start it again, changing only:

```bash
--max-num-batched-tokens 1024
```

Rerun **Experiment A** with output filename `larger-budget.json` and **Experiment B** with output filename `long-prompt-1024.json`. Use the same command arguments and seeds as their corresponding 256-token runs.

> **Note:** In the original short-prompt commands, the seed was not explicitly set. For a stronger future comparison, explicitly set the same `--seed` for every matched pair, warm up the server, and repeat each configuration multiple times.

## Results

All measurements below come from the benchmark JSON printed during Day 50. Values are rounded for presentation.

### Experiment A — Short prompts (24 requests × 128 input / 128 output tokens)

| Metric | Budget 256 | Budget 1024 | 1024 vs. 256 |
| --- | ---: | ---: | ---: |
| Completed / failed | 24 / 0 | 24 / 0 | Same |
| Mean TTFT (ms) | **553.11** | 565.79 | +2.3% |
| P95 TTFT (ms) | **679.68** | 702.81 | +3.4% |
| Mean ITL (ms) | **4.71** | 4.84 | +2.7% |
| P95 ITL (ms) | **5.33** | 5.65 | +6.0% |
| P99 ITL (ms) | 11.39 | **8.94** | −21.5% |
| Output throughput (tokens/s) | **815.33** | 796.25 | −2.3% |
| Request throughput (requests/s) | **6.37** | 6.22 | −2.3% |
| Test duration (s) | **3.77** | 3.86 | +2.4% |

**Interpretation:** The 256-token configuration was modestly better on most metrics, but the differences were small and each configuration was tested only once. The 1024-token configuration had a lower observed P99 ITL in this short run.

### Experiment B — Long prompts (48 requests × 768 input / 256 output tokens)

| Metric | Budget 256 | Budget 1024 | 1024 vs. 256 |
| --- | ---: | ---: | ---: |
| Completed / failed | 48 / 0 | 48 / 0 | Same |
| Mean TTFT (ms) | **1,276.01** | 1,744.40 | +36.7% |
| Median TTFT (ms) | **1,375.93** | 1,399.41 | +1.7% |
| P95 TTFT (ms) | **1,423.11** | 3,192.30 | +124.3% |
| Mean ITL (ms) | **4.95** | 7.10 | +43.3% |
| P95 ITL (ms) | **5.31** | 11.45 | +115.9% |
| P99 ITL (ms) | **11.99** | 14.78 | +23.3% |
| Mean TPOT (ms) | **4.95** | 7.10 | +43.3% |
| Output throughput (tokens/s) | **772.49** | 538.63 | −30.3% |
| Request throughput (requests/s) | **3.02** | 2.10 | −30.3% |
| Test duration (s) | **15.91** | 22.81 | +43.4% |

The long-prompt test produced **36,864 input tokens** and **12,288 output tokens** in each configuration. The 256-token setting produced around **43.4% higher output tokens/second** than the 1024-token setting, or equivalently, the larger budget produced **30.3% fewer output tokens/second** relative to the baseline. Both statements describe the same comparison with different denominators.

**Interpretation:** The larger budget exhibited a pronounced regression, especially in tail TTFT and ITL. This is worth investigating, but the data alone do not establish whether the cause was scheduling, KV-cache preemption, GPU contention, caching effects, or another source of run-to-run variation.

### Operational interpretation: client vs. server concurrency

- `--max-concurrency 8` limits the **benchmark client's** in-flight requests.
- `--max-num-seqs 4` limits how many sequences the **server scheduler** can include in an iteration.
- The benchmark's `queue_times` are not a direct measurement of vLLM's internal waiting queue; client concurrency gating can create waiting time before requests reach the server.
- Do not treat `max_concurrent_requests` from the benchmark JSON as proof of the server's instantaneous active-sequence count. For scheduler behavior, inspect the vLLM `/metrics` endpoint and logs.

## Example SLO-based decision

For classroom evaluation, assume the following **hypothetical** chatbot SLOs:

- P95 TTFT < **2,000 ms**
- P95 ITL < **10 ms**

| Long-prompt run | P95 TTFT SLO | P95 ITL SLO | Provisional choice |
| --- | --- | --- | --- |
| 256-token budget | Pass (1,423 ms) | Pass (5.31 ms) | **Candidate** |
| 1024-token budget | Fail (3,192 ms) | Fail (11.45 ms) | Not preferred |

**Decision:** Prefer the 256-token configuration *provisionally for this tested workload*. A real production decision would require repeated runs, traffic representative of actual users, a defined error budget, and monitoring under sustained load.

## Troubleshooting and follow-up work

### Monitor vLLM scheduling metrics

Run while a benchmark is active:

```bash
curl -s http://127.0.0.1:8000/metrics | \
  grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total)'
```

Metric names and exposed labels depend on the installed vLLM version. The preemption counter, when available, is cumulative; record it **before and after** each test and compare the change.

### Monitor GPU activity

```bash
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv \
  -l 1
```

Check the server logs for preemption, resource exhaustion, warnings, or restarts. **Preemption acts on requests** to recover KV-cache capacity; it does not mean the cache itself is "preempted." An increase in preemptions would be evidence of memory pressure, not by itself a complete root-cause diagnosis.

### Recommended next experiments

1. Repeat each short- and long-prompt configuration at least **3–5 times** with a controlled random seed; report average, spread, and run order.
2. Alternate run order (**256 → 1024 → 1024 → 256**) to reduce systematic warm-up, thermal, or cache effects.
3. Capture server-side running/waiting sequences, KV-cache usage, and changes in the preemption counter.
4. Record hardware, driver, vLLM version, and GPU power/temperature; check for competing GPU workloads.
5. Explore other prompt lengths, client concurrency limits, and decode lengths to determine whether the regression reproduces.

## Portfolio takeaways

- **Continuous batching is not unlimited concurrency:** token budgets, sequence counts, and KV-cache memory all constrain admission and progress.
- **Latency has multiple dimensions:** TTFT measures initial responsiveness; ITL captures streaming smoothness; P95/P99 help expose problematic tail behavior.
- **High GPU utilization is not proof of efficient inference:** use throughput, latency, reliability, and ideally cost/resource efficiency together.
- **More generous scheduler budgets are not always better:** in our observed long-prompt benchmark, a 4× larger token budget coincided with a 30.3% drop in output throughput.
- **Do not overclaim:** these observations motivate controlled follow-up testing, rather than a general conclusion that one token budget is optimal.

## Suggested repository layout

```text
day50-continuous-batching/
├── README.md
└── results/
    ├── baseline.json              # 128-in / 128-out; budget 256
    ├── larger-budget.json         # 128-in / 128-out; budget 1024
    ├── long-prompt-256.json       # 768-in / 256-out; budget 256
    └── long-prompt-1024.json      # 768-in / 256-out; budget 1024
```

The four JSON files were supplied in the study conversation, **not included in this README deliverable**; copy the original benchmark outputs into `results/` before pushing if they are not already present locally.

## References

- [vLLM documentation](https://docs.vllm.ai/)
- [vLLM OpenAI-compatible serving CLI](https://docs.vllm.ai/en/latest/cli/serve/)
- [vLLM benchmarking CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

---

**Day 50 retained concept:** Continuous batching improves serving efficiency by rescheduling active sequences across generation iterations. Chunked prefill balances prompt processing with decoding, while `max_num_seqs`, `max_num_batched_tokens`, and KV-cache capacity constrain the scheduler. Tune these parameters by measuring both throughput and tail latency against application SLOs; do not assume increasing a budget improves performance.
