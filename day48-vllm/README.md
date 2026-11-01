# Day 48 — vLLM Fundamentals: GPU-Backed LLM Serving

**AI Infrastructure & ML Systems — 220-Day Roadmap · Phase 3: Model Serving**

A hands-on introduction to serving a small language model on a local NVIDIA GPU with **vLLM**. This lab covers model initialization, an OpenAI-compatible inference API, KV-cache allocation, continuous batching, concurrent requests, and basic serving telemetry.

> **Status:** Completed. A GPU-backed vLLM server started successfully, returned an inference response, exposed Prometheus-style metrics, and handled all eight requests in the concurrency smoke test. These results demonstrate functionality, **not** production-scale capacity or a controlled performance benchmark.

## Learning objectives

- Explain the roles of **prefill** and **decode** during autoregressive generation.
- Understand the **KV cache** and vLLM's paged KV-cache memory management.
- Describe **continuous batching**, chunked prefill, and request scheduling.
- Deploy an LLM through vLLM's OpenAI-compatible HTTP API.
- Observe request, cache, and preemption metrics; test a small concurrent workload.
- Distinguish GPU memory **capacity** from **bandwidth** bottlenecks.

## Architecture

```text
Clients / concurrent requests
            |
            v
    vLLM API server (:8000)
            |
            v
       vLLM engine
       /         \
  Scheduler    KV-cache manager
       \         /
        Model execution
        (prefill / decode)
            |
            v
    CUDA + NVIDIA GPU
```

vLLM is the **serving engine**, not the language model. Here, it serves **Qwen/Qwen3-0.6B** on a single GPU.

## Environment and configuration

The following values come from the successful lab's vLLM **0.32.0** startup logs.

| Item | Configuration or observed value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` |
| vLLM version | `0.32.0` |
| Serving image | `vllm/vllm-openai` (digest recorded below) |
| GPU | One CUDA GPU; **7.65 GiB** reported total VRAM |
| Model dtype | `torch.bfloat16` |
| Max model length | `2048` tokens |
| GPU memory utilization setting | `0.65` |
| Max simultaneous scheduled sequences | `4` |
| Max scheduled tokens per iteration | `512` |
| Chunked prefill | Enabled |
| Prefix caching | Enabled |
| Model memory reported on load | **1.12 GiB** |
| Available GPU KV-cache memory | **3.38 GiB** |
| GPU KV-cache token capacity | **31,648 token positions** |

The startup log's *15.45× theoretical concurrency* assumes sequences of length 2,048 from a KV-cache capacity perspective; it **does not override** the configured `--max-num-seqs 4` scheduling limit.

## Prerequisites

- Linux host with an NVIDIA GPU and working driver (`nvidia-smi`).
- Docker Engine with NVIDIA Container Toolkit configured for GPU access.
- Internet access on first run to retrieve the container image and model weights.
- `curl`, Python 3, and a writable Hugging Face cache directory.

Check the GPU and Docker installation:

```bash
nvidia-smi
docker info >/dev/null && echo 'Docker available'
```

## Run the server

```bash
mkdir -p ~/ai-infra/day48-vllm
cd ~/ai-infra/day48-vllm

docker run --rm \
  --name day48-vllm \
  --gpus 'device=0' \
  --shm-size=1g \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34 \
  Qwen/Qwen3-0.6B \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.65 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512
```

The image digest is taken from the **successful original pull**. The model revision was not pinned in the lab; for stricter reproducibility, pin an approved model revision as well. The command uses the **positional model argument** recommended by the installed vLLM version (the original `--model` spelling worked but emitted a deprecation warning).

The port is bound to `127.0.0.1` on the host for a **local-only lab**; this is not a production-hardened, authenticated public endpoint.

Wait until the API reports startup completion. Check from another terminal:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

The result should list `Qwen/Qwen3-0.6B` as a served model.

## Send an inference request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Explain GPU scheduling in Kubernetes."}],
    "max_tokens": 80
  }' | python3 -m json.tool
```

**Observed:** HTTP `200 OK`, `prompt_tokens: 15`, `completion_tokens: 80`, and `finish_reason: "length"`. The generated text began with Qwen3's `<think>` section and was **truncated by the token limit**; this confirms a functioning endpoint, not a complete high-quality answer.

For a brief answer without Qwen3's thinking mode, try:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Explain GPU scheduling in Kubernetes briefly."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "max_tokens": 128
  }' | python3 -m json.tool
```

This second variation is a suggested follow-up, **not** part of the recorded successful measurements.

## Observe serving metrics

```bash
curl -s http://localhost:8000/metrics | grep -E \
'^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total)\{'
```

| Metric | What it measures | Lab observation after requests completed |
|---|---|---|
| `vllm:num_requests_running` | Active executing requests | `0` |
| `vllm:num_requests_waiting` | Requests queued for execution | `0` |
| `vllm:kv_cache_usage_perc` | Fraction of GPU KV-cache space in use | `0.0` |
| `vllm:num_preemptions_total` | Cumulative request preemptions | `0.0` |

The samples were collected with the server **idle**. A zero value at that instant does **not** mean the KV cache was unused during generation. For live observation, sample while sending concurrent requests:

```bash
watch -n 0.5 \
  "curl -s http://localhost:8000/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total)\\{'"
```

## Concurrent request smoke test

The repository's `concurrent_test.py` issues **8 requests** via a `ThreadPoolExecutor(max_workers=4)`, varying the output budget among 32, 64, and 96 tokens. Run it from the directory containing the script:

```bash
python3 concurrent_test.py
```

(An activated virtual environment with `python` also works; on the original host `python` was unavailable until its environment was activated.)

### Recorded results

| Request | Output tokens | End-to-end duration (s) |
|---:|---:|---:|
| 0 | 32 | 0.22 |
| 1 | 64 | 0.39 |
| 2 | 96 | 0.56 |
| 3 | 32 | 0.22 |
| 4 | 64 | 0.36 |
| 5 | 96 | 0.51 |
| 6 | 32 | 0.18 |
| 7 | 64 | 0.30 |

**Smoke-test summary:** 8/8 successful requests; **480 output tokens**; **0.343 s** mean per-request end-to-end duration (rounded). The server separately reported **48.0 generated tokens/s** for one logging interval after the test; this interval-specific number is **not** a sustained benchmark result.

**Measurement caveats:** Output budgets varied; the recorded request durations are not TTFT or inter-token latency (ITL); the test did not record whole-run wall time, GPU power/utilization over time, latency percentiles across repeated runs, or a concurrency sweep. Do not interpret the table as maximum throughput, a speedup versus sequential serving, or production SLO evidence.

## Troubleshooting notes

| Symptom | Likely area to investigate | Initial response |
|---|---|---|
| vLLM fails to initialize with GPU OOM | Available VRAM, model weights, KV-cache allocation, CUDA graph/activation overhead | Check `nvidia-smi`; reduce memory budget or context/concurrency; consider a supported quantized model if quality permits. |
| High time to first token (TTFT) | Queueing, prefill cost, long prompts, scheduler budget | Check waiting requests, input lengths, prefill scheduling, and GPU compute activity. |
| High inter-token latency (ITL) | Decode efficiency, GPU memory bandwidth, long contexts, contention | Inspect active sequences, GPU behavior, and memory/cache pressure. |
| Increasing preemptions | KV-cache capacity pressure | Inspect KV usage and concurrent token demand; tune concurrency/context or increase viable cache capacity. |
| `finish_reason: "length"` | Generation budget exhausted | Increase `max_tokens` within the context budget or disable thinking for concise Qwen3 answers. |
| `python: command not found` | Host Python alias not installed | Use `python3` or activate an existing virtual environment. |

**Capacity vs. bandwidth:** A request can decode slowly despite free VRAM if memory transfers are the bottleneck. Conversely, KV-cache capacity exhaustion can force scheduling delays or preemptions without proving memory-bandwidth saturation.

## Engineering takeaways

1. **Prefill** computes attention representations for input tokens; **decode** generates subsequent tokens while reusing previous KV entries.
2. **Paged KV-cache allocation** limits memory waste by mapping logical KV blocks to physical GPU blocks without needing one large contiguous reservation per request.
3. **Continuous batching** allows requests to join and leave execution batches as generation progresses; **chunked prefill** can share scheduling capacity with ongoing decode work.
4. **Scheduling limits** (`max_num_seqs`, `max_num_batched_tokens`) and **memory capacity** are distinct constraints.
5. Reliable optimization requires actual TTFT, ITL, throughput, queue, preemption, and GPU metrics rather than assuming more concurrency always helps.

## Next steps

- **Day 49 — LLM serving with vLLM:** Extend the client and serving configuration, test streaming, and investigate generation controls.
- **Day 50 — Continuous batching:** Explore concurrency and scheduler behavior more systematically.
- **Day 51 — KV cache:** Investigate context length and cache utilization under load.
- **Day 53 — Load testing:** Add repeated trials, workload control, TTFT/ITL, latency percentiles, and throughput measurement.
- Integrate this service into the ongoing **production-style Kubernetes GPU/ML platform** portfolio project.

## References

- [vLLM documentation](https://docs.vllm.ai/)
- [vLLM GitHub repository](https://github.com/vllm-project/vllm)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [AI infrastructure learning roadmap](https://github.com/ZachJon1/ai-infra-engineer-learning)

---

**Project context:** Day 48 of a 220-day AI Infrastructure & ML Systems learning roadmap. All lab figures above are from the recorded local run; untested follow-ups and prospective improvements are labeled accordingly.
