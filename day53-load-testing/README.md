# Day 53 — LLM Inference Load Testing with vLLM

**AI Infrastructure Engineering · Hands-on performance engineering lab · November 6, 2026**

A reproducible study of **request throughput, SLO-constrained goodput, tail latency, and scheduler admission pressure** for a local LLM inference server. I benchmarked a vLLM-served Qwen model on a laptop GPU, identified first-token latency degradation as load increased, and tested a scheduler configuration change using an **A/B/A experiment**.

## Key result

At an offered load of **4 requests/second**, increasing vLLM's `--max-num-seqs` from **4 to 8** raised SLO-compliant goodput from **2.93–3.01 to 3.96 requests/second**, while overall completed throughput remained approximately **3.97 requests/second**. Restoring the four-sequence limit reproduced the latency degradation.

| Metric | A1: 4 sequences | B: 8 sequences | A2: 4 sequences |
|---|---:|---:|---:|
| Successful requests | 480/480 | 480/480 | 480/480 |
| Completed throughput (req/s) | 3.97 | 3.97 | 3.97 |
| **Goodput (req/s)** | **2.93** | **3.96** | **3.01** |
| Requests meeting both SLOs | 354/480 | **479/480** | 364/480 |
| SLO compliance | 73.75% | **99.79%** | 75.83% |
| Mean TTFT (ms) | 344.89 | **67.55** | 298.53 |
| p95 TTFT (ms) | 1,267.01 | **97.90** | 1,016.30 |
| p95 TPOT (ms) | 13.23 | 15.37 | **12.65** |
| p95 end-to-end latency (ms) | 2,073.62 | **1,052.50** | 1,787.54 |

Relative to A1, the tuned configuration increased goodput by approximately **35.3%** and reduced p95 time to first token (TTFT) by approximately **92.3%**. Its p95 time per output token (TPOT) was roughly **16.2% higher**—a reminder that admission and decoding performance involve trade-offs.

## Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, 8 GiB VRAM |
| OS | Linux |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Serving engine | vLLM, OpenAI-compatible chat endpoint |
| API address | `http://127.0.0.1:8000/v1/chat/completions` |
| Model dtype | `half` (FP16) |
| Maximum model length | `2048` |
| GPU memory utilization parameter | `0.70` |
| Tuning parameter | `--max-num-seqs 4` versus `8` |

**Prerequisites:** A working NVIDIA/CUDA environment, Python environment with the `vllm` CLI installed, sufficient GPU memory for the model, `curl`, and access to the model weights. The commands reflect the CLI used for this experiment; options may differ across vLLM releases.

## Benchmark design

- **Traffic model:** Open-loop, Poisson-arrival requests (`--burstiness 1.0`).
- **Load sweep:** 1, 2, 3, and 4 offered requests/second, with 120, 240, 360, and 480 requests respectively; approximately two minutes per run.
- **Workload:** Random prompts with 128 requested input tokens and 64 output tokens (`--ignore-eos`). The actual chat-template-expanded inputs contained 157 tokens per request.
- **Random seed:** `42`.
- **Illustrative per-request SLOs:** TTFT ≤ **500 ms** **and** end-to-end latency ≤ **2,000 ms**.
- **Service-level acceptance example:** At least **95%** of requests satisfy both per-request SLOs.

**Throughput** counts all completed requests per second; **goodput** counts only requests meeting *both* latency SLOs per second. The latency limits and 95% threshold are lab objectives, **not production guarantees**.

## Reproduce the experiment

### 1. Start the inference server

In terminal 1, choose either `4` (baseline) or `8` (tuned):

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 8
```

Check the model endpoint from another terminal:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

### 2. Monitor scheduler state

In terminal 2, create the results directory and record server metrics:

```bash
mkdir -p results

while true; do
  date -Is
  curl -fsS http://127.0.0.1:8000/metrics \
    | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)\{'
  sleep 1
done | tee results/vllm-seqs8-metrics.txt
```

Use a different output filename (for example, `vllm-seqs4-validation-metrics.txt`) when running the four-sequence configuration. Stop monitoring using `Ctrl+C` after the benchmark.

Optional GPU telemetry:

```bash
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw,clocks.sm,memory.used \
  --format=csv -l 1 | tee results/gpu-detailed.csv
```

### 3. Run the 4 req/s benchmark

In terminal 3:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 64 \
  --num-prompts 480 \
  --request-rate 4 \
  --ignore-eos \
  --seed 42 \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,95,99 \
  --goodput ttft:500 e2el:2000 \
  --save-result \
  --save-detailed \
  --result-dir results \
  --result-filename tuned-4rps-seqs8.json
```

For the original configuration, restart the server with `--max-num-seqs 4` and save the benchmark to `baseline-4rps.json`. Restore the four-sequence configuration again for the validation run and save to `validation-4rps-seqs4.json`. **Only change one serving parameter**; keep the workload and other server settings fixed.

For a lower-rate load sweep, adjust `--request-rate` and `--num-prompts` together: `1/120`, `2/240`, and `3/360`.

> The benchmark displayed a warning that sampling temperature was not explicitly set to greedy mode. Reproduction should record the vLLM version and sampling configuration; for a new controlled experiment, consider adding a consistent `--temperature 0` across all runs.

## Load-sweep observations

| Offered load (req/s) | Achieved throughput (req/s) | Goodput (req/s) | p95 TTFT (ms) | p95 E2E (ms) |
|---|---:|---:|---:|---:|
| 1 (repeat baseline) | 0.99 | 0.99 | 63.82 | 819.25 |
| 2 | 1.99 | 1.99 | 101.60 | 860.15 |
| 3 | 2.98 | 2.95 | 362.44 | 1,153.89 |
| 4 (4 sequences, A1) | 3.97 | 2.93 | 1,267.01 | 2,073.62 |
| 4 (8 sequences, B) | 3.97 | **3.96** | **97.90** | **1,052.50** |

Increasing load initially increased both throughput and goodput. At 4 req/s with four scheduled sequences, **completed throughput still tracked demand**, but high TTFT caused widespread SLO misses. Increasing scheduler concurrency sharply reduced that first-token delay.

## Scheduler telemetry

The `/metrics` logs provide more direct evidence than client-side concurrency alone:

| Active one-second samples | 4 sequences (A2) | 8 sequences (B) |
|---|---:|---:|
| Total active samples | 119 | 115 |
| Mean running requests | 3.02 | 3.48 |
| Maximum running requests | 4 | 8 |
| Mean waiting requests | 0.97 | 0.02 |
| Maximum waiting requests | 6 | 1 |
| Samples with ≥1 waiting request | **46 (38.7%)** | **2 (1.7%)** |
| Peak reported KV-cache occupancy | 0.225% | 0.420% |

The four-sequence configuration frequently had waiting requests while its running slots were full. The eight-sequence configuration rarely showed queueing at the one-second sampling resolution. **The KV-cache occupancy metric is a fraction of allocated KV-cache capacity, not total GPU VRAM usage.**

### Interpretation

The A/B/A performance pattern and queue telemetry together support **scheduler admission contention** as a major cause of the observed TTFT degradation. Allowing eight sequences reduced admission waits and improved SLO-constrained goodput **without requiring more completed requests per second**. A moderate increase in TPOT illustrates that higher concurrency can make per-request decoding slower even as first-token responsiveness improves.

## Repository layout

Suggested layout once the README, report, and local result files are placed together:

```text
day53-load-testing/
├── README.md
├── day53_vllm_load_testing_report.md
└── results/
    ├── baseline-1rps.json
    ├── baseline-3rps.json
    ├── baseline-4rps.json
    ├── tuned-4rps-seqs8.json
    ├── validation-4rps-seqs4.json
    ├── vllm-seqs8-metrics.txt
    ├── vllm-seqs4-validation-metrics.txt
    └── gpu-detailed.csv
```

The tree is illustrative; add the other benchmark outputs and GPU logs you collected locally. Large detailed JSON outputs and telemetry may be stored separately or excluded from Git, but preserve enough artifacts to substantiate the figures.

For methods, a fuller comparison, and limitations, see [the Day 53 experiment report](day53_vllm_load_testing_report.md).

## Limitations and next steps

These results cover one small LLM, one RTX 3070 laptop GPU, a short synthetic fixed-token workload, and approximately two-minute trials. They should **not** be generalized to larger models, longer contexts, different GPUs, or real-world traffic. The run order was not randomized, hardware temperature/power conditions were not fully controlled across the final A/B/A trials, and one-second queue sampling can miss brief waiting events.

Useful next experiments: repeat A/B/A over multiple seeds and randomized run orders; test additional `--max-num-seqs` settings; vary prompt and generation lengths; test load above 4 req/s; monitor GPU clocks, thermals, and power throughout; and run longer soak tests.

## Lessons learned

1. **Throughput is not goodput:** all requests can complete successfully while many violate latency SLOs.
2. **TTFT and TPOT expose different bottlenecks:** queueing and admission may dominate TTFT, while decoding concurrency affects TPOT.
3. **Use server telemetry:** `num_requests_running`, `num_requests_waiting`, and KV-cache occupancy help interpret client-side latency.
4. **Validate tuning changes:** an A/B/A test provides stronger evidence than a one-off before/after result, though further replication is still needed.

**Day 53 outcome:** A measured, reproducible inference-serving performance investigation with a documented scheduler tuning improvement. Next topic: **Day 54 — Inference Autoscaling**.
