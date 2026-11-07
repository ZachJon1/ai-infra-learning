# Day 53 — LLM Inference Load Testing and vLLM Scheduler Tuning

**Date:** November 6, 2026  
**Hardware:** NVIDIA GeForce RTX 3070 (8 GiB VRAM)  
**Server:** vLLM serving `Qwen/Qwen2.5-0.5B-Instruct` locally via `/v1/chat/completions`  
**Project:** AI Infrastructure Engineering — Day 53

## Executive summary

An open-loop, Poisson-arrival benchmark at 4 requests/second completed all 480 requests on both scheduler configurations. Increasing `--max-num-seqs` from **4 to 8** improved SLO-constrained goodput from **2.93–3.01 requests/s** to **3.96 requests/s**, while completed throughput stayed near 3.97 requests/s. SLO compliance improved from **73.75–75.83%** to **99.79%**. Restoring the four-sequence configuration reproduced the degradation (A/B/A validation). Server-side one-second metrics showed a nonempty scheduler waiting queue in **46/119 active samples** with 4 sequences, versus **2/115** with 8 sequences. The results strongly support scheduler admission contention as an important cause of first-token latency, within this workload.

## Experimental design

- **Traffic:** Poisson arrivals at 4 requests/second (`--request-rate 4`, `--burstiness 1.0`); 480 requests per run, approximately 121 seconds per run. Initial load sweep also tested 1, 2, and 3 req/s.
- **Workload:** random dataset; `--random-input-len 128` (actual tokenized chat input: 157 tokens/request) and `--random-output-len 64` (`--ignore-eos`).
- **Model/serving:** `Qwen/Qwen2.5-0.5B-Instruct`, FP16 (`--dtype half`), 2048 maximum model length, `--gpu-memory-utilization 0.70`, local API at `127.0.0.1:8000`.
- **Single changed parameter:** `--max-num-seqs`: A1=4, B=8, A2=4. The seed was fixed (`--seed 42`) and token-length targets held constant.
- **Illustrative SLOs:** TTFT ≤ 500 milliseconds **and** end-to-end latency ≤ 2000 milliseconds. Goodput is the number of requests satisfying both conditions divided by benchmark duration. A ≥95% compliance criterion was used for discussing a serviceable operating point, not as a production contract.

## A/B/A results

| Metric | A1: 4 sequences | B: 8 sequences | A2: 4 sequences |
|---|---:|---:|---:|
| Completed requests | 480 | 480 | 480 |
| Failed requests | 0 | 0 | 0 |
| Throughput (req/s) | 3.97 | 3.97 | 3.97 |
| Goodput (req/s) | 2.93 | 3.96 | 3.01 |
| Requests meeting both SLOs | 354/480 | 479/480 | 364/480 |
| SLO compliance | 73.75% | 99.79% | 75.83% |
| Mean TTFT (ms) | 344.89 | 67.55 | 298.53 |
| p95 TTFT (ms) | 1267.01 | 97.90 | 1016.30 |
| p95 TPOT (ms) | 13.23 | 15.37 | 12.65 |
| p95 end-to-end latency (ms) | 2073.62 | 1052.50 | 1787.54 |
| p99 end-to-end latency (ms) | 2373.43 | 1104.99 | 2080.68 |

On B versus A1, goodput improved by **35.3%**, mean TTFT decreased by approximately **80.4%**, and p95 TTFT decreased by approximately **92.3%**. Conversely, p95 TPOT was higher by approximately **16.2%**, demonstrating the throughput/latency trade-off associated with greater concurrent decoding.

## Server-side scheduler evidence

| Metric (active one-second samples) | B: 8 sequences | A2: 4 sequences |
|---|---:|---:|
| Active samples | 115 | 119 |
| Mean running requests | 3.48 | 3.02 |
| Maximum running requests | 8 | 4 |
| Mean waiting requests | 0.02 | 0.97 |
| Maximum waiting requests | 1 | 6 |
| Samples with nonempty waiting queue | 2/115 (1.74%) | 46/119 (38.66%) |
| Maximum KV-cache occupancy reported | 0.420% | 0.225% |

The waiting queue was present substantially more often with the four-sequence scheduler. The KV-cache metric is a *fraction of allocated KV-cache capacity*, not total GPU VRAM consumption. One-second polling can miss short-lived queue events. Differences in reported cache occupancy can also reflect different engine cache allocations after restarts.

## Reproduce the workload

Start the server with either `--max-num-seqs 4` or `--max-num-seqs 8`, keeping other options unchanged:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype half \
  --max-model-len 2048 --gpu-memory-utilization 0.70 \
  --max-num-seqs 8
```

The benchmarking invocation used during A/B/A:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 128 --random-output-len 64 \
  --num-prompts 480 --request-rate 4 \
  --ignore-eos --seed 42 \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,95,99 \
  --goodput ttft:500 e2el:2000 \
  --save-result --save-detailed \
  --result-dir results \
  --result-filename tuned-4rps-seqs8.json
```

Monitor the scheduler on a separate terminal using `/metrics` and collect `vllm:num_requests_running`, `vllm:num_requests_waiting`, and `vllm:kv_cache_usage_perc` at a regular sampling interval.

## Interpretation and limitations

The A/B/A pattern plus server-side queue evidence supports the conclusion that **four-sequence admission was a major bottleneck at this load**. Increasing the scheduled-sequence limit reduced request waiting and TTFT, but did not increase total completed throughput because the offered load was already being fully served. It slightly increased TPOT, illustrating how concurrency settings trade decoding efficiency and responsiveness.

These are local, approximately two-minute runs on a single model and short fixed-length prompts. The experiments were not fully randomized or replicated across different hardware conditions; no GPU clock/temperature telemetry was paired with the final A/B/A trials. The synthetic prompt distribution is not representative of all production traffic. The benchmark emitted a warning that temperature defaults are server-dependent; it did not explicitly set `--temperature 0`. Additional validation should randomize run order, repeat runs across more seeds, control sampling temperature and GPU conditions, test longer prompts and outputs, test higher arrival rates, and monitor longer-duration stability.

## Recommended operating configuration for this test

**Use `--max-num-seqs 8`** for this model and the tested 4 req/s, 157-input/64-output workload under the illustrative 500 ms TTFT and 2000 ms end-to-end SLOs. Do not extrapolate this selection to larger models or workloads without benchmarking.

## Portfolio-ready takeaway

> Benchmarked a local Qwen2.5-0.5B vLLM inference endpoint across multiple arrival rates on an RTX 3070. Identified a latency-related service quality limit despite stable request throughput; instrumented vLLM scheduler queue and KV-cache metrics; performed an A/B/A scheduler tuning experiment. Increasing `max-num-seqs` from 4 to 8 improved measured SLO-compliant goodput from approximately 3.0 to 3.96 requests/sec, with throughput unchanged near 3.97 requests/sec. Reverting to four sequences reproduced the queueing and latency degradation.

## Key concepts to retain

Inference load testing must separate completed throughput from SLO-constrained goodput. TTFT can deteriorate sharply even when TPOT and completed throughput remain relatively stable, especially when admission concurrency is limited. Monitor scheduler running and waiting counts alongside GPU utilization and KV-cache usage, change one serving parameter at a time, and validate improvements with repeatable A/B/A or randomized trials before making production performance claims.
