# GPU Model Serving Capstone — Day 55

**Local vLLM inference, performance benchmarking, prefix-cache experiments, and failure-recovery drills.**

> **Scope:** This is a reproducible, single-machine learning capstone, **not a production deployment**. The vLLM server ran directly on an Ubuntu host. A separate Dockerized FastAPI demonstration and Minikube GPU scheduling lab were also validated. The LLM was **not** deployed into Kubernetes in these experiments.

## Executive summary

I ran an OpenAI-compatible vLLM inference endpoint on an NVIDIA RTX 3070 (8 GiB), measured request and output-token throughput under increasing client concurrency, compared automatic prefix caching ON vs OFF, and conducted four failure-injection/recovery exercises covering API validation, service outage, context limits, and Kubernetes taints/tolerations.

**Key measured observations:**

- **19.149 requests/s** and **1,053.179 output tokens/s** at **16 concurrent requests** on a short, repeated-prompt, cache-enabled benchmark. Observed p95 end-to-end latency was **860.079 ms**. This is not a guaranteed sustainable service rate.
- Lowest observed p95 latency in the concurrency sweep was **684.592 ms at concurrency 4**, with **6.069 requests/s**.
- At concurrency 4, enabling prefix caching improved throughput by **21.6%** on the long shared-prefix workload (5.454 vs 4.486 req/s), with a **16.0%** lower p95 latency (761.477 vs 906.434 ms).
- For varied-context prompts, enabling caching improved throughput by **6.0%** (3.108 vs 2.933 req/s) with a **2.5%** lower p95 latency (1475.619 vs 1513.797 ms).
- **560/560** requests completed successfully in the five-level concurrency sweep, and **128/128** in the four cache tests. These samples do not establish production reliability.

## Environment

| Component | Observed configuration |
|---|---|
| OS | Ubuntu 24.04 |
| GPU | NVIDIA GeForce RTX 3070, 8 GiB |
| NVIDIA driver | 580.196.07 |
| CUDA compatibility reported by `nvidia-smi` | 13.0 (driver-reported maximum) |
| GPU power limit at test time | 55 W (reported hardware default 125 W; **not altered**) |
| Serving software | vLLM 0.33.0 |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Precision | float16 |
| Context limit | 2,048 tokens |
| vLLM GPU memory utilization setting | 0.65 |
| Host endpoint | `http://127.0.0.1:8000` |
| Kubernetes lab | Minikube `gpu-lab`, Kubernetes 1.35.1, NVIDIA Device Plugin |

## Architecture: what actually ran

```text
Client (curl / Python urllib)
       |
       v
Host Ubuntu 24.04 — 127.0.0.1:8000
       |
       v
vLLM OpenAI-compatible API
       |
       v
Qwen2.5-0.5B-Instruct / CUDA / RTX 3070

SEPARATE earlier experiments:
  Docker -> ai-inference:gpu -> FastAPI demo on localhost:8002
  Minikube gpu-lab -> Kubernetes scheduler and NVIDIA Device Plugin
```

The Minikube GPU resource advertisement (`nvidia.com/gpu=1`) does not imply that the host-based vLLM process is a Kubernetes Pod. Likewise, Kubernetes allocatable resources do not measure host GPU idleness.

## Quick reproduction

This assumes Python/vLLM and the model weights are **already installed/cached** and the NVIDIA host GPU can be used by vLLM. Model downloading or package installation are intentionally not automated.

```bash
# Terminal A: serve the local cached model, with default prefix-caching behavior.
HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype float16 \
  --max-model-len 2048 --gpu-memory-utilization 0.65

# Terminal B: confirm readiness and do a small inference request.
curl -i http://127.0.0.1:8000/health
python scripts/smoke_test.py
```

See [reproducibility steps](docs/reproducibility.md) for the benchmark commands, prefix-cache ON/OFF protocol, and GPU monitoring instructions.

## Concurrency sweep — measured results

All tests used the same short prompt, non-streaming responses, `temperature=0`, `max_tokens=64`, and a warmed model. Concurrency 1/2 used 40 requests each; levels 4/8/16 used 160 each. This should be viewed as a **warm/repeated-prompt workload** rather than representative mixed production traffic.

| Concurrency | Requests | Failures | Requests/s | Output tokens/s | Mean latency (ms) | p95 latency (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 0 | 1.284 | 70.602 | 778.886 | 814.863 |
| 2 | 40 | 0 | 2.913 | 160.203 | 686.516 | 693.865 |
| 4 | 160 | 0 | 6.069 | 333.794 | 658.863 | 684.592 |
| 8 | 160 | 0 | 10.864 | 597.542 | 735.580 | 760.779 |
| 16 | 160 | 0 | 19.149 | 1053.179 | 833.472 | 860.079 |

The highest tested level did **not** demonstrate throughput saturation. We did not measure TTFT, TPOT, multi-node scaling, or a sustained production SLO. See [benchmark analysis](docs/benchmark_analysis.md) and [raw concurrency summary](results/concurrency_summary.csv).

## Prefix-cache A/B comparison — measured results

Each run used 32 requests, concurrency 4, non-streaming responses, and `max_tokens=64`. Compared within each mode, ON vs OFF uses the same prompt generator. Runs were not repeated or interleaved, so the numbers are exploratory. The shared and unique workloads have different input-token demands and should not be compared as if only the cache policy differed.

| Prompt type | Prefix cache | Requests/s | p95 latency (ms) | Cache hit ratio (token-level) |
|---|---|---:|---:|---:|
| Long shared-prefix | ON | 5.454 | 761.477 | 92.02% |
| Long shared-prefix | OFF | 4.486 | 906.434 | N/A |
| Varied context | ON | 3.108 | 1475.619 | 9.98% |
| Varied context | OFF | 2.933 | 1513.797 | N/A |

Cache OFF was set using the **server** flag `--disable-prefix-caching`. The benchmark script's `--cache-label` is for results labeling only. See [reproducibility](docs/reproducibility.md) and the four original summary JSON files in `results/`.

## Reliability drills

| Scenario | Signal | Recovery / learning |
|---|---|---|
| Invalid chat payload | HTTP 400, `body.messages` wrong type | Correct the client; engine remained healthy |
| Graceful server shutdown | Connection refused; port 8000 had no listener | Restart vLLM, verify `/health`, `/v1/models`, generation, and GPU process |
| Oversized model context | HTTP 400, >2,048 input tokens | Reject or shorten request; not equivalent to CUDA OOM |
| Kubernetes taint without toleration | Pod `Pending`, `FailedScheduling` | Add `dedicated=gpu:NoSchedule` toleration; Pod became `Running`, then test Pod deleted |

See [incident runbook](docs/incident_runbook.md) for diagnostics and commands. Kubernetes device-plugin health remained 1/1 after the Pod lab; one GPU was allocatable.

## GPU telemetry and interpretation

During the cache workload, GPU utilization reached 100%, power was approximately 55 W (at the configured limit), and sampled temperatures reached 80°C. The power budget **may** constrain sustained clock speeds, but the provided telemetry alone does not prove thermal throttling. Power settings were not modified. Earlier idle/stopped states showed GPU temperatures of roughly 71–74°C. Treat any performance comparisons as sensitive to hardware power and thermal conditions.

## Repository contents

```text
.
├── README.md
├── .gitignore
├── docs/
│   ├── architecture.md
│   ├── benchmark_analysis.md
│   ├── incident_runbook.md
│   ├── interview_brief.md
│   └── reproducibility.md
├── scripts/
│   ├── day55_benchmark_vllm.py
│   ├── day55_stage4_cache_benchmark.py
│   └── smoke_test.py
├── k8s/
│   ├── taint-test.yaml
│   └── toleration-patch.json
└── results/
    ├── concurrency_summary.csv
    └── {shared,unique}_cache_{on,off}_c4_summary.json
```

## Limitations and next steps

- Single RTX 3070, single model, one machine; no distributed model serving or Kubernetes-based LLM deployment demonstrated.
- Cache results are from one 32-request run per configuration. Repeat with randomized run order, equalized prompt-token distributions, multi-trial statistics, and aligned GPU telemetry.
- Measured **client-side complete-response latency**, not first-token latency. No streaming TTFT or TPOT metrics.
- p99 with small samples is unstable and should not be overinterpreted.
- No authenticated external-facing serving, TLS, rate limiting, production alerting, rolling rollout, load balancer, or resilience under real traffic demonstrated.
- No evidence that all tests would maintain these rates indefinitely; the power limit and cooling deserve investigation before extended load.

## Interview-ready explanation

> I built and benchmarked a GPU-backed model-serving lab using vLLM on an RTX 3070. I measured latency and throughput as concurrency increased, then ran a controlled ON/OFF experiment for KV-prefix caching. For a long shared-prefix workload, caching improved request throughput by 21.6% in our measured run. I also injected four failure scenarios and verified recovery at the API, process, model, GPU, and Kubernetes scheduling layers. The next step would be packaging the vLLM service into Kubernetes and adding production observability and SLO-driven load testing.

**Evidence:** original benchmark code and four JSON summaries; concurrency results transcribed from experiment terminal output. No unmeasured performance or production deployment claims are made.
