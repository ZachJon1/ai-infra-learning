# Reproducibility protocol

## Environment prerequisites

- Ubuntu host with NVIDIA drivers, CUDA-capable RTX 3070 or comparable GPU, working `nvidia-smi`.
- Python virtual environment with vLLM 0.33.0 already installed, and `Qwen/Qwen2.5-0.5B-Instruct` model present in the Hugging Face cache for offline runs.
- The scripts use the Python standard library. No separate benchmark Python dependencies are required.
- Ensure another GPU service is not consuming the memory the vLLM engine needs. In the observed experiment, the earlier Docker `inference-gpu` container was stopped.

## Verify configuration

```bash
nvidia-smi
vllm --version
ss -ltnp | grep ':8000' || true
```

### Cache ON server

```bash
HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype float16 \
  --max-model-len 2048 --gpu-memory-utilization 0.65
```

Verify startup configuration reports `enable_prefix_caching=True` before running cache-ON tests; defaults can differ by version. In a second terminal:

```bash
curl -i http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models | python -m json.tool
python scripts/smoke_test.py
```

## Fixed-concurrency throughput sweep

```bash
python scripts/day55_benchmark_vllm.py \
  --concurrency 1 2 --requests-per-level 40 \
  --warmup 3 --max-tokens 64 \
  --output-dir benchmark_results/sweep_1_2

python scripts/day55_benchmark_vllm.py \
  --concurrency 4 8 16 --requests-per-level 160 \
  --warmup 5 --max-tokens 64 \
  --output-dir benchmark_results/sweep_4_8_16
```

The files `summary.csv`, `requests.csv`, and `configuration.json` will appear in each output directory. Different durations, thermal conditions, and workload order can affect results. The archival `results/concurrency_summary.csv` combines **observed** values, not the output of a fresh local run.

## Cache ON experiment

```bash
python scripts/day55_stage4_cache_benchmark.py --mode shared --cache-label on --requests 32 --concurrency 4
python scripts/day55_stage4_cache_benchmark.py --mode unique --cache-label on --requests 32 --concurrency 4
```

## Cache OFF experiment

Gracefully stop vLLM with Ctrl+C. Confirm port release. Restart with the same arguments plus `--disable-prefix-caching`:

```bash
HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype float16 \
  --max-model-len 2048 --gpu-memory-utilization 0.65 \
  --disable-prefix-caching
```

Verify startup reports `enable_prefix_caching=False`.

```bash
python scripts/day55_stage4_cache_benchmark.py --mode shared --cache-label off --requests 32 --concurrency 4
python scripts/day55_stage4_cache_benchmark.py --mode unique --cache-label off --requests 32 --concurrency 4
```

**Warning:** `--cache-label off` is not a server control; the server start flag is the control.

## GPU telemetry

In a second terminal, before a run:

```bash
mkdir -p local_logs
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.graphics \
  --format=csv -l 1 > local_logs/gpu_samples.csv
```

Stop with Ctrl+C after the run. Watch power/temperature before increasing load. Do not change laptop/workstation power limits without checking thermal/firmware requirements.

## Interpreting the measurements

- `requests/sec = successful requests / elapsed wall time`.
- `output tokens/sec = sum(completion_tokens for successful requests) / elapsed wall time`.
- Latency is client-side, end-to-end, complete response (non-streaming).
- Latency percentiles exclude failed requests; **report failure rate alongside them**.
- Prefix-cache hit ratio is token-level, computed from change in cache-hit and query counters over the run. N/A is not zero.
- Shared and unique workloads have different token demands, so compare ON vs OFF **within** each workload.
- The model's maximum context length is **2,048**; a prompt longer than that can be rejected with HTTP 400, not a CUDA OOM.
