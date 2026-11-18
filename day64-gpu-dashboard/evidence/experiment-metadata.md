# Experiment provenance and limitations

- Date: November 18, 2026 (America/Chicago).
- Source: user-provided terminal output and two Grafana screenshots from the Day 64 session.
- CSV: manually transcribed from the user-provided `nvidia-smi` timestamped log; **not** an original data export from the laptop.
- Dashboard screenshots: copies of user-supplied images. The screenshots show trends and a working panel layout but are not authoritative for exact numerical extrema.
- Exact benchmark summary: `load-test.txt` (236 successes, 90.62 s, 2.60 req/s, 250.01 generated tokens/s, 0.7679 s mean client latency, 1.0478 s p95 client latency).
- Sampled GPU readings: `gpu-samples.csv`; sampling about once every five seconds. Measurements are snapshots and may miss faster transients.
- Scrape state: `up=1` for `prometheus`, `dcgm-exporter`, `vllm`; user verified during session.
- DCGM image: `nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4`, using GeForce RTX 3070 Laptop GPU. Some profiling fields were unavailable; core utilization, memory, temperature, power fields worked.
- vLLM: `0.33.0` in `~/ai-infra-venv`; model Qwen2.5-0.5B-Instruct.
- Prometheus & Grafana: Docker containers with host network (`day57-prometheus`, `day60-grafana`); existing scrape jobs already configured, no config edit necessary.
- There are **no measured comparison runs at other concurrency levels**. Saturation, thermal throttling, power-limit behavior, and SLO compliance were not established.
- The attached Grafana JSON is a **reconstructed importable template**, NOT a direct export from the user's live Grafana instance. Export the real dashboard in Grafana for a pixel-exact or setting-exact artifact.
