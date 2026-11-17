# Day 62 - Inference Latency Metrics (LaTeX source bundle)

This bundle accompanies the 18-page student/engineering chapter on inference latency for a small Qwen2.5-0.5B-Instruct model served by vLLM 0.33.0 with Prometheus and Grafana.

## Contents

- `Day62_Inference_Latency_Notes.tex` - editable chapter, including diagrams authored in TikZ and reproducible PromQL examples.
- `figures/` - vector benchmark plots (`.pdf`) and cropped images of the nine-panel Grafana dashboard (`.png`).
- `data/day62_raw_client_latencies.csv` - 110 client latency measurements, transcribed from the user's terminal logs: baseline 10 requests plus five 20-request runs.
- `data/day62_benchmark_summary.csv` - original 20-request, concurrency-1/2/5 comparison.
- `Makefile` - compilation shortcut.

## Rebuild the PDF

Requires XeLaTeX and common TeX Live packages (`tikz`, `tcolorbox`, `listings`, `fontspec`, etc.). The source uses the Noto Serif, Noto Sans and DejaVu Sans Mono font families. Build in this folder:

```bash
xelatex -interaction=nonstopmode -halt-on-error Day62_Inference_Latency_Notes.tex
xelatex -interaction=nonstopmode -halt-on-error Day62_Inference_Latency_Notes.tex
```

Or run `make pdf` if Make is available.

## Interpretation caveats

All actual measurements came from the Day 62 study session; no additional live benchmark was run for this document. Client request latencies, vLLM internal histograms, and Prometheus bucket estimates are distinct measures. The benchmarks are short, single-run experiments with a fixed prompt, and their high percentiles are not statistically robust production SLO evidence. The new `request_latency` bucket boundaries are fine near 100-150 ms but remain coarse between 300 and 500 ms. DCGM-exporter data was not used to attribute bottlenecks to GPU utilization.

Further documentation sources and student exercises are included in the PDF/TeX chapter.
