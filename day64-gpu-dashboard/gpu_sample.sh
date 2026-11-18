#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-evidence/gpu-live.csv}"
mkdir -p "$(dirname "$OUT")"
echo "Capturing GPU telemetry into $OUT. Press Ctrl+C to stop."
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv -l 5 | tee "$OUT"
