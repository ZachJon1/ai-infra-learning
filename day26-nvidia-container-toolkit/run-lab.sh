#!/usr/bin/env bash
set -euo pipefail

lab_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image='pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime'

echo '1. Host GPU visibility'
nvidia-smi

echo '2. Container GPU visibility'
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

echo '3. Actual CUDA computation'
docker run --rm --gpus all \
  -v "$lab_dir:/lab:ro" "$image" \
  python /lab/gpu-check.py --expect visible

for selection in void none; do
  echo "4. Hide GPU using NVIDIA_VISIBLE_DEVICES=$selection"
  docker run --rm --runtime=nvidia \
    -e "NVIDIA_VISIBLE_DEVICES=$selection" \
    -v "$lab_dir:/lab:ro" "$image" \
    python /lab/gpu-check.py --expect hidden --count 0
done

echo '5. Expose only host GPU 0'
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -v "$lab_dir:/lab:ro" "$image" \
  python /lab/gpu-check.py --expect visible --count 1

echo 'All Day 26 checks passed.'
