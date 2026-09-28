#!/usr/bin/env bash

nvidia-smi \
  --query-gpu=timestamp,pstate,utilization.gpu,memory.used,memory.free,clocks.sm,clocks.mem,temperature.gpu,power.draw \
  --format=csv \
  -l 1
