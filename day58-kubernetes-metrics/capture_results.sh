#!/usr/bin/env bash
# Read-only evidence capture for Day 58; intentionally does not modify resources.
set -euo pipefail
expected_context='gpu-lab'
context=$(kubectl config current-context)
if [[ "$context" != "$expected_context" ]]; then
  printf 'Refusing to capture: expected context %s, got %s\n' "$expected_context" "$context" >&2
  exit 1
fi
mkdir -p evidence
stamp=$(date +%Y%m%d-%H%M%S)
out="evidence/hpa-snapshot-${stamp}.txt"
{
  echo 'DAY 58 — HPA STATE SNAPSHOT'
  date --iso-8601=seconds
  echo '### current context'
  kubectl config current-context
  echo '### HPA'
  kubectl get hpa -n day58
  echo '### Deployment'
  kubectl get deployment php-apache -n day58
  echo '### Pods'
  kubectl get pods -n day58 -o wide
  echo '### Metrics'
  kubectl top pods -n day58 || true
  echo '### HPA conditions + events'
  kubectl describe hpa php-apache -n day58
  echo '### metrics API availability'
  kubectl get apiservice v1beta1.metrics.k8s.io
} | tee "$out"
printf '\nSaved snapshot to %s\n' "$out"
