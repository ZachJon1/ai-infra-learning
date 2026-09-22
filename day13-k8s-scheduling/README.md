# Day 13 — Kubernetes Scheduling & Resource Management

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Understand requests, limits, node selection, affinity, taints, and tolerations.
- Diagnose why Pods remain Pending.
- Reason about placement constraints for CPU, memory, and accelerator workloads.

## Core topics
- CPU and memory requests
- Resource limits
- `nodeSelector`
- Node affinity
- Taints and tolerations
- Scheduler decisions

## Key lessons
- Requests influence scheduling.
- Limits constrain runtime resource usage.
- A toleration permits scheduling onto a tainted node but does not force placement there.
- Required affinity is a hard constraint; preferred affinity is a preference.

## Commands to retain
```bash
kubectl describe pod <pod>
kubectl describe node <node>
kubectl get nodes --show-labels
kubectl taint nodes <node> key=value:NoSchedule
```

## Interview takeaway
For a Pending Pod, inspect scheduler events first. CPU/memory shortages, affinity, selectors, and untolerated taints are scheduling-layer failures, not application failures.
