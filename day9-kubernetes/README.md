# Day 9 — Kubernetes Fundamentals

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Understand Pods, ReplicaSets, Deployments, the scheduler, and kubelet.
- Learn how Kubernetes continuously reconciles desired and actual state.
- Distinguish bare Pods from controller-managed workloads.

## Core topics
- Pods
- ReplicaSets
- Deployments
- Scheduler
- kubelet
- Labels and selectors
- Declarative desired state

## Key lessons
- A ReplicaSet maintains the requested number of Pods.
- A Deployment manages ReplicaSets and enables controlled rollouts.
- The scheduler chooses a node; the kubelet runs the assigned Pod.
- Bare Pods are not automatically recreated by a higher-level controller.

## Commands to retain
```bash
kubectl get pods
kubectl get deployments
kubectl get replicasets
kubectl describe pod <pod>
kubectl apply -f <manifest>
kubectl delete -f <manifest>
```

## Interview takeaway
Kubernetes is a reconciliation system: controllers continually compare desired state with actual state and act to reduce the difference.
