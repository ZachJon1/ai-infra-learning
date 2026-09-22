# Day 10 — Kubernetes Networking

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Understand how Services provide stable access to changing Pods.
- Use labels, selectors, EndpointSlices, DNS, and Ingress to trace traffic.
- Diagnose Service-to-Pod connectivity problems.

## Core topics
- Pod IPs
- Services
- ClusterIP
- EndpointSlices
- Kubernetes DNS
- Ingress
- NetworkPolicy basics

## Diagnostic path
```text
Client
  ↓
Ingress
  ↓
Service
  ↓
EndpointSlice
  ↓
Pod
```

## Commands to retain
```bash
kubectl get svc
kubectl describe svc <service>
kubectl get endpointslices
kubectl get pods --show-labels
kubectl get ingress
```

## Key lesson
A Service selector must match Pod labels. Healthy Pods do not help if the Service has no matching endpoints.

## Interview takeaway
When a Service exists but traffic fails, verify selectors, EndpointSlices, target ports, DNS, and the application listener before restarting Pods.
