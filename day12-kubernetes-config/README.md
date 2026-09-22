# Day 12 — ConfigMaps, Secrets & Configuration Rollouts

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Manage application configuration separately from container images.
- Understand ConfigMaps, Secrets, environment injection, and mounted configuration.
- Recognize when configuration changes do and do not reach running Pods.

## Core topics
- ConfigMaps
- Secrets
- `envFrom`
- `configMapKeyRef`
- `secretKeyRef`
- Volume-mounted configuration
- Rollout restarts

## Key lessons
- Environment variables are captured when the container starts.
- Updating a ConfigMap does not automatically update environment variables in existing containers.
- Mounted configuration can behave differently depending on how the application reads it.
- Missing required keys can prevent container creation.

## Commands to retain
```bash
kubectl get configmap
kubectl get secret
kubectl describe pod <pod>
kubectl rollout restart deployment <deployment>
kubectl rollout status deployment <deployment>
```

## Interview takeaway
Always determine how configuration enters the container before assuming that changing the Kubernetes object changes the running application.
