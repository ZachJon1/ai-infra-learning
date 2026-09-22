# Day 19 — Kubernetes Observability, Debugging & Failure Diagnosis

## Objectives

Day 19 focused on developing a systematic approach to diagnosing Kubernetes failures rather than immediately changing configuration.

The main troubleshooting workflow used was:

```text
GET → DESCRIBE → LOGS → METRICS → EXEC
```

The specific path depends on the observed failure state.

## Core Diagnostic Commands

```bash
kubectl get pods
kubectl get pods -o wide
kubectl describe pod <pod>
kubectl logs <pod>
kubectl logs <pod> --previous
kubectl top pods
kubectl top nodes
kubectl exec -it <pod> -- /bin/sh
kubectl get endpointslices
kubectl describe svc <service>
kubectl get events --sort-by=.metadata.creationTimestamp
```

## Failure States Covered

### ImagePullBackOff

A Deployment referenced an invalid nginx image tag.

Observed progression:

```text
ErrImagePull
→ retry
→ BackOff
→ ImagePullBackOff
```

The decisive evidence came from Pod events:

```text
manifest ... not found
```

The container never started, so application logs were not useful.

**Fix:** correct the image tag.

---

### CrashLoopBackOff

A BusyBox worker successfully started but exited with a non-zero exit code because required application configuration was missing.

Diagnostic distinction:

```text
ImagePullBackOff
→ container never starts

CrashLoopBackOff
→ container starts
→ process fails
→ Kubernetes restarts it
```

Useful evidence included:

```bash
kubectl describe pod <pod>
kubectl logs <pod>
kubectl logs <pod> --previous
```

**Fix:** provide the missing configuration and ensure the process remains alive.

---

### Running but NotReady

An nginx Pod was running, but its readiness probe requested:

```text
/healthz
```

which returned HTTP 404.

The application itself was accessible through `/`, proving that the process and port were functioning.

Observed state:

```text
STATUS: Running
READY: 0/1
```

EndpointSlice inspection showed that the endpoint existed but was not considered ready for normal Service traffic.

**Fix:** configure the readiness probe to use a valid health endpoint.

Key lesson:

```text
Running ≠ Ready
```

---

### Service Selector / EndpointSlice Failure

Two `model-api` Pods were healthy:

```text
app=model-api
```

but the Service selected:

```text
app=inference-api
```

As a result, the Service had no usable endpoints.

Diagnostic path:

```text
Healthy Pods
     ↓
Inspect Service selector
     ↓
Inspect EndpointSlice
     ↓
Compare selector with Pod labels
```

**Fix:** change the Service selector to:

```yaml
selector:
  app: model-api
```

Restarting Pods would not solve the problem because the failure was in declarative Service configuration.

---

### FailedScheduling / Insufficient CPU

A Deployment requested:

```yaml
resources:
  requests:
    cpu: "20"
    memory: "512Mi"
```

The Minikube node had only 16 allocatable CPUs.

Observed state:

```text
Status: Pending
Node: <none>
PodScheduled: False
```

Scheduler event:

```text
0/1 nodes are available: 1 Insufficient cpu
```

Because the Pod was never scheduled and its container never started, application logs were not useful.

Possible solutions include:

* Reduce the CPU request.
* Add or use a node with sufficient compute capacity.

The failed rollout also demonstrated Kubernetes preserving the previous healthy replica while the replacement Pod remained unschedulable.

## Troubleshooting Mental Model

Use the failure state to decide where to investigate first:

```text
Pending
→ scheduling, resources, affinity, selectors, taints, storage

ContainerCreating
→ volume mounts, runtime, networking, node preparation

ImagePullBackOff
→ image name, tag, registry, authentication

CrashLoopBackOff
→ application startup, configuration, dependencies, previous logs

OOMKilled
→ memory limits and memory consumption

Running 0/1
→ readiness configuration

Healthy Pods + Service failure
→ Service selector, ports, EndpointSlices, networking
```

## Key Lesson

Kubernetes troubleshooting should begin by collecting evidence rather than immediately changing resources. Use `kubectl get` to identify the failing state, `kubectl describe` for Kubernetes conditions and events, `kubectl logs` or `--previous` for application failures, `kubectl top` for current resource behavior, and `kubectl exec` for investigation from inside a running container. Always separate the visible symptom from the underlying root cause and correlate Pod, Service, EndpointSlice, resource, and node-level evidence before applying a fix.
