# Day 20 — Kubernetes Capstone: AI Inference Service

## Overview

This capstone integrates the Kubernetes concepts covered during Days 9–20 into a small production-style AI inference workload.

The goal is not only to deploy an application, but to practice operating, diagnosing, and recovering a Kubernetes-hosted inference service.

The project includes:

* Dockerized Python/Flask inference API
* Kubernetes Deployment with multiple replicas
* ClusterIP Service
* ConfigMap-based application configuration
* Secret-based API authentication
* CPU and memory requests/limits
* Startup, readiness, and liveness probes
* Horizontal Pod Autoscaling
* Rolling deployment and rollback
* Service and EndpointSlice troubleshooting
* Scheduling failure diagnosis
* OOMKilled and CrashLoopBackOff diagnosis

---

## Architecture

```text
                         Client
                           |
                           v
                 Kubernetes Service
                     ClusterIP :80
                           |
                     EndpointSlice
                           |
                +----------+----------+
                |                     |
                v                     v
       inference-api Pod      inference-api Pod
             :8080                  :8080
                |                     |
                +----------+----------+
                           |
                    Flask API
                  /healthz
                  /readyz
                  /predict
                           |
                ConfigMap + Secret
                           |
                  Resource Requests
                     and Limits
                           |
                          HPA
```

---

## Project Structure

```text
day20-k8s-capstone/
├── app/
│   ├── app.py
│   └── requirements.txt
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.example.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── oom-demo.yaml
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Inference API

The Flask application exposes three endpoints.

### Health

```text
GET /healthz
```

Used by the liveness/startup health checks.

### Readiness

```text
GET /readyz
```

Reports whether the simulated model is ready to receive traffic.

### Prediction

```text
POST /predict
```

Example:

```bash
curl -X POST http://inference-service/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"value":25}'
```

Example response:

```json
{
  "model": "resnet-demo",
  "prediction": 50,
  "version": "v1"
}
```

The prediction implementation is intentionally simple because this lab focuses on infrastructure rather than model accuracy.

---

## Build the Container

For Minikube:

```bash
eval $(minikube docker-env)

docker build -t day20-inference:v1 .
```

Verify:

```bash
docker images | grep day20
```

---

## Configure the Secret

Do not commit real credentials.

Create the local Secret manifest from the provided example:

```bash
cp k8s/secret.example.yaml k8s/secret.yaml
```

Edit:

```bash
nano k8s/secret.yaml
```

and replace the placeholder API key before deployment.

---

## Deploy to Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

Verify:

```bash
kubectl get deployment -n inference
kubectl get pods -n inference
kubectl get svc -n inference
kubectl get endpointslices -n inference
kubectl get hpa -n inference
```

---

## Internal Service Test

Create a temporary curl container:

```bash
kubectl run curl-test \
  -n inference \
  --rm -it \
  --image=curlimages/curl \
  --restart=Never \
  -- sh
```

Inside the container:

```bash
curl http://inference-service/healthz
curl http://inference-service/readyz
```

Test inference:

```bash
curl -X POST http://inference-service/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"value":25}'
```

---

# Troubleshooting Experiments

## Scenario 1 — Incorrect Service Target Port

The Service `targetPort` was deliberately changed from `8080` to `9999`.

Observed state:

```text
Pods:           Ready
Service:        Present
EndpointSlices: Present
Traffic:        Failed
```

Diagnosis:

The Service correctly discovered the Pods, but forwarded traffic to a port where the application was not listening.

Key lesson:

> Endpoint discovery does not guarantee application connectivity.

Useful checks:

```bash
kubectl get endpointslices -n inference
kubectl describe svc inference-service -n inference
kubectl get pods -n inference -o wide
```

---

## Scenario 2 — Missing Kubernetes Secret

The Deployment was modified to reference:

```text
wrong-inference-secret
```

The new Pod entered:

```text
CreateContainerConfigError
```

`kubectl describe pod` reported:

```text
secret "wrong-inference-secret" not found
```

The container never started, so application logs were unavailable.

Meanwhile, the previous healthy Pods remained available because the Deployment used a rolling-update strategy.

Recovery was performed using:

```bash
kubectl rollout history deployment/inference-api -n inference

kubectl rollout undo deployment/inference-api -n inference

kubectl rollout status deployment/inference-api -n inference
```

Key lesson:

> A Deployment can remain available while its newest rollout is failing.

---

## Scenario 3 — Unschedulable CPU Request

A new Pod was configured to request:

```yaml
requests:
  cpu: "100"
```

The Minikube node provided only 16 allocatable CPU cores.

The Pod remained:

```text
Pending
```

with:

```text
NODE=<none>
```

Events reported:

```text
FailedScheduling
0/1 nodes are available: 1 Insufficient cpu
```

Even though actual node CPU utilization was very low, Kubernetes could not schedule the Pod because scheduling decisions are based primarily on resource requests rather than current utilization.

Key lesson:

> Low CPU utilization does not imply enough schedulable CPU capacity.

---

## Scenario 4 — Memory Limit / OOMKilled

A test container continuously allocated memory while configured with:

```yaml
requests:
  memory: "32Mi"

limits:
  memory: "64Mi"
```

The Pod successfully scheduled because its resource request could be satisfied.

After exceeding its memory limit, the container terminated with:

```text
Reason: OOMKilled
Exit Code: 137
```

Because it was restarted repeatedly, the Pod eventually showed:

```text
CrashLoopBackOff
```

Useful commands:

```bash
kubectl describe pod oom-demo -n inference

kubectl logs oom-demo -n inference --previous
```

Key distinction:

```text
OOMKilled
    = why the previous container terminated

CrashLoopBackOff
    = Kubernetes delaying repeated restart attempts
```

---

# Kubernetes Troubleshooting Funnel

A useful general troubleshooting sequence developed during the capstone is:

```text
kubectl get
     |
     v
kubectl describe
     |
     v
kubectl logs
     |
     +--> kubectl logs --previous
     |
     v
Inspect resource requests / limits
     |
     v
Inspect Service + selectors
     |
     v
Inspect Pod labels
     |
     v
Inspect EndpointSlices
     |
     v
Inspect port / targetPort
     |
     v
Check DNS / connectivity
     |
     v
kubectl exec when deeper inspection is required
```

The central troubleshooting question is:

> How far did Kubernetes get before the failure occurred?

Examples:

```text
Pending + NODE=<none>
    -> scheduling problem

CreateContainerConfigError
    -> container configuration problem

OOMKilled
    -> runtime memory-limit violation

CrashLoopBackOff
    -> repeated container failure

Ready Pods + no endpoints
    -> selector/label discovery problem

Endpoints exist + traffic fails
    -> Service routing/application connectivity problem
```

---

## Key Takeaways

This capstone demonstrated that Kubernetes troubleshooting is fundamentally about isolating failures by layer. Resource requests affect scheduling, while limits constrain runtime consumption. A Pod can therefore fail scheduling even when a node has low actual utilization, or it can schedule successfully and later be OOMKilled. Pod health does not guarantee Service connectivity, and existing EndpointSlices do not guarantee that the configured target port is correct. Rolling Deployments can preserve healthy old replicas while a new version fails, allowing recovery without necessarily interrupting service. Effective troubleshooting therefore starts with workload state, progresses through Events and container logs, and then examines resources, configuration, networking, and application behavior.

---

## Phase Completion

Day 20 completes the initial Kubernetes and systems-foundation portion of the AI Infrastructure roadmap.

Next:

**Day 21 — GPU Architecture: SMs, CUDA Cores, Tensor Cores, and VRAM.**
