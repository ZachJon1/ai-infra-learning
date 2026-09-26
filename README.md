# AI Infrastructure & ML Systems Learning Labs

Hands-on labs documenting my progression toward **AI Infrastructure / ML Systems engineering**.

This repository focuses on building practical skills in Linux, containers, Kubernetes, GPU infrastructure, model serving, observability, distributed systems, and production ML infrastructure.

The goal is not only to learn tools, but to develop the ability to **deploy, operate, troubleshoot, and design production-style AI systems**.

---

## Current Progress
**Roadmap progress:** Day 21 | GPU architecture refresher: SMs, CUDA cores, Tensor Cores, VRAM | ✅  
**Current:** GPU Architecture Refresher

### Learning Path

```text
Linux
  ↓
Docker
  ↓
Kubernetes
  ↓
GPU Infrastructure
  ↓
Model Serving / Inference
  ↓
Observability
  ↓
Cloud + Terraform
  ↓
Distributed Systems
  ↓
Distributed ML
  ↓
LLM Infrastructure
  ↓
Reliability + System Design
```

The current phase is focused on building a strong Kubernetes operations foundation before moving into GPU infrastructure and production model serving.

---

## Skills Covered So Far

| Area | Topics |
|---|---|
| Linux / Systems | Processes, networking, storage, I/O, services, resource management |
| Containers | Docker images, containers, volumes, networking, Docker Compose |
| Kubernetes Fundamentals | Pods, ReplicaSets, Deployments, scheduler, kubelet |
| Kubernetes Networking | Services, EndpointSlices, DNS, Ingress, NetworkPolicy |
| Storage | `emptyDir`, PVs, PVCs, StorageClasses, StatefulSets |
| Configuration | ConfigMaps, Secrets, environment variables, mounted configuration |
| Scheduling | Requests, limits, node selectors, affinity, taints and tolerations |
| Health | Liveness, readiness, and startup probes |
| Autoscaling | HPA, resource-based scaling, stabilization, queue-based scaling concepts |
| Package Management | Helm charts, values, templates, releases, upgrades, rollbacks |
| Security | RBAC, Roles, RoleBindings, ServiceAccounts |
| Batch Workloads | Jobs, retries, parallelism, Indexed Jobs |
| Scheduled Workloads | CronJobs, concurrency policies, scheduling controls |
| Pod Lifecycle | Init containers, graceful termination, lifecycle hooks, TTL cleanup |

---

## Repository Structure

```text
ai-infra-learning/
│
├── day5-web/
├── day7-data/
├── day7-gpu-output/
├── day7-python/
├── day8-compose/
│
├── day9-kubernetes/
├── day10-kubernetes-networking/
├── day11-kubernetes-storage/
├── day12-kubernetes-config/
├── day13-k8s-scheduling/
├── day14-probes/
├── day15-autoscaling/
├── day16-helm/
├── day17-rbac/
├── day18-workloads/
│
└── README.md
```

The roadmap began before all labs were committed to this repository, so the repository currently contains the practical artifacts preserved from the later systems and Kubernetes portions of the learning path.

---

## Selected Hands-On Labs

### Kubernetes Networking

Built and tested Kubernetes networking components including Services, Ingress configuration, NetworkPolicy, and service-to-pod connectivity.

**Directory:** [`day10-kubernetes-networking/`](day10-kubernetes-networking/)

### Persistent Storage

Worked with ephemeral and persistent storage, including:

```text
emptyDir
PersistentVolumes
PersistentVolumeClaims
StatefulSets
```

**Directory:** [`day11-kubernetes-storage/`](day11-kubernetes-storage/)

### Configuration and Secrets

Explored multiple ways of providing application configuration:

```text
ConfigMaps
Secrets
envFrom
configMapKeyRef
secretKeyRef
volume-mounted configuration
```

The labs also intentionally introduce broken configurations to practice failure diagnosis.

**Directory:** [`day12-kubernetes-config/`](day12-kubernetes-config/)

### Scheduling and Resource Management

Tested Kubernetes scheduling behavior using:

```text
CPU requests
resource limits
node selectors
node affinity
taints
tolerations
unschedulable workloads
```

**Directory:** [`day13-k8s-scheduling/`](day13-k8s-scheduling/)

### Health Probes

Built examples demonstrating the differences between:

```text
startup probes
readiness probes
liveness probes
```

and how incorrect health checks affect application availability and container restarts.

**Directory:** [`day14-probes/`](day14-probes/)

### Kubernetes Autoscaling

Configured Horizontal Pod Autoscaling and experimented with application load, CPU utilization, replica scaling, and scaling behavior.

**Directory:** [`day15-autoscaling/`](day15-autoscaling/)

### Helm

Created and modified a Helm chart for a Kubernetes application, including:

```text
values
templates
Deployments
Services
Ingress
HPA
ServiceAccounts
release upgrades
rollbacks
```

**Directory:** [`day16-helm/`](day16-helm/)

### RBAC and ServiceAccounts

Practiced Kubernetes authorization using Roles, RoleBindings, ServiceAccounts, and restricted API access.

**Directory:** [`day17-rbac/`](day17-rbac/)

### Jobs, CronJobs, and Workload Lifecycle

Implemented and tested:

```text
Jobs
failure retries
backoffLimit
completions
parallelism
Indexed Jobs
CronJobs
concurrencyPolicy
init containers
graceful shutdown
TTL cleanup
```

**Directory:** [`day18-workloads/`](day18-workloads/)

---

## Troubleshooting Practice

A major focus of this repository is intentionally breaking infrastructure and then diagnosing the failure rather than only deploying successful examples.

Examples include:

```text
Missing ConfigMap keys
Incorrect Secret names
Incorrect Secret keys
Unschedulable Pods
Resource constraints
Taints without matching tolerations
Readiness failures
Liveness failures
Job retry exhaustion
CronJob concurrency behavior
Init-container dependencies
```

This troubleshooting work is intended to build the operational reasoning required for AI infrastructure and ML systems roles.

---

## Why This Is AI Infrastructure

The Kubernetes foundation in this repository is preparation for workloads such as:

```text
Users
  ↓
Load Balancer
  ↓
Kubernetes Service
  ↓
Inference Pods
  ↓
GPU
  ↓
Model Server
  ↓
Metrics / Observability
```

The later phases of the roadmap will extend these labs into production-style AI systems where Kubernetes manages GPU-backed model-serving workloads.

---

## Upcoming Work

### GPU Infrastructure

Upcoming labs will cover:

```text
GPU architecture
CUDA runtime and drivers
nvidia-smi
GPU memory and OOM diagnosis
PyTorch GPU profiling
NVIDIA Container Toolkit
Kubernetes NVIDIA Device Plugin
GPU scheduling
MIG
multi-GPU workloads
```

### Model Serving

The next major portfolio phase will focus on:

```text
FastAPI inference services
GPU-backed inference
batching
dynamic batching
latency vs throughput
model loading and caching
NVIDIA Triton
vLLM
continuous batching
KV cache
quantization
load testing
autoscaling
```

This phase will produce a larger end-to-end model-serving project.

---

## Planned Portfolio Architecture

The long-term project will evolve toward an architecture similar to:

```text
                       USERS
                         │
                  Load Balancer
                         │
                 Kubernetes Service
                         │
             ┌───────────┴───────────┐
             │                       │
       Inference Pod           Inference Pod
             │                       │
            GPU                     GPU
             │                       │
        vLLM / Triton           vLLM / Triton
             │
             ▼
       Metrics Collection
             │
      Prometheus + DCGM
             │
          Grafana
```

The goal is to demonstrate not only deployment, but also GPU scheduling, model serving, observability, autoscaling, reliability, and troubleshooting.

---

## Engineering Approach

Each topic is approached through a combination of:

```text
Concepts
   ↓
Hands-on implementation
   ↓
Break / fix exercises
   ↓
Troubleshooting
   ↓
Interview-style reasoning
   ↓
Documentation
```

The emphasis is on understanding **why infrastructure behaves the way it does** rather than memorizing commands.

---

## Roadmap Goal

The broader roadmap progresses through:

```text
Kubernetes
→ GPU Infrastructure
→ Model Serving
→ Observability
→ Cloud Infrastructure
→ CI/CD
→ Distributed Systems
→ Distributed Training
→ LLM Infrastructure
→ Reliability Engineering
→ System Design
```

The long-term objective is to build the systems knowledge and practical portfolio required for roles such as:

- AI Infrastructure Engineer
- ML Systems Engineer
- ML Platform Engineer
- GPU Infrastructure Engineer
- Inference Engineer
- MLOps Engineer
- Distributed Systems Engineer

---

## Repository Status

This repository is actively updated as new labs are completed. Future additions will include Kubernetes observability, GPU infrastructure, production model serving, distributed training, LLM infrastructure, and system-design exercises.
