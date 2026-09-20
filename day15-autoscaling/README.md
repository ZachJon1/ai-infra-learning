Goal:
Understand Kubernetes Horizontal Pod Autoscaling.

Setup:
- Minikube
- Metrics Server
- CPU-intensive web application
- CPU request: 200m
- CPU limit: 500m
- HPA target: 50%
- minReplicas: 1
- maxReplicas: 5

Observed behavior:
1 replica
↓
load generated
↓
CPU reached ~445m
↓
HPA detected utilization > target
↓
scaled to 4 replicas
↓
scaled to 5 replicas
↓
ScalingLimited=True because maxReplicas=5

Key lesson:
HPA CPU utilization is calculated relative to CPU requests,
not CPU limits.

AI infrastructure takeaway:
CPU may not accurately represent GPU inference pressure.
Production AI workloads may need GPU utilization, queue depth,
latency, request rate, or workload-specific custom metrics.
