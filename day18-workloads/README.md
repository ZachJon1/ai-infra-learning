# Day 18 — Jobs, CronJobs, Init Containers & Workload Lifecycle

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Distinguish long-running services from finite/batch workloads.
- Use Jobs and CronJobs for completion-oriented work.
- Understand init containers and workload lifecycle sequencing.

## Core topics
- Jobs
- CronJobs
- Restart/completion behavior
- Init containers
- Pod lifecycle
- Batch workload patterns

## Mental model
```text
Deployment → keep service running
Job        → run until successful completion
CronJob    → create Jobs on a schedule
Init container → finish setup before app containers start
```

## Commands to retain
```bash
kubectl get jobs
kubectl get cronjobs
kubectl describe job <job>
kubectl logs job/<job>
kubectl create job
```

## Interview takeaway
Choose the controller that matches workload semantics. A batch task should normally model completion rather than being forced into a long-running Deployment pattern.
