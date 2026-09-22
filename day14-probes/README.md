# Day 14 — Kubernetes Health Probes

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Distinguish liveness, readiness, and startup probes.
- Understand how probe failures affect traffic and container lifecycle.
- Avoid probes that create unnecessary restarts.

## Core topics
- Liveness probes
- Readiness probes
- Startup probes
- HTTP, TCP, and exec probes

## Mental model
```text
startup  → Can the application finish starting?
readiness → Should this Pod receive traffic?
liveness → Is the running container unhealthy enough to restart?
```

## Key lessons
- Readiness failure removes a Pod from normal Service traffic without necessarily restarting it.
- Liveness failure can restart the container.
- Startup probes protect slow-starting applications from premature liveness/readiness evaluation.

## Interview takeaway
Choose probes based on the failure response you actually want. External dependency failure is often a poor liveness signal because restarting the application may not fix the dependency.
