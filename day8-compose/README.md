# Day 8 — Docker Compose

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Run multiple dependent services as one application stack.
- Configure service-to-service networking, volumes, environment variables, and health checks.
- Understand dependency and readiness behavior in multi-service systems.

## Core topics
- Compose services
- Networks
- Volumes
- Environment configuration
- Health checks
- Multi-container application lifecycle

## Example architecture
```text
Application
    |
    +--> API / worker
    |
    +--> PostgreSQL
```

## Commands to retain
```bash
docker compose up
docker compose up -d
docker compose ps
docker compose logs
docker compose down
```

## AI-infrastructure relevance
Compose is a useful bridge between single-container development and orchestration systems such as Kubernetes.

## Interview takeaway
A container being started does not necessarily mean its application is ready. Health checks and dependency behavior matter in multi-service systems.
