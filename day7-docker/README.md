# Day 7 — Docker Fundamentals

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Understand images, containers, ports, networks, volumes, and container lifecycle.
- Package a reproducible application environment.
- Connect containers to later Kubernetes concepts.

## Core topics
- Images vs containers
- Dockerfiles
- Port publishing
- Volumes
- Container networking
- GPU-enabled containers
- Container lifecycle

## Commands to retain
```bash
docker build
docker run
docker ps
docker logs
docker exec
docker inspect
docker images
docker stop
docker rm
docker volume ls
docker network ls
```

## AI-infrastructure relevance
Production inference and training workloads are commonly distributed as container images. Container literacy is therefore a prerequisite for Kubernetes-based AI infrastructure.

## Interview takeaway
An image is the immutable package/template; a container is a running instance of that image with runtime state, networking, and optional persistent mounts.
