# Day 1 — Linux Fundamentals

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Build comfort with the Linux shell and filesystem.
- Understand processes, permissions, environment variables, and common command-line workflows.
- Develop the foundation needed for container, Kubernetes, and server troubleshooting.

## Core topics
- Shell navigation and file operations
- Processes and process inspection
- Permissions and ownership
- Environment variables
- Pipes, redirection, and command composition

## Commands to retain
```bash
pwd
ls -lah
cd
cp
mv
rm
cat
less
grep
find
ps
top
chmod
chown
env
```

## AI-infrastructure relevance
Most AI infrastructure ultimately runs on Linux. Being able to inspect files, processes, permissions, and runtime state quickly is foundational for debugging model servers, containers, GPU nodes, and cluster workloads.

## Interview takeaway
When diagnosing a Linux host, first establish the current system state before changing anything: inspect processes, resources, files, permissions, and logs, then narrow the problem to the appropriate layer.
