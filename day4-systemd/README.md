# Day 4 — systemd, Services & Resource Controls

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Understand how Linux services are started and supervised.
- Inspect service failures and restart behavior.
- Connect service management to cgroups and resource isolation.

## Core topics
- `systemd` units
- Service status and logs
- Restart policies
- cgroups
- CPU and memory controls

## Commands to retain
```bash
systemctl status <service>
systemctl start <service>
systemctl stop <service>
systemctl restart <service>
systemctl enable <service>
journalctl -u <service>
journalctl -u <service> -f
```

## AI-infrastructure relevance
GPU agents, container runtimes, node services, exporters, and model-serving components commonly run as managed system services.

## Interview takeaway
A service being configured correctly does not guarantee it is healthy. Inspect both service state and its logs, and understand whether repeated restarts are a symptom rather than the root cause.
