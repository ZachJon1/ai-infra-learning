# Day 5 — Linux Networking Fundamentals

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Understand interfaces, IP addressing, routes, DNS, sockets, ports, and CIDR.
- Trace the path between an application and a remote service.
- Diagnose common connectivity failures.

## Core topics
- Network interfaces
- IP addresses and routing
- TCP/UDP ports
- DNS resolution
- Listening sockets
- CIDR notation

## Commands to retain
```bash
ip addr
ip route
ss -tulpn
ping
curl
wget
nslookup
dig
```

## AI-infrastructure relevance
Distributed training, Kubernetes Services, model APIs, storage systems, and GPU clusters all depend on reliable networking.

## Interview takeaway
Separate name-resolution problems, routing problems, port/listener problems, and application-level failures rather than treating every connectivity issue as simply 'the network.'
