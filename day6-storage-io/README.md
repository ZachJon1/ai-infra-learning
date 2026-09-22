# Day 6 — Storage & I/O Fundamentals

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Understand filesystems, mounts, inodes, and disk usage.
- Identify storage and I/O bottlenecks.
- Connect Linux storage concepts to container and Kubernetes storage.

## Core topics
- Filesystems and mount points
- Block devices
- Disk usage
- Inodes
- I/O utilization and bottlenecks

## Commands to retain
```bash
df -h
df -i
du -sh
lsblk
mount
findmnt
iostat
```

## AI-infrastructure relevance
Model checkpoints, datasets, container layers, logs, and training pipelines can all become storage bottlenecks. Capacity and I/O performance are separate concerns.

## Interview takeaway
A filesystem can fail because it is out of bytes, out of inodes, unavailable, mounted incorrectly, or too slow. Diagnose which dimension is actually constrained.
