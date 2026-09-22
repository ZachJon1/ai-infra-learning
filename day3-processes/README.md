# Day 3 — Processes, Signals & Job Control

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Understand how Linux creates, tracks, and terminates processes.
- Use signals and job-control tools safely.
- Recognize zombie and orphan process behavior.

## Core topics
- PIDs and parent/child relationships
- Foreground and background jobs
- Signals
- `nice` / priority
- `nohup`
- Zombie processes

## Commands to retain
```bash
ps aux
ps -ef
pgrep
kill
kill -TERM <pid>
kill -KILL <pid>
jobs
bg
fg
nohup <command> &
nice
renice
```

## AI-infrastructure relevance
Long-running training and inference jobs are ordinary operating-system processes underneath higher-level orchestration. Process-level reasoning remains useful even when Kubernetes or Slurm is involved.

## Interview takeaway
Prefer graceful termination (`SIGTERM`) before forced termination (`SIGKILL`), and distinguish a process that is consuming resources from a zombie, which has exited but still has an unreaped process-table entry.
