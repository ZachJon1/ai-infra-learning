# Day 11 — Kubernetes Storage

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Understand ephemeral and persistent storage in Kubernetes.
- Distinguish PVs, PVCs, StorageClasses, and StatefulSets.
- Reason about storage lifecycle independently from Pod lifecycle.

## Core topics
- `emptyDir`
- PersistentVolumes (PV)
- PersistentVolumeClaims (PVC)
- StorageClasses
- Access modes
- StatefulSets

## Key lessons
- A PVC expresses a storage request; a PV represents provisioned storage.
- Deleting a Pod does not necessarily delete its persistent data.
- StatefulSets provide stable identities and predictable storage relationships.

## Commands to retain
```bash
kubectl get pv
kubectl get pvc
kubectl describe pvc <pvc>
kubectl get statefulset
kubectl describe pod <pod>
```

## Interview takeaway
Separate Pod lifecycle from storage lifecycle. A recreated Pod can often reconnect to persistent storage if the PVC and backing volume still exist.
