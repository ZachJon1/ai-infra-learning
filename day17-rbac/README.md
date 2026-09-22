# Day 17 — Kubernetes RBAC, ServiceAccounts & Basic Security

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Understand how Kubernetes identities receive permissions.
- Distinguish ServiceAccounts, Roles, ClusterRoles, and bindings.
- Apply least-privilege thinking to workloads.

## Core topics
- ServiceAccounts
- Roles
- ClusterRoles
- RoleBindings
- ClusterRoleBindings
- Authorization checks
- Least privilege

## Mental model
```text
Identity
   ↓
ServiceAccount / user
   ↓
Binding
   ↓
Role / ClusterRole
   ↓
Allowed API actions
```

## Commands to retain
```bash
kubectl get serviceaccounts
kubectl get roles
kubectl get rolebindings
kubectl auth can-i <verb> <resource>
kubectl auth can-i --as=system:serviceaccount:<namespace>:<sa> <verb> <resource>
```

## Interview takeaway
Authentication identifies the caller; authorization determines what that identity may do. Prefer narrowly scoped permissions instead of broad cluster-wide access.
