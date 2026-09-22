# Day 16 — Helm Fundamentals

**Roadmap:** AI Infrastructure & ML Systems  
**Status:** Completed

## Objectives
- Package and parameterize Kubernetes applications.
- Understand charts, values, templates, releases, upgrades, and rollbacks.
- Reduce duplicated manifests across environments.

## Core topics
- Helm charts
- `values.yaml`
- Templates
- Releases
- Upgrade
- Rollback
- `upgrade --install`

## Key lessons
- A chart is a package of Kubernetes templates and defaults.
- A release is an installed instance of a chart.
- Rollback creates another release revision; revision numbers continue increasing.
- Runtime controllers such as HPA can change replica counts independently of Helm values.

## Commands to retain
```bash
helm install
helm upgrade
helm upgrade --install
helm list
helm history
helm rollback
helm get values
```

## Interview takeaway
Helm manages desired deployment configuration, but Kubernetes controllers can still modify live state. Always distinguish chart values from current cluster state.
