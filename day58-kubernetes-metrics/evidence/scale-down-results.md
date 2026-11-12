# Scale-down evidence — Day 58

**Status:** Verified from user-provided terminal output. Final namespace cleanup not yet confirmed.

## Observed behavior

- Initial HPA replica count: **1**.
- After HTTP load: **4 current / 4 desired**; `SuccessfulRescale` event with `New size: 4`.
- Load generator was deleted, and replicas initially remained at 4.
- Following deletion: one reading `cpu: 78%/50%`; later `40% (80m)/50%`, with `AbleToScale=True, ScaleDownStabilized`.
- Final output: `0% (1m) / 50%`, `1 current / 1 desired`.
- Final running Pod: `php-apache-7985cc77f8-7hxsk`, `1/1 Running`, `0` restarts.

## Final HPA evidence excerpt

```text
Metrics: resource cpu on pods (as a percentage of request): 0% (1m) / 50%
Min replicas: 1
Max replicas: 4
Deployment pods: 1 current / 1 desired
Conditions:
  AbleToScale     True  ReadyForNewScale
  ScalingActive   True  ValidMetricFound
  ScalingLimited  True  TooFewReplicas
Events:
  Normal SuccessfulRescale  20m    New size: 4; reason: cpu resource utilization (percentage of request) above target
  Normal SuccessfulRescale  9m24s  New size: 1; reason: All metrics below target
```

**Interpretation:** The HPA first expanded the Deployment to four replicas under CPU load, then reduced it to its minimum of one after demand dropped. `TooFewReplicas` is a normal lower-bound condition at very low CPU usage. The 0% utilization is displayed rounding, not proof of zero CPU use. Relative event ages are relative to the final `kubectl describe` invocation and do not establish the exact time elapsed since load-generator deletion.

## Cleanup verification

Not yet provided. After preserving any local evidence, run:

```bash
kubectl delete namespace day58 --ignore-not-found
kubectl get namespace day58
```

Expected result after deletion completes: the namespace is not found.
