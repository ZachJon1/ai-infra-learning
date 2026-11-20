# Day 65 observations — empirical results only

These values are transcribed from command outputs in the study session and are not automated benchmark data.

| Test | Client status | Client duration | Gateway duration | Notes |
|---|---:|---:|---:|---|
| Gateway health | 200 | not recorded | 0.58 ms | `INFO`; earlier gateway test |
| Simulated outage | 503 | not recorded | 0.32 ms | Intentional test response; not a real vLLM outage |
| Valid vLLM inference via :8001 | 200 | 80.616 ms | 79.62 ms | `Ready.`; 35 prompt + 3 completion tokens |
| Invalid model via :8001 | 404 | 25.492 ms | 24.69 ms | `NotFoundError`; upstream `WARNING` |
| Misconfigured upstream via :8002 | 502 | 29.794 ms | 28.85 ms | `ConnectError`; configured port 8999 |
| Corrected upstream via :8002 | 200 | 70.697 ms | unverified | `Ready.`; printed log line truncated |

## Kubernetes restart demonstration

- Context: `gpu-lab`, Kubernetes `v1.35.1`.
- Namespace: `ai-infra-logging`; Pod: `logging-demo`; container: `logger`.
- Previous container: `Terminated`, `Reason: Error`, `Exit Code: 1`.
- Pod status after restart: `Running`, `Ready=True`, `Restart Count=1`.
- Current log event: `{"level":"INFO","event":"heartbeat","pod":"logging-demo"}`.
- `kubectl logs --previous`: `{"level":"ERROR","event":"simulated_crash","pod":"logging-demo"}`.
- Data lifecycle: `emptyDir` survives container restarts in the same Pod, not Pod replacement.

## Evidence caveats

- Durations come from single requests; no p50/p95 or statistical confidence claim.
- The 502 was created by controlled wrong-port configuration, not a real backend crash.
- A returned 200 confirms functionality, not correctness of every operational property.
- The final recovery log line's integrity remains to be checked on the actual workstation.
