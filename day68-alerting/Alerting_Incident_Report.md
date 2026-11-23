# Incident Report — Prometheus Alert Simulation Blocked by a Stale File Bind Mount

**Date:** November 23, 2026  
**System:** Local Prometheus/Alertmanager monitoring for a vLLM inference server  
**Incident type:** Lab configuration propagation failure (not a production outage)  
**Outcome:** Resolved; end-to-end notification test completed

## Objective

Simulate a failure of a configured vLLM metrics scrape target, validate Prometheus alert firing, and verify Alertmanager notification delivery and recovery.

## Impact and scope

The initial fault injection did not reach the running Prometheus configuration. As a result, `up{job="vllm"}` stayed at `1`, and no `VLLMServerDown` alert fired. The inference service itself was not stopped. This was an observability test, not a customer-facing outage.

## Detection and investigation

1. The host `prometheus.yml` scrape target was edited from `127.0.0.1:8000` to `127.0.0.1:18000` using `sed -i`.
2. A Prometheus configuration reload was requested, but the query continued to report `instance="127.0.0.1:8000"`, `value="1"`.
3. Configuration comparison showed **host file:** `18000`; **inside Prometheus container:** `8000`.
4. This established that the effective running configuration differed from the edited host file. The behavior was consistent with a Docker **single-file bind mount** retaining the prior file after `sed -i` replaced the host file.

## Resolution

The Prometheus container was recreated:

```bash
docker compose up -d --force-recreate prometheus
```

The container now showed port `18000`, and Prometheus returned `up = 0` for the new target. The `VLLMServerDown` alert entered `firing` state.

## Validated notification and recovery

- **12:40:42, November 23, 2026:** Python receiver logged `Group status: firing`, `VLLMServerDown`, `severity: critical`, HTTP 200.
- **12:47:12:** Receiver logged `Group status: resolved` for the same alert, HTTP 200.
- Original target `127.0.0.1:8000` was restored, Prometheus container recreated, `up = 1`, and the active-alert array was empty.

The webhook timestamps indicate successful delivery, **not** exact detection latency, mean-time-to-recovery, or outage length.

## Root cause

A single-file Docker bind mount exposed an earlier version of `prometheus.yml` inside the existing container after a `sed -i` update replaced the host file. Reloading Prometheus alone did not change that file mount. Recreating the container refreshed the mounted file.

## Corrective and preventive actions

- **Completed:** Recreated Prometheus and confirmed the updated target inside the container.
- **Completed:** Confirmed `up = 0`, a firing alert, a webhook delivery, a resolution notification, `up = 1`, and no active alerts.
- **Proposed:** Mount a configuration directory to tolerate atomic replacements of contained files.
- **Proposed:** Add a standard post-reload check comparing intended and effective scrape targets.
- **Proposed:** Pin image versions and document the notification receiver's role in the failure-injection procedure.
- **Not yet tested:** Latency alert behavior under controlled inference load.

## Reusable reliability lesson

A valid configuration and an HTTP-successful reload do not prove a service adopted intended changes. Verify the actual container-visible configuration and observable runtime behavior. Test alert **delivery** and **resolution**, not just expression evaluation.
