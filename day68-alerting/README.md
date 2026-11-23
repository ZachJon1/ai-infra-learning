# Day 68 — Production-Style Alerting for a vLLM Inference Service

**Track:** AI Infrastructure & ML Systems — 220-Day Roadmap  
**Phase:** Observability (Day 68 — Alerting)  
**Lab date:** November 23, 2026  
**Status:** **Availability-alert end-to-end test completed**; latency-alert rule configured but **not load-tested**.

## Overview

This lab implements a local, Prometheus-based alerting pipeline for a vLLM server hosting **Qwen2.5-0.5B-Instruct**. Prometheus scrapes the inference server's metrics endpoint and evaluates alert rules; Alertmanager routes notifications to a local Python webhook receiver. A deliberately unavailable *scrape target* is used to prove the full alert lifecycle without shutting down the actual model service.

**Verified outcome:** A simulated scrape outage changed `up` from `1` to `0`, triggered the `VLLMServerDown` alert, delivered a `firing` webhook, and subsequently delivered a `resolved` webhook after the original target was restored. Prometheus again returned `up = 1` and an empty active-alert array.

> **Scope:** This exercise simulates a **metrics scrape failure**. It does **not** demonstrate an actual vLLM process crash, GPU out-of-memory incident, or a validated latency-SLO alert.

## Architecture

```text
           Qwen2.5-0.5B-Instruct on vLLM
             http://127.0.0.1:8000/metrics
                           |
                    Prometheus scrape
                           v
               Prometheus :19090
          [up metric, alert rule evaluation]
                           |
                  firing/resolved alerts
                           v
                Alertmanager :19093
            [group, route, notify]
                           |
                      HTTP POST
                           v
             Python webhook :19094
                [print notification]
```

All listed addresses are localhost endpoints on the **Linux host**. The monitoring containers use `network_mode: host` on Linux; the webhook receiver is a local Python process. The lab intentionally avoids configuring external notification channels or public network exposure.

## Technology and repository layout

- Linux, Docker Compose, Python 3, vLLM, Prometheus, Alertmanager.
- `compose.yaml` — Prometheus and Alertmanager services.
- `prometheus.yml` — scrape targets, evaluation interval, rule file, Alertmanager endpoint.
- `vllm-alerts.yml` — availability and high-p95-latency alert rules.
- `alertmanager.yml` — route to local webhook receiver, including resolution notifications.
- `webhook_receiver.py` — prints received JSON alert status and labels.
- `README.md` — this guide.

## Ports

| Component | Endpoint | Purpose |
|---|---|---|
| vLLM | `127.0.0.1:8000` | Inference API and `/metrics` |
| Prometheus | `127.0.0.1:19090` | Metrics and alerts API/UI |
| Alertmanager | `127.0.0.1:19093` | Alerts management UI/API |
| Python webhook | `127.0.0.1:19094` | Local notification receiver |

## Alert rules

### 1. `VLLMServerDown` — critical

```yaml
- alert: VLLMServerDown
  expr: up{job="vllm"} == 0
  for: 30s
  labels:
    severity: critical
    service: vllm
```

`up == 0` means Prometheus **cannot scrape an existing configured target**. The lab uses a short `for: 30s` duration to make the behavior easy to observe. This expression is not sufficient for a target that disappears completely from service discovery; that requires a separate missing-target strategy.

### 2. `VLLMHighP95Latency` — warning

The latency rule estimates p95 using `histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{job="vllm"}[5m])))`, compares the result with **1 second**, and applies a traffic guard `sum(rate(vllm:e2e_request_latency_seconds_count{job="vllm"}[5m])) > 0.1` with a `for: 5m` condition. These are **demonstration thresholds**, not measured SLOs. The rule was syntax-validated, but **no controlled load test was performed to confirm that it fires**.

## Run the lab

### 1. Confirm vLLM is serving metrics

```bash
curl -fsS http://127.0.0.1:8000/metrics | \
  grep -E '^vllm:e2e_request_latency_seconds_(count|bucket)' | head -n 5
```

Use the existing vLLM launch procedure if the server is not running.

### 2. Start the notification receiver and monitoring services

In terminal A:

```bash
cd ~/ai-infra/day68-alerting
python3 -u webhook_receiver.py
```

In terminal B:

```bash
cd ~/ai-infra/day68-alerting
docker compose config
docker compose up -d
docker compose ps
```

### 3. Validate configuration

```bash
docker compose exec -T prometheus \
  promtool check config /etc/prometheus/prometheus.yml

docker compose exec -T prometheus \
  promtool check rules /etc/prometheus/vllm-alerts.yml

docker compose exec -T alertmanager \
  amtool check-config /etc/alertmanager/alertmanager.yml
```

**Observed result:** Prometheus configuration valid; two alert rules found; Alertmanager configuration valid.

### 4. Confirm a healthy baseline

```bash
curl -fsSG http://127.0.0.1:19090/api/v1/query \
  --data-urlencode 'query=up{job="vllm"}' | python3 -m json.tool
```

**Observed before simulation:** target `127.0.0.1:8000`, value `"1"`.

### 5. Safely simulate a scrape outage

Keep terminal A (webhook receiver) running. The next command **does not stop vLLM**; it changes only the Prometheus lab's scrape destination to a port with no expected listener.

```bash
sed -i 's/127.0.0.1:8000/127.0.0.1:18000/' prometheus.yml

docker compose up -d --force-recreate prometheus
```

Recreation was required in the observed environment because an in-place `sed -i` edit replaced the host file while Docker's existing **single-file bind mount** still exposed the previous file to the container. Confirm effective configuration:

```bash
grep 'targets:' prometheus.yml
docker compose exec -T prometheus \
  grep 'targets:' /etc/prometheus/prometheus.yml
```

Then confirm failed scrape:

```bash
curl -fsSG http://127.0.0.1:19090/api/v1/query \
  --data-urlencode 'query=up{job="vllm",instance="127.0.0.1:18000"}' | \
  python3 -m json.tool
```

**Observed:** `instance="127.0.0.1:18000"`, value `"0"`.

Allow for the configured scrape interval (15 s), evaluation interval (15 s), and 30 s `for` period. Observe:

```bash
curl -fsS http://127.0.0.1:19090/api/v1/alerts | \
  python3 -m json.tool
```

**Observed:** `alertname="VLLMServerDown"`, `severity="critical"`, `state="firing"`.

### 6. Restore metrics collection

```bash
sed -i 's/127.0.0.1:18000/127.0.0.1:8000/' prometheus.yml

docker compose up -d --force-recreate prometheus

curl -fsSG http://127.0.0.1:19090/api/v1/query \
  --data-urlencode 'query=up{job="vllm",instance="127.0.0.1:8000"}' | \
  python3 -m json.tool

curl -fsS http://127.0.0.1:19090/api/v1/alerts | \
  python3 -m json.tool
```

**Observed:** target `127.0.0.1:8000`, value `"1"`; `"alerts": []`.

## Verified results / evidence

Evidence is from the terminal output captured during the lab on November 23, 2026 (local timestamps where shown).

| Check | Actual observation | Status |
|---|---|---|
| Docker Compose | Prometheus and Alertmanager both running | Pass |
| `promtool check config` | Prometheus configuration valid; one rule file | Pass |
| `promtool check rules` | Two valid rules | Pass |
| `amtool check-config` | Alertmanager configuration valid; one receiver | Pass |
| Healthy scrape | `up{job="vllm",instance="127.0.0.1:8000"} = 1` | Pass |
| Simulated bad target | `up{job="vllm",instance="127.0.0.1:18000"} = 0` | Pass |
| Alert evaluation | `VLLMServerDown`, `state: firing` | Pass |
| Webhook firing delivery | `Group status: firing`, HTTP 200 at **12:40:42** | Pass |
| Webhook resolution delivery | `Group status: resolved`, HTTP 200 at **12:47:12** | Pass |
| Recovery | `up = 1`, `alerts: []` | Pass |
| High-p95 latency firing under load | Not tested | Not verified |

Selected receiver output:

```text
=== ALERTMANAGER NOTIFICATION ===
Group status: firing
{'alertname': 'VLLMServerDown', 'severity': 'critical', 'status': 'firing'}
127.0.0.1 - - [23/Nov/2026 12:40:42] "POST / HTTP/1.1" 200 -

=== ALERTMANAGER NOTIFICATION ===
Group status: resolved
{'alertname': 'VLLMServerDown', 'severity': 'critical', 'status': 'resolved'}
127.0.0.1 - - [23/Nov/2026 12:47:12] "POST / HTTP/1.1" 200 -
```

> The time between these webhook messages is **not** a measured outage duration or time to detect: the exact time of the injected failure, restoration, and each scrape/evaluation is not established by those notification timestamps alone.

## Break/fix incident: Docker single-file bind mount

**Symptom:** After changing the host's scrape target from port `8000` to `18000` and calling Prometheus's `/-/reload` endpoint, `up{job="vllm"}` continued to show the original `8000` target with value `1`. No outage alert fired.

**Investigation:** Comparing `grep 'targets:' prometheus.yml` on the host against `docker compose exec -T prometheus grep 'targets:' /etc/prometheus/prometheus.yml` showed different contents. The host file showed `18000`, but the container still showed `8000`.

**Root cause:** The active container was still seeing the previous file through a **single-file bind mount** after the host's `sed -i` replacement. Prometheus reloaded the file available *inside* the container, which was not the newly edited host file.

**Fix:** `docker compose up -d --force-recreate prometheus` refreshed the bind mount to the current host file. The container then showed `18000`; Prometheus returned `up = 0` for the new target; the critical alert fired.

**Prevention:** Compare **effective container configuration** with host configuration when a reload appears ineffective. Prefer mounting a configuration **directory**, not individual files, if atomic host-side file replacement will be used. Validate both syntax and runtime behavior. For production, also pin versions and use controlled deployment/reload procedures.

## Operational lessons

1. **Config validity ≠ active config.** A valid YAML file can differ from the file a running service actually reads.
2. **Metric collection ≠ alert delivery.** Test the full chain from scrape failure to Prometheus firing and Alertmanager's webhook, then resolution.
3. **Avoid false incident claims.** The exercise intentionally broke metrics scraping, not the vLLM process itself.
4. **Alert on actionable symptoms.** GPU utilization alone is not necessarily service degradation. Combine availability, errors, latency, and load when choosing thresholds.
5. **Treat latency thresholds as hypotheses.** Validate histogram series and baseline request distributions before using p95 thresholds for operational decisions.

## Next improvements

- Move configuration files into a mounted `config/` directory; pin Prometheus and Alertmanager image versions.
- Add alerts for a **missing/disappeared target**, and separate the meaning of telemetry failure from service health.
- Use a workload generator to test the p95-latency alert with actual traffic and a documented baseline.
- Add a production-style runbook (what to check, links to dashboards, escalation policy), plus appropriate grouping, silences, and noise reduction.
- Capture reproducible screenshots or sanitized terminal logs, plus a concise demo walkthrough, for the portfolio.

## Interview-ready summary

> I built a local Prometheus/Alertmanager alerting pipeline for a vLLM inference service, with a Python webhook notification receiver. I validated rule syntax, safely injected a scrape-target failure, confirmed the `up` metric changed from 1 to 0, observed the critical alert fire, and captured firing and resolved webhook notifications. During the test I found a stale Docker single-file bind mount after an atomic configuration-file edit, diagnosed it by comparing host and in-container configuration, and fixed it by recreating the Prometheus container. I verified recovery using `up = 1` and no active alerts.
