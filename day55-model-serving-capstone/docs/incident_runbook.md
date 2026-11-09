# Failure-injection and recovery runbook

**Safety:** Test only on the isolated local lab. Do not deliberately exhaust GPU memory or forcibly kill the NVIDIA driver; preserve work on the host. Record timestamps and relevant logs for new incidents.

## Incident 1 — Invalid chat API payload

**Failure injected:** `messages` provided as a string instead of array of role/content objects.

**Evidence:** HTTP 400 with `param: body.messages`, validation type `list_type`, and error `Input should be a valid list`.

**Diagnosis:** Client input schema, not GPU or Kubernetes.

**Recovery:** Change `messages` to `[{"role":"user","content":"..."}]`. Verified `/health` HTTP 200 and successful new inference after rejection. No restart needed.

## Incident 2 — vLLM server outage

**Failure injected:** Graceful Ctrl+C in vLLM terminal.

**Evidence:** `curl: (7) Failed to connect` and `ss` had no port 8000 listener. GPU memory fell to ~544 MiB; vLLM EngineCore disappeared from `nvidia-smi` process list.

**Recovery:** Restart exact vLLM command, then check:

```bash
curl -i --max-time 10 http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models | python -m json.tool
python scripts/smoke_test.py
nvidia-smi
```

After recovery, `/health` returned 200, the correct model was listed, inference generated 55 tokens in the original smoke-test output, and EngineCore GPU allocation reappeared (~4,866 MiB process).

## Incident 3 — Model context limit

**Failure injected:** Prompt `"hello " * 3000` with 32 output tokens requested on a model configured with `--max-model-len 2048`.

**Evidence:** HTTP 400 with `param: input_tokens`; error indicated prompt contained at least 2,049 tokens and violated the 2,048-token input context limit even with no generation.

**Diagnosis:** Request context length, **not** CUDA OOM. GPU process remained alive; `/health` returned 200; normal inference returned 55 tokens afterward.

**Recovery:** Reduce/summarize/truncate input or select a model/configuration supporting a larger context if available and capacity permits. Do not treat this as a reason to restart the GPU driver.

## Incident 4 — Kubernetes taint and scheduling

**Failure injected:** Create CPU-only Pod from `k8s/taint-test.yaml` without a matching toleration on the single-node Minikube `gpu-lab` node tainted `dedicated=gpu:NoSchedule`.

**Evidence:** `0/1 Pending`; `FailedScheduling`, untolerated taints, preemption not helpful.

**Recovery:** Apply `k8s/toleration-patch.json` to the Pod and verify:

```bash
kubectl patch pod day55-taint-test --type=json --patch-file k8s/toleration-patch.json
kubectl get pod day55-taint-test -o wide
kubectl describe pod day55-taint-test
kubectl delete pod day55-taint-test
kubectl -n kube-system get daemonset nvidia-device-plugin-daemonset
```

After patch: `Scheduled -> Pulled -> Created -> Started`, `Running 1/1`, node `gpu-lab`. Pod cleaned up. NVIDIA Device Plugin stayed `1/1` and GPU allocatable stayed `1`.

### Kubernetes note

This test Pod requests only 25m CPU and 32Mi memory, not a GPU; it demonstrates admission through a tainted node scheduling rule without competing for GPU allocations. `NoSchedule` taints do not evict already-running Pods. The plugin DaemonSet initially required its own matching toleration, a separate earlier configuration fix.

## Initial triage decision table

| Symptom | First inspection |
|---|---|
| HTTP 400/422 | API JSON, endpoint, model ID, context/token budget |
| Connection refused | Listening process and network binding (`ss`, process logs) |
| HTTP 500/503 | Engine logs, GPU process/VRAM, readiness, queueing |
| Pod Pending | `kubectl describe pod`, taints/tolerations, node selection and resources |
| Pod ImagePullBackOff | Image name, registry/auth, network; distinguish from scheduling |
| Suspected CUDA OOM | CUDA traces, VRAM use, model/KV memory, context, concurrency |
| High latency | p95/p99, input/output lengths, queueing, power/clock/thermal conditions |
