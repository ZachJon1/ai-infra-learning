# System architecture and scope

## Actual execution layers

**Host Linux inference:** Ubuntu 24.04 -> host-local vLLM 0.33.0 -> Qwen/Qwen2.5-0.5B-Instruct -> NVIDIA RTX 3070. The HTTP API was bound to `127.0.0.1:8000`; tested `/health`, `/v1/models`, `/v1/chat/completions`, and `/metrics`.

**Separate Docker demonstration:** The earlier `ai-inference:gpu` image ran a FastAPI demo linear model, exposed via host port 8002 while active. Its `/health`, `/ready`, and `/predict` endpoints were tested. The container was then stopped to free GPU VRAM for vLLM.

**Separate Kubernetes lab:** A Minikube Docker-driver cluster `gpu-lab`, with Kubernetes 1.35.1, GPU advertised via the NVIDIA Device Plugin DaemonSet. A custom `dedicated=gpu:NoSchedule` node taint initially blocked the plugin until a matching toleration was added. In Stage 5 a separate CPU-only BusyBox Pod reproduced and resolved an untolerated-taint scheduling failure.

## Serving path

```text
Python/curl client
  -> loopback HTTP (:8000)
  -> Uvicorn/vLLM API process
  -> vLLM scheduler/engine
  -> Qwen model / GPU compute + KV cache
  -> OpenAI-compatible JSON response
```

## Non-goals / not demonstrated

- No Kubernetes Deployment or Service for vLLM.
- No node-to-node distributed inference, autoscaling, or high-availability failover.
- No public API exposure, external auth, load balancer, TLS, or SLO-based alerting.
- Kubernetes GPU allocatable is a scheduler resource quantity, not a measurement of host GPU idleness.

## GPU constraint

`nvidia-smi` showed **55 W** current power limit vs **125 W** driver-reported default, GPU 100% utilization during portions of sustained workloads, and sampled temperatures up to 80°C in the cache experiment. The constraint might influence measured throughput, but no causal attribution or confirmed throttling event was established. Power limits were not changed.
