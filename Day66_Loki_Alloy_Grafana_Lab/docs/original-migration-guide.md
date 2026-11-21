# Day 66 Add-on — Migrate vLLM to Docker with Alloy Docker log collection

This is an **overlay for your existing `Day66_Loki_Alloy_Grafana_Lab` directory**. It is not a standalone project. It preserves your existing Loki/Grafana/Prometheus/DCGM services and does **not** replace the original `compose.yaml` or `.env`.

## Architecture

```text
vLLM OpenAI API container (localhost:8000, GPU)
  -> Docker json-file logs (bounded rotation)
  -> Grafana Alloy discovery.docker + loki.source.docker
  -> existing Loki (localhost:13100)
  -> existing Grafana (localhost:13000)
```

## Safety and assumptions

- Your existing native vLLM process was originally PID `37546`, but **verify it is still the right process** before stopping it. PID reuse is possible.
- Stop the native server **only after** pre-pulling and testing the Docker image; port 8000 cannot be occupied by both simultaneously.
- The sample pinned image is `vllm/vllm-openai:v0.11.0`. Check `python -m pip show vllm` and `nvidia-smi` to compare your native version and host driver. You can pin a different verified compatible tag by adding `VLLM_IMAGE=vllm/vllm-openai:v<version>` in `.env`, without revealing its contents.
- Official vLLM image is **large**, so confirm free disk space.
- The Docker socket mounted into Alloy grants **powerful Docker daemon access even with `:ro`**. Suitable for a trusted local training lab only. In production use an access-controlled proxy or Kubernetes-native log discovery.
- Loki is unauthenticated and bound only to loopback. Do not expose it or vLLM to untrusted networks.
- Avoid putting credentials, sensitive data, or prompts in application logs.

## 1. Extract files into your existing working directory

```bash
cd ~/ai-infra/Day66_Loki_Alloy_Grafana_Lab
unzip ~/Downloads/Day66_vLLM_Docker_Migration.zip
# The zip adds compose.vllm.yaml, config.docker.alloy, MIGRATION_README.md
```

Do not overwrite your `.env` or original `compose.yaml`. Copy or rename the add-on README as desired.

## 2. Preflight WITHOUT stopping native vLLM

```bash
nvidia-smi
python -m pip show vllm | grep -E '^Version:'
df -h .
docker compose -f compose.yaml -f compose.vllm.yaml config --quiet
docker compose -f compose.yaml -f compose.vllm.yaml pull vllm
# Basic NVIDIA Container Toolkit check, without loading the model:
docker compose -f compose.yaml -f compose.vllm.yaml run --rm --no-deps --entrypoint nvidia-smi vllm
```

If GPU access, driver compatibility, or image pull fails, **leave the native server running** and troubleshoot first.

## 3. Stop the native server gracefully

```bash
ps -p 37546 -o pid,ppid,comm,args
# Execute ONLY if PID 37546 is still the native vLLM process:
kill -TERM 37546
sleep 5
ss -ltnp | grep ':8000' || echo 'Port 8000 is now free'
```

If the process PID is different, inspect `ps -eo pid,comm,args | grep '[v]llm serve'` and stop only that process. If `8000` is still listening, investigate before starting Docker. If it is in a terminal you control, Ctrl+C is also a graceful stop.

## 4. Start vLLM container and reconfigure Alloy

```bash
docker compose -f compose.yaml -f compose.vllm.yaml up -d --no-deps vllm alloy
docker compose -f compose.yaml -f compose.vllm.yaml ps
# Follow startup logs; Ctrl+C exits the follower, not the container:
docker compose -f compose.yaml -f compose.vllm.yaml logs -f --tail=50 vllm
```

Wait for model loading, then check:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

## 5. Send a real inference request

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":16,"temperature":0}' \
  | python3 -m json.tool
```

## 6. Query **real** container stdout/stderr in Loki

Grafana Explore > Loki-Day66:

```logql
{app="vllm", source="docker"}
```

Or CLI:

```bash
curl -fsSG 'http://127.0.0.1:13100/loki/api/v1/query_range' \
  --data-urlencode 'query={app="vllm",source="docker"}' \
  --data-urlencode 'since=1h' | python3 -m json.tool
```

If empty:

```bash
docker compose -f compose.yaml -f compose.vllm.yaml logs --tail=80 alloy
docker compose -f compose.yaml -f compose.vllm.yaml logs --tail=80 vllm
```

Check that Alloy can access `/var/run/docker.sock`, and that the container is named `day66-vllm`. Queries match Docker logs, not the old `logs/vllm.log` file.

## 7. Roll back if needed

```bash
docker compose -f compose.yaml -f compose.vllm.yaml stop vllm
# Run the same known-working original native vLLM command (with its venv active):
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype float16 \
  --max-model-len 2048 --gpu-memory-utilization 0.65
```

If you want Alloy to return to the original file-only config:

```bash
docker compose -f compose.yaml up -d --no-deps --force-recreate alloy
```

Do **not** use `docker compose down -v`; that deletes local Loki and Grafana data volumes.

## Official references

- https://docs.vllm.ai/en/latest/deployment/docker/
- https://docs.docker.com/compose/how-tos/gpu-support/
- https://grafana.com/docs/alloy/latest/monitor/monitor-docker-containers/
