"""Sequential traffic generator used for Day 60 (24 requests, fixed prompt)."""
import json
import time
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8000"
with urlopen(f"{BASE}/v1/models", timeout=10) as resp:
    models = json.load(resp)["data"]
if not models:
    raise SystemExit("No served model found")
model = models[0]["id"]
print("Using model:", model)
for i in range(1, 25):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Explain what GPU utilization means in one sentence."}],
        "max_tokens": 48,
        "temperature": 0,
    }
    req = Request(f"{BASE}/v1/chat/completions", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=90) as resp:
            result = json.load(resp)
        elapsed = time.perf_counter()-start
        tokens = result.get("usage", {}).get("completion_tokens")
        print(f"Request {i:02d}/24: success | client_latency={elapsed:.3f}s | output_tokens={tokens}")
    except Exception as exc:
        print(f"Request {i:02d}/24: failed | {exc}")
    time.sleep(1.5)
print("Workload completed.")
