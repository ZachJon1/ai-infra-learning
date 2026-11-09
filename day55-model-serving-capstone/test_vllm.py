import json
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": (
                "Explain Kubernetes GPU scheduling "
                "in two sentences."
            )
        }
    ],
    "max_tokens": 100,
    "temperature": 0
}

request = urllib.request.Request(
    f"{BASE_URL}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

start = time.perf_counter()

with urllib.request.urlopen(
    request, timeout=120
) as response:
    result = json.load(response)

elapsed = time.perf_counter() - start

print("\nGenerated response:")
print(result["choices"][0]["message"]["content"])

print("\nToken usage:")
print(json.dumps(result.get("usage", {}), indent=2))

print(f"\nEnd-to-end latency: {elapsed:.3f} seconds")
