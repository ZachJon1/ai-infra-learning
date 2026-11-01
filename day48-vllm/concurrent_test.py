import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"

def send_request(i):
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{
            "role": "user",
            "content": f"Explain AI infrastructure concept {i} briefly."
        }],
        "max_tokens": [32, 64, 96][i % 3]
    }

    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)

    elapsed = time.perf_counter() - start
    return i, elapsed, result["usage"]["completion_tokens"]

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(send_request, range(8)))

for i, elapsed, tokens in results:
    print(f"Request {i}: {elapsed:.2f}s, {tokens} output tokens")