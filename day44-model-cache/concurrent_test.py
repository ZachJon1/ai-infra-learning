import json
import time

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from urllib.request import Request, urlopen

N = 10
barrier = Barrier(N)

def send_request(i):
    barrier.wait()

    request = Request(
        "http://127.0.0.1:8000/predict",
        data=b'{"scale":1.0}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urlopen(request, timeout=30) as response:
        result = json.load(response)

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "request": i,
        "total_ms": round(elapsed, 2),
        "loading_wait_ms": result["loading_wait_ms"],
        "load_count": result["load_count"],
    }


with ThreadPoolExecutor(max_workers=N) as pool:
    results = list(pool.map(send_request, range(N)))

for result in sorted(results, key=lambda x: x["request"]):
    print(result)

with urlopen("http://127.0.0.1:8000/status") as response:
    status = json.load(response)

print("\nFinal model status:")
print(json.dumps(status, indent=2))
