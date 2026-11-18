
import argparse
import json
import math
import re
import statistics
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROM_METRIC = re.compile(
    r'^vllm:(request_success_total|prompt_tokens_total|'
    r'generation_tokens_total|e2e_request_latency_seconds_count)'
    r'(?:\{([^}]*)\})?\s+([0-9.eE+-]+)'
)


def get_counters():
    """Read cumulative counters from the vLLM metrics endpoint."""
    with urllib.request.urlopen(
        BASE_URL + "/metrics", timeout=10
    ) as response:
        body = response.read().decode()

    counters = defaultdict(float)

    for line in body.splitlines():
        match = PROM_METRIC.match(line)
        if not match:
            continue

        metric, labels, value = match.groups()

        if metric == "request_success_total":
            reason = re.search(
                r'finished_reason="([^"]+)"', labels or ""
            )
            suffix = reason.group(1) if reason else "unknown"
            metric = f"{metric}:{suffix}"

        counters[metric] += float(value)

    return dict(counters)


def send_request(request_id, max_tokens):
    """Send one non-streaming inference request."""
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain one benefit of GPU inference "
                    "in one short sentence."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request, timeout=60
        ) as response:
            data = json.loads(response.read())

        latency = time.perf_counter() - start
        usage = data.get("usage") or {}

        return {
            "id": request_id,
            "success": True,
            "latency": latency,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "finish_reason": data["choices"][0]["finish_reason"],
        }

    except Exception as exc:
        return {
            "id": request_id,
            "success": False,
            "latency": time.perf_counter() - start,
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests and concurrency must be positive")

    print("\n=== Day 63: vLLM Throughput Benchmark ===")
    print(f"Requests: {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max output tokens: {args.max_tokens}")

    before = get_counters()

    start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(send_request, i, args.max_tokens)
            for i in range(args.requests)
        ]

        results = [future.result() for future in as_completed(futures)]

    elapsed = time.perf_counter() - start
    after = get_counters()

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = sorted(r["latency"] for r in successful)
    output_tokens = sum(r["output_tokens"] for r in successful)
    prompt_tokens = sum(r["prompt_tokens"] for r in successful)

    rps = len(successful) / elapsed
    output_tps = output_tokens / elapsed

    summary = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_seconds": elapsed,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "rps": rps,
        "output_tps": output_tps,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "mean_latency_s": (
            statistics.mean(latencies) if latencies else None
        ),
        "p95_latency_s": (
            latencies[math.ceil(0.95 * len(latencies)) - 1]
            if latencies else None
        ),
        "counter_deltas": {
            key: after.get(key, 0) - before.get(key, 0)
            for key in set(before) | set(after)
        },
        "results": results,
    }

    print("\n--- Client Measurements ---")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"RPS: {rps:.2f}")
    print(f"Output TPS: {output_tps:.2f}")

    if latencies:
        print(f"Mean latency: {summary['mean_latency_s']:.3f} s")
        print(f"p95 latency: {summary['p95_latency_s']:.3f} s")

    print("\n--- Server Counter Changes ---")
    for key, value in sorted(summary["counter_deltas"].items()):
        print(f"{key}: +{value:g}")

    for failure in failed[:5]:
        print(f"Request {failure['id']} failed: {failure['error']}")

    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"results/concurrency_{args.concurrency}_{timestamp}.json"
    )

    with open(filename, "w") as file:
        json.dump(summary, file, indent=2)

    print(f"\nSaved: {filename}")


if __name__ == "__main__":
    main()
