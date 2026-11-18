#!/usr/bin/env python3
"""Reproducible closed-loop vLLM test used for Day 64-style experiments.

Defaults match the November 18 study run: two workers, 90s, 96 output tokens.
This script reports client-observed latency; it is not a production load generator.
"""
import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"


def run_worker(worker_id, deadline, args):
    completed, failures = [], []
    index = 0
    while time.monotonic() < deadline:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": (
                f"Request {worker_id}-{index}. Explain GPU monitoring, batching, "
                "throughput and latency with practical examples."
            )}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        request = urllib.request.Request(
            args.url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                body = json.load(response)
            completed.append((time.perf_counter() - start, body.get("usage", {}).get("completion_tokens", 0)))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failures.append(str(exc))
        index += 1
    return completed, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.workers < 1 or args.duration <= 0 or args.max_tokens < 1:
        parser.error("workers, duration, max-tokens must be positive")

    started = time.monotonic()
    deadline = started + args.duration
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_worker, wid, deadline, args) for wid in range(args.workers)]
        batches = [f.result() for f in futures]
    elapsed = time.monotonic() - started
    results = [sample for successes, _ in batches for sample in successes]
    errors = [error for _, failures in batches for error in failures]
    latencies = sorted(lat for lat, _ in results)
    tokens = sum(tokens for _, tokens in results)
    print(f"Workers: {args.workers}")
    print(f"Completed requests: {len(results)}")
    print(f"Failed requests: {len(errors)}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Requests/sec: {len(results) / elapsed:.2f}")
    print(f"Output tokens/sec: {tokens / elapsed:.2f}")
    if latencies:
        p95 = latencies[math.ceil(0.95 * len(latencies)) - 1]
        print(f"Mean client latency: {statistics.mean(latencies):.4f}s")
        print(f"P95 client latency: {p95:.4f}s")
    if errors:
        print("Example error:", errors[0])


if __name__ == "__main__":
    main()
