#!/usr/bin/env python3
"""Day 55 | vLLM fixed-concurrency, non-streaming end-to-end benchmark.

Python standard library only. Measures client-side completed-request E2E latency,
request throughput and generation throughput. Does NOT measure TTFT or TPOT.

Usage:
    python day55_benchmark_vllm.py --concurrency 1 2 --requests-per-level 20
    python day55_benchmark_vllm.py --concurrency 1 2 4 8 16 --requests-per-level 50
"""
import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROMPT = "Explain Kubernetes GPU scheduling in two sentences."


def fetch_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def percentile(values, p):
    """Linear interpolation at the fractional (n-1)*p rank."""
    v = sorted(values)
    if not v:
        return None
    x = (len(v) - 1) * p / 100.0
    low, high = math.floor(x), math.ceil(x)
    return v[low] + (v[high] - v[low]) * (x - low)


def submit(base, model, timeout, max_tokens, index, concurrency, warmup=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    output = {
        "concurrency": concurrency, "request_id": index,
        "warmup": warmup, "start_utc": started_at,
        "latency_ms": "", "prompt_tokens": "", "completion_tokens": "",
        "finish_reason": "", "success": False, "error": "",
    }
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
        usage = result.get("usage") or {}
        choices = result.get("choices") or []
        output.update(
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            prompt_tokens=usage.get("prompt_tokens", ""),
            completion_tokens=usage.get("completion_tokens", ""),
            finish_reason=(choices[0].get("finish_reason", "") if choices else ""),
            success=bool(choices),
            error="" if choices else "No choices in response",
        )
    except urllib.error.HTTPError as e:
        output.update(latency_ms=round((time.perf_counter() - start) * 1000, 3),
                      error=f"HTTP {e.code}: {e.read(300).decode('utf-8', 'replace')}")
    except Exception as e:
        output.update(latency_ms=round((time.perf_counter() - start) * 1000, 3),
                      error=f"{type(e).__name__}: {e}")
    return output


def write_csv(path, rows, headers):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--requests-per-level", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()
    if any(c < 1 for c in args.concurrency) or args.requests_per_level < 1 or args.warmup < 0 or args.max_tokens < 1:
        parser.error("concurrency, requests-per-level and max-tokens must be positive; warmup cannot be negative")
    base = args.base_url.rstrip("/")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        model = fetch_json(base + "/v1/models", 10)["data"][0]["id"]
    except Exception as e:
        sys.exit(f"Unable to discover running vLLM model at {base}: {e}\nStart the vLLM server first.")

    print(f"Server: {base} | Model: {model}")
    print(f"Prompt: {PROMPT!r} | max_tokens={args.max_tokens} | stream=False")
    print("Note: identical prompts exercise warm/prefix-cached behavior; not a diverse production workload.")

    config = {"base_url": base, "model": model, "prompt": PROMPT,
              "max_tokens": args.max_tokens, "concurrency": args.concurrency,
              "requests_per_level": args.requests_per_level, "warmup": args.warmup,
              "timeout": args.timeout,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "latency_metric": "client-side end-to-end (non-streaming)",
              "percentile_method": "linear interpolation of sorted samples at (n-1)*p/100"}
    (outdir / "configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for i in range(args.warmup):
        warm = submit(base, model, args.timeout, args.max_tokens, i, 1, True)
        if not warm["success"]:
            sys.exit(f"Warmup failed: {warm['error']}")
    print(f"Warmup: {args.warmup} successful requests (excluded from measurements)")

    raw_rows = []
    summary = []
    for c in args.concurrency:
        print(f"\nRunning concurrency={c} with {args.requests_per_level} requests...")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as executor:
            futures = [executor.submit(submit, base, model, args.timeout,
                                       args.max_tokens, i, c) for i in range(args.requests_per_level)]
            run_rows = [f.result() for f in as_completed(futures)]
        wall_s = time.perf_counter() - t0
        run_rows.sort(key=lambda r: r["request_id"])
        raw_rows.extend(run_rows)
        good = [r for r in run_rows if r["success"]]
        latencies = [r["latency_ms"] for r in good]
        output_tokens = sum(int(r["completion_tokens"] or 0) for r in good)
        input_tokens = sum(int(r["prompt_tokens"] or 0) for r in good)
        data = {
            "concurrency": c, "requests_total": len(run_rows),
            "success": len(good), "failed": len(run_rows) - len(good),
            "wall_s": round(wall_s, 3),
            "requests_per_s": round(len(good) / wall_s, 3),
            "output_tokens_per_s": round(output_tokens / wall_s, 3),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "mean_e2e_ms": round(sum(latencies) / len(latencies), 3) if latencies else "",
            "p50_e2e_ms": round(percentile(latencies, 50), 3) if latencies else "",
            "p95_e2e_ms": round(percentile(latencies, 95), 3) if latencies else "",
            "p99_e2e_ms": round(percentile(latencies, 99), 3) if latencies else "",
        }
        summary.append(data)
        print(f"Success: {data['success']}/{data['requests_total']} | "
              f"Requests/s: {data['requests_per_s']} | Output tokens/s: {data['output_tokens_per_s']}")
        print(f"E2E latency ms: p50={data['p50_e2e_ms']} p95={data['p95_e2e_ms']} p99={data['p99_e2e_ms']}")
        if data["failed"]:
            for row in run_rows:
                if not row["success"]:
                    print(f"  Failure example: {row['error']}")
                    break
    write_csv(outdir / "requests.csv", raw_rows,
              ["concurrency", "request_id", "warmup", "start_utc", "latency_ms", "prompt_tokens",
               "completion_tokens", "finish_reason", "success", "error"])
    write_csv(outdir / "summary.csv", summary,
              ["concurrency", "requests_total", "success", "failed", "wall_s", "requests_per_s",
               "output_tokens_per_s", "input_tokens", "output_tokens", "mean_e2e_ms",
               "p50_e2e_ms", "p95_e2e_ms", "p99_e2e_ms"])
    print(f"\nSaved: {outdir / 'requests.csv'}, {outdir / 'summary.csv'}, {outdir / 'configuration.json'}")
    if any(r["failed"] for r in summary):
        print("Warning: Failed requests occurred; do not treat successful-only latency percentiles as complete service performance.")


if __name__ == "__main__":
    main()
