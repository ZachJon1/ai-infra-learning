#!/usr/bin/env python3
"""Day 55 Stage 4: compare vLLM shared-prefix vs unique-prompt traffic.

Run identical commands against two separate server starts with prefix caching
ON and OFF. Keep model, hardware, context, max tokens, and concurrency fixed.

This is a small exploratory benchmark, not a production load generator.
"""
import argparse
import csv
import json
import math
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# A deliberately long reusable prefix to expose prefill/prefix-cache behavior.
# Completely original technical text, with no external data dependencies.
SHARED_CONTEXT = (
    "You are reviewing an AI inference cluster. A request enters an HTTP "
    "service, passes through an inference scheduler, and is executed on an "
    "NVIDIA GPU. Investigations should distinguish networking, request "
    "queueing, CPU tokenization, prompt prefill, token decoding, KV cache "
    "occupancy, GPU compute, and output transfer. Record exact symptoms, "
    "timestamps, affected endpoints, deployment versions, and measured "
    "resource counters. A container being alive does not prove it is ready. "
    "A working GPU driver does not guarantee a device plugin registered GPU "
    "resources with Kubernetes. Taints can prevent infrastructure DaemonSets "
    "from scheduling. Inference throughput and tail latency can move in "
    "different directions under higher concurrency. A successful benchmark "
    "should document prompt shape, model configuration, hardware conditions, "
    "sample size, warmup, failed requests, throughput, and latency percentiles. "
) * 2

TOPICS = [
    "GPU memory usage", "readiness probe behavior", "prefix cache reuse",
    "throughput saturation", "tail latency", "temperature and GPU clocks",
    "container startup", "inference request queueing", "KV cache capacity",
    "resource taints and tolerations", "CPU tokenization overhead",
    "CUDA kernel execution", "service networking", "GPU scheduler behavior",
    "model loading", "concurrent decoding",
]


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    r = (len(s) - 1) * p / 100
    a, b = math.floor(r), math.ceil(r)
    return s[a] + (s[b] - s[a]) * (r - a)


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def get_metrics(url):
    try:
        with urllib.request.urlopen(url + '/metrics', timeout=10) as resp:
            data = resp.read().decode('utf-8', 'replace')
    except Exception as exc:
        print(f'Metrics unavailable: {exc}')
        return {}
    out = {}
    for name in ('vllm:prefix_cache_hits', 'vllm:prefix_cache_queries'):
        # Counter samples may be exposed with a _total suffix and model labels.
        pattern = rf'^{re.escape(name)}(?:_total)?(?:\{{[^\n]*?\}})?\s+([-+0-9.eE]+)(?:\s|$)'
        values = [float(x) for x in re.findall(pattern, data, re.MULTILINE)]
        if values:
            out[name] = sum(values)
    return out


def make_prompt(index, mode):
    question = f"Case {index}: In two sentences, describe how to investigate {TOPICS[index % len(TOPICS)]}."
    if mode == 'shared':
        return SHARED_CONTEXT + '\n' + question
    # Changing the start (and much of the long context) limits useful cache reuse.
    chunks = [f'Experiment {index}, step {k}: verify {TOPICS[(index * 3 + k) % len(TOPICS)]}.'
              for k in range(90)]
    return ' '.join(chunks) + '\n' + question


def send(base, model, prompt, max_tokens, timeout, index):
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0,
        'max_tokens': max_tokens,
        'stream': False,
    }
    req = urllib.request.Request(
        base + '/v1/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.load(resp)
        usage = result.get('usage') or {}
        choices = result.get('choices') or []
        if not choices:
            raise ValueError('No choices returned')
        return {
            'index': index, 'success': True,
            'latency_ms': round(1000 * (time.perf_counter() - start), 3),
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'error': '',
        }
    except Exception as exc:
        return {
            'index': index, 'success': False,
            'latency_ms': round(1000 * (time.perf_counter() - start), 3),
            'prompt_tokens': 0, 'completion_tokens': 0,
            'error': f'{type(exc).__name__}: {exc}',
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--mode', choices=['shared', 'unique'], required=True)
    p.add_argument('--cache-label', choices=['on', 'off'], required=True,
                   help='Label only: MUST match your vLLM server configuration')
    p.add_argument('--requests', type=int, default=32)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=64)
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--output-dir', default='stage4_results')
    a = p.parse_args()
    if min(a.requests, a.concurrency, a.max_tokens, a.timeout) < 1:
        p.error('Requests, concurrency, max-tokens, timeout must be >= 1')
    base = a.base_url.rstrip('/')
    model = get_json(base + '/v1/models')['data'][0]['id']
    prompts = [make_prompt(i, a.mode) for i in range(a.requests)]
    # Warm the model and repeated prefix before measuring; not included in results.
    for i in range(3):
        warm = send(base, model, prompts[i % len(prompts)], a.max_tokens, a.timeout, -1 - i)
        if not warm['success']:
            raise RuntimeError('Warmup failed: ' + warm['error'])
    before = get_metrics(base)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        jobs = [ex.submit(send, base, model, prompt, a.max_tokens, a.timeout, i)
                for i, prompt in enumerate(prompts)]
        rows = [job.result() for job in as_completed(jobs)]
    wall = time.perf_counter() - start
    after = get_metrics(base)
    rows.sort(key=lambda r: r['index'])
    valid = [r for r in rows if r['success']]
    latencies = [r['latency_ms'] for r in valid]
    hits_key = 'vllm:prefix_cache_hits'
    queries_key = 'vllm:prefix_cache_queries'
    hits = after.get(hits_key, float('nan')) - before.get(hits_key, float('nan'))
    queries = after.get(queries_key, float('nan')) - before.get(queries_key, float('nan'))
    summary = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'model': model, 'mode': a.mode, 'cache_label': a.cache_label,
        'requests': a.requests, 'concurrency': a.concurrency,
        'max_tokens': a.max_tokens, 'successful': len(valid),
        'failed': len(rows) - len(valid), 'wall_s': round(wall, 3),
        'rps': round(len(valid) / wall, 3),
        'output_tps': round(sum(int(r['completion_tokens']) for r in valid) / wall, 3),
        'mean_ms': round(sum(latencies) / len(latencies), 3) if latencies else None,
        'p50_ms': round(percentile(latencies, 50), 3) if latencies else None,
        'p95_ms': round(percentile(latencies, 95), 3) if latencies else None,
        'p99_ms': round(percentile(latencies, 99), 3) if latencies else None,
        'cache_hit_tokens_delta': round(hits) if not math.isnan(hits) else None,
        'cache_query_tokens_delta': round(queries) if not math.isnan(queries) else None,
        'cache_hit_rate': round(hits / queries, 4) if queries and not math.isnan(queries) else None,
    }
    folder = Path(a.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'{a.mode}_cache_{a.cache_label}_c{a.concurrency}'
    (folder / f'{stem}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (folder / f'{stem}_requests.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f'Saved: {folder}/{stem}_summary.json and _requests.csv')


if __name__ == '__main__':
    main()
