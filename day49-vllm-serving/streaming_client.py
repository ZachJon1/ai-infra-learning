
import json
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
)

model = "Qwen/Qwen2.5-0.5B-Instruct"

prompts = [
    "Explain GPU memory allocation in six numbered steps.",
    "Describe how Kubernetes schedules GPU workloads.",
    "Explain model serving latency and throughput with examples.",
]

results = []

for i, prompt in enumerate(prompts, start=1):
    print(f"\n--- Request {i} ---")

    start = time.perf_counter()
    first_token_time = None
    content_chunks = 0
    output = []

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=240,
        temperature=0,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta.content or ""

        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()

            content_chunks += 1
            output.append(delta)
            print(delta, end="", flush=True)

    end = time.perf_counter()

    total_latency = end - start
    ttft = (
        first_token_time - start
        if first_token_time is not None
        else None
    )

    result = {
        "request": i,
        "ttft_seconds": round(ttft, 4) if ttft is not None else None,
        "total_latency_seconds": round(total_latency, 4),
        "content_chunks": content_chunks,
    }

    results.append(result)
    print("\n\nMetrics:", result)

with open("streaming_results.json", "w") as f:
    json.dump(results, f, indent=2)
