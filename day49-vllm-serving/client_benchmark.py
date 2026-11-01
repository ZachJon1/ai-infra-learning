
import json
import statistics
import time
from openai import OpenAI

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
    timeout=90.0,
)

prompt = "Explain GPU memory management in three short sentences."
results = []

for i in range(6):
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=160,
        temperature=0,
    )

    elapsed = time.perf_counter() - start
    tokens = response.usage.completion_tokens

    result = {
        "run": i,
        "latency_seconds": round(elapsed, 4),
        "prompt_tokens": response.usage.prompt_tokens,
        "output_tokens": tokens,
        "effective_output_tokens_per_second": round(
            tokens / elapsed, 2
        ),
        "finish_reason": response.choices[0].finish_reason,
    }

    if i == 0:
        print("Warmup response:")
        print(response.choices[0].message.content)
    else:
        results.append(result)
        print(result)

latencies = [r["latency_seconds"] for r in results]

print("\nMedian latency:", round(statistics.median(latencies), 4))
print("Mean latency:", round(statistics.mean(latencies), 4))

with open("day49_baseline.json", "w") as f:
    json.dump(results, f, indent=2)
