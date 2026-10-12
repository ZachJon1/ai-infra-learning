
import re
from pathlib import Path

def get_rate(filename):
    text = Path(filename).read_text()
    match = re.search(r"iterations/s = ([0-9.]+)", text)
    if not match:
        raise ValueError(f"No throughput found in {filename}")
    return float(match.group(1))

solo = get_rate("solo.log")
a = get_rate("model-a.log")
b = get_rate("model-b.log")

shared_total = a + b

print(f"Solo throughput: {solo:.2f} iterations/s")
print(f"Shared A: {a:.2f} iterations/s")
print(f"Shared B: {b:.2f} iterations/s")
print(f"Shared total: {shared_total:.2f} iterations/s")

print(f"\nA vs solo: {100 * a / solo:.1f}%")
print(f"B vs solo: {100 * b / solo:.1f}%")
print(f"Aggregate shared vs solo: {100 * shared_total / solo:.1f}%")
