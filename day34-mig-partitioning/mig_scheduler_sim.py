
# Day 34: MIG scheduling simulation
# SIMULATED inventory, not real hardware measurements.

from collections import Counter

inventory = Counter({
    "nvidia.com/mig-1g.5gb": 1,
    "nvidia.com/mig-2g.10gb": 1,
    "nvidia.com/mig-3g.20gb": 1,
})

workloads = [
    ("classifier", "nvidia.com/mig-1g.5gb"),
    ("embeddings", "nvidia.com/mig-2g.10gb"),
    ("reranker", "nvidia.com/mig-3g.20gb"),
    ("vision", "nvidia.com/mig-3g.20gb"),
]

available = inventory.copy()

print("=== SIMULATED MIG INVENTORY ===")
for resource, count in inventory.items():
    print(f"{resource}: {count}")

print("\n=== SIMULATED SCHEDULING ===")

for name, requested in workloads:
    if available[requested] > 0:
        available[requested] -= 1
        print(f"{name}: SCHEDULED -> {requested}")
    else:
        print(f"{name}: PENDING -> unavailable {requested}")
        print("  Other free profiles cannot automatically substitute.")

print("\n=== REMAINING FREE DEVICES ===")
for resource, count in available.items():
    print(f"{resource}: {count}")
