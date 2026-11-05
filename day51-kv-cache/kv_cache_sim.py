
import math

# Configuration
BLOCK_SIZE = 16
FIXED_CAPACITY = 128

# Each request currently has this many cached tokens
requests = {
    "A": 35,
    "B": 100,
    "C": 16,
    "D": 48,
}

def analyze(requests):
    total_used = sum(requests.values())
    fixed_total = len(requests) * FIXED_CAPACITY
    paged_total = 0

    print(f"{'Request':<10} {'Tokens':<10} {'Blocks':<10} {'Allocated Slots'}")

    for name, tokens in requests.items():
        blocks = math.ceil(tokens / BLOCK_SIZE)
        allocated = blocks * BLOCK_SIZE
        paged_total += allocated
        print(f"{name:<10} {tokens:<10} {blocks:<10} {allocated}")

    print("\n--- Memory Allocation Summary ---")
    print(f"Actual tokens:       {total_used}")
    print(f"Fixed allocation:    {fixed_total} slots")
    print(f"Paged allocation:    {paged_total} slots")
    print(f"Fixed waste:         {fixed_total - total_used} slots")
    print(f"Paged waste:         {paged_total - total_used} slots")
    print(f"Paged efficiency:    {total_used / paged_total:.2%}")

print("\n=== After request B completes ===")
requests["B"] = 120
# requests.pop("B")
analyze(requests)


