from pathlib import Path
import csv, statistics
with (Path(__file__).parent/'client_latencies.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
for run in ('1','2'):
    l=[float(r['client_latency_seconds']) for r in rows if r['run']==run]
    print(f"Run {run}: n={len(l)}, mean={statistics.mean(l)*1000:.1f}ms, min={min(l)*1000:.0f}ms, max={max(l)*1000:.0f}ms")
print(f"All requests: {sum(r['status']=='success' for r in rows)}/{len(rows)} successful")
