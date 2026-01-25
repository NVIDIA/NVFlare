# FedAvg Memory Profiling

Tests for comparing memory usage (RSS - Resident Set Size) with different 
`server_memory_gc_rounds` settings in the FedAvg controller.

RSS is the portion of memory occupied by a process that is held in RAM.

## Quick Reference

```bash
# Basic test
python test_fedavg_memory.py

# With memory arena limit (recommended for realistic results)
MALLOC_ARENA_MAX=4 python test_fedavg_memory.py

# Detailed profiling with memory_profiler
MALLOC_ARENA_MAX=4 mprof run test_fedavg_memory.py

# View the plot (requires GUI display)
mprof plot

# Or save to file (for headless environments)
mprof plot -o memory_profile.png
```

## What It Tests

| Setting | Expected Behavior |
|---------|-------------------|
| `server_memory_gc_rounds=0` | RSS grows continuously (baseline) |
| `server_memory_gc_rounds=5` | RSS drops every 5 rounds |
| `server_memory_gc_rounds=1` | RSS stays stable (most aggressive) |

## Configuration

Edit `test_fedavg_memory.py` to adjust:

- `num_rounds`: Number of FL rounds (default: 10)
- `num_clients`: Number of simulated clients (default: 2)
- `model_size_mb`: Model size in MB (default: 100)

## Sample Output

```
FedAvg Memory Profiling Test
MALLOC_ARENA_MAX: 4
============================================================
Testing: server_memory_gc_rounds=0
Rounds: 10, Clients: 2, Model: ~100MB
============================================================
Initial RSS: 150.0 MB (before model creation)
Final RSS: 650.0 MB
RSS increase: +500.0 MB

============================================================
SUMMARY
============================================================
Setting                         Initial MB     Final MB     Increase
------------------------------------------------------------------
gc_rounds=0 (disabled)             150.0        650.0       +500.0
gc_rounds=5                        152.0        480.0       +328.0
gc_rounds=1                        151.0        350.0       +199.0
------------------------------------------------------------------
gc_rounds=5 vs disabled: 34.4% reduction
gc_rounds=1 vs disabled: 60.2% reduction
```

**Note:** Initial RSS is measured before model creation. RSS growth beyond
baseline indicates memory fragmentation that cleanup helps reduce.

