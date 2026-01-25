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
- `model_size_mb`: Model size in MB (default: 500)

## Sample Output

```
FedAvg Memory Profiling Test
MALLOC_ARENA_MAX: 4
============================================================
Testing: server_memory_gc_rounds=0
Rounds: 10, Clients: 2, Model: ~500MB
============================================================
Initial RSS: 150.0 MB (before model creation)
Final RSS: 2500.0 MB
RSS increase: +2350.0 MB

============================================================
SUMMARY
============================================================
Setting                         Initial MB     Final MB     Increase
------------------------------------------------------------------
gc_rounds=0 (disabled)             150.0       2500.0      +2350.0
gc_rounds=5                        152.0       1800.0      +1648.0
gc_rounds=1                        151.0       1200.0      +1049.0
------------------------------------------------------------------
gc_rounds=5 vs disabled: 29.9% reduction
gc_rounds=1 vs disabled: 55.4% reduction
```

**Note:** Initial RSS is measured before model creation. The 500MB model plus
NVFlare overhead accounts for the baseline memory. RSS growth beyond that
indicates memory fragmentation that cleanup helps reduce.

