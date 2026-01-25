# FedAvg Memory Profiling

Tests for comparing memory usage (RSS - Resident Set Size) with different 
`server_memory_gc_rounds` settings in the FedAvg controller.

RSS is the portion of memory occupied by a process that is held in RAM.

## Quick Reference

```bash
# Basic test
python test_fedavg_memory.py

# With memory arena limit (recommended)
MALLOC_ARENA_MAX=4 python test_fedavg_memory.py

# Detailed profiling with memory_profiler
mprof run test_fedavg_memory.py
mprof plot
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
- `model_size_mb`: Model size in MB (default: 50)

## Sample Output

```
FedAvg Memory Profiling Test
MALLOC_ARENA_MAX: 4
============================================================
Testing: server_memory_gc_rounds=0
Rounds: 10, Clients: 2, Model: ~50MB
============================================================
Initial RSS: 150.0 MB
Final RSS: 280.0 MB
RSS increase: +130.0 MB

============================================================
SUMMARY
============================================================
Setting                         Initial MB     Final MB     Increase
------------------------------------------------------------------
gc_rounds=0 (disabled)             150.0        280.0       +130.0
gc_rounds=5                        152.0        220.0        +68.0
gc_rounds=1                        151.0        185.0        +34.0
------------------------------------------------------------------
gc_rounds=5 vs disabled: 47.7% reduction
gc_rounds=1 vs disabled: 73.8% reduction
```

