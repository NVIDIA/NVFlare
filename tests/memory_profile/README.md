# Memory Profiling Tests

This directory contains scripts to test and compare memory usage (RSS - Resident Set Size)
with different `server_memory_gc_rounds` settings across various FL algorithms.

RSS is the portion of memory occupied by a process that is held in RAM. It's the key metric
for monitoring memory usage in long-running federated learning jobs.

## Directory Structure

```
tests/memory_profile/
├── requirements.txt          # Dependencies for all memory tests
├── README.md                 # This file
├── common/                   # Shared utilities and base tests
│   └── test_memory_utils.py  # Tests for cleanup_memory() utility
├── fedavg/                   # FedAvg algorithm tests
│   └── test_fedavg_memory.py # FedAvg memory comparison
├── swarm/                    # Swarm Learning tests (future)
├── scaffold/                 # SCAFFOLD tests (future)
└── cyclic/                   # Cyclic tests (future)
```

## Setup

```bash
cd tests/memory_profile
pip install -r requirements.txt
```

## Running Tests

### Common Utility Tests

```bash
# Test cleanup_memory() effectiveness
python common/test_memory_utils.py
```

### FedAvg Tests

```bash
# Compare memory with different gc_rounds settings
python fedavg/test_fedavg_memory.py

# With MALLOC_ARENA_MAX set (recommended for realistic results)
MALLOC_ARENA_MAX=4 python fedavg/test_fedavg_memory.py

# Detailed profiling with mprof
cd fedavg
MALLOC_ARENA_MAX=4 mprof run test_fedavg_memory.py
mprof plot
```

### Future Algorithm Tests

```bash
# Swarm Learning (when implemented)
python swarm/test_swarm_memory.py

# SCAFFOLD (when implemented)
python scaffold/test_scaffold_memory.py
```

## Adding New Algorithm Tests

1. Create a new directory: `mkdir <algorithm>/`
2. Add `__init__.py` 
3. Create `test_<algorithm>_memory.py` following the pattern in `fedavg/`
4. Key things to test:
   - `server_memory_gc_rounds=0` (disabled)
   - `server_memory_gc_rounds=5` (every 5 rounds)
   - `server_memory_gc_rounds=1` (every round)
   - With/without `MALLOC_ARENA_MAX`

## Expected Results

| Setting | Expected Behavior |
|---------|-------------------|
| `server_memory_gc_rounds=0` | RSS grows continuously |
| `server_memory_gc_rounds=5` | RSS drops every 5 rounds (sawtooth) |
| `server_memory_gc_rounds=1` | RSS stays stable |

With `MALLOC_ARENA_MAX=4`, peak RSS should be lower overall.

## Environment Variables

| Variable | Recommended Value | Description |
|----------|-------------------|-------------|
| `MALLOC_ARENA_MAX` | 4 (server), 2 (client) | Limit glibc memory arenas |
