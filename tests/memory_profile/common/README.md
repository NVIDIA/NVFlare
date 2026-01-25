# Common Memory Utils Tests

Tests for the `nvflare.fuel.utils.memory_utils` module.

These tests measure RSS (Resident Set Size) - the portion of memory occupied by a 
process that is held in RAM.

## Quick Reference

```bash
# Run all tests
python test_memory_utils.py
```

## What It Tests

1. **cleanup_memory() effectiveness**
   - Allocates ~100MB, deletes it, then runs cleanup
   - Measures if RSS decreases after `cleanup_memory()`

2. **Repeated allocations**
   - Simulates 10 rounds of allocate/deallocate
   - Compares RSS growth with vs without cleanup

## Platform Notes

| Platform | `malloc_trim()` | Expected Behavior |
|----------|-----------------|-------------------|
| Linux (glibc) | ✅ Available | RSS should decrease after cleanup |
| macOS | ❌ Not available | `gc.collect()` only, less RSS reduction |
| Alpine Linux (musl) | ❌ Not available | `gc.collect()` only |

## Sample Output

```
Testing memory cleanup effectiveness
Platform: linux
malloc_trim available: True

1. Baseline RSS: 85.0 MB
2. After allocating 100MB: 185.0 MB (+100.0)
3. After del (no cleanup): 185.0 MB
4. After cleanup_memory(): 92.0 MB

Summary:
  Memory allocated: 100.0 MB
  After del only: 100.0 MB remaining
  After cleanup: 7.0 MB remaining
  Cleanup recovered: 93.0 MB
```

