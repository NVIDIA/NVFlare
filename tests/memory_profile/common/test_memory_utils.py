#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test memory_utils cleanup effectiveness.

This script tests whether cleanup_memory() actually reduces RSS.

Usage:
    python test_memory_utils.py
"""

import gc
import sys

import psutil


def get_rss_mb() -> float:
    """Get current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def allocate_memory(size_mb: int = 100) -> list:
    """Allocate approximately size_mb MB of memory.

    Args:
        size_mb: Amount of memory to allocate in MB.

    Returns:
        List containing allocated data.
    """
    # Each float is 8 bytes
    num_floats = (size_mb * 1024 * 1024) // 8
    return [0.0] * num_floats


def test_cleanup_effectiveness():
    """Test that cleanup_memory reduces RSS after deallocation."""
    from nvflare.fuel.utils.memory_utils import cleanup_memory, try_malloc_trim

    print("Testing memory cleanup effectiveness")
    print(f"Platform: {sys.platform}")
    print(f"malloc_trim available: {try_malloc_trim() is not None}")
    print()

    # Baseline
    gc.collect()
    baseline_rss = get_rss_mb()
    print(f"1. Baseline RSS: {baseline_rss:.1f} MB")

    # Allocate memory
    data = allocate_memory(100)
    after_alloc_rss = get_rss_mb()
    print(f"2. After allocating 100MB: {after_alloc_rss:.1f} MB (+{after_alloc_rss - baseline_rss:.1f})")

    # Delete reference but don't cleanup
    del data
    after_del_rss = get_rss_mb()
    print(f"3. After del (no cleanup): {after_del_rss:.1f} MB")

    # Run cleanup
    cleanup_memory()
    after_cleanup_rss = get_rss_mb()
    print(f"4. After cleanup_memory(): {after_cleanup_rss:.1f} MB")

    # Summary
    print()
    print("Summary:")
    print(f"  Memory allocated: {after_alloc_rss - baseline_rss:.1f} MB")
    print(f"  After del only: {after_del_rss - baseline_rss:.1f} MB remaining")
    print(f"  After cleanup: {after_cleanup_rss - baseline_rss:.1f} MB remaining")

    recovered = after_del_rss - after_cleanup_rss
    if recovered > 0:
        print(f"  Cleanup recovered: {recovered:.1f} MB")
    else:
        print("  Note: cleanup_memory may not show immediate RSS reduction on all platforms")


def test_repeated_allocations():
    """Test that cleanup prevents RSS growth over repeated allocations."""
    from nvflare.fuel.utils.memory_utils import cleanup_memory

    print("\nTesting repeated allocations")
    print("=" * 50)

    iterations = 10
    alloc_size_mb = 50

    # Test 1: Without cleanup
    gc.collect()
    initial_rss = get_rss_mb()
    print(f"\nWithout cleanup (simulating {iterations} rounds):")

    for i in range(iterations):
        data = allocate_memory(alloc_size_mb)
        del data
        # No cleanup

    final_rss_no_cleanup = get_rss_mb()
    print(f"  Initial: {initial_rss:.1f} MB")
    print(f"  Final: {final_rss_no_cleanup:.1f} MB")
    print(f"  Growth: {final_rss_no_cleanup - initial_rss:.1f} MB")

    # Force cleanup before next test
    cleanup_memory()

    # Test 2: With cleanup every iteration
    gc.collect()
    initial_rss = get_rss_mb()
    print("\nWith cleanup every round:")

    for i in range(iterations):
        data = allocate_memory(alloc_size_mb)
        del data
        cleanup_memory()

    final_rss_with_cleanup = get_rss_mb()
    print(f"  Initial: {initial_rss:.1f} MB")
    print(f"  Final: {final_rss_with_cleanup:.1f} MB")
    print(f"  Growth: {final_rss_with_cleanup - initial_rss:.1f} MB")

    # Comparison
    print("\nComparison:")
    diff = final_rss_no_cleanup - final_rss_with_cleanup
    if diff > 0:
        print(f"  Cleanup reduced final RSS by: {diff:.1f} MB")
    else:
        print("  Results similar (platform may handle memory differently)")


def main():
    """Run all memory tests."""
    test_cleanup_effectiveness()
    test_repeated_allocations()


if __name__ == "__main__":
    main()
