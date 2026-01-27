# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Memory management utilities for federated learning.

This module provides memory cleanup utilities to help manage RSS (Resident Set Size)
in long-running FL jobs using Python + PyTorch + glibc/jemalloc.

Allocator Support:
- glibc: Uses malloc_trim() to return freed pages to OS
- jemalloc: Relies on auto-decay (MALLOC_CONF), no manual action needed

Best Practices:
- Client: Set MALLOC_ARENA_MAX=2, cleanup every round
- Server: Set MALLOC_ARENA_MAX=4, cleanup every 5 rounds
- jemalloc: Set MALLOC_CONF="dirty_decay_ms:5000,muzzy_decay_ms:5000"

Usage:
    from nvflare.fuel.utils.memory_utils import cleanup_memory, get_allocator_type

    # Check which allocator is in use
    allocator = get_allocator_type()  # "glibc", "jemalloc", or "unknown"

    # At end of each round (client) or every N rounds (server)
    cleanup_memory(torch_cuda_empty_cache=True)  # True for PyTorch GPU clients
"""

import ctypes
import gc
import logging
from ctypes import CDLL, c_size_t
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_glibc() -> Optional[CDLL]:
    """Get glibc library handle if available (Linux only).

    Returns:
        CDLL handle to glibc with malloc_trim configured, or None if not available.
    """
    try:
        libc = CDLL("libc.so.6")
        if not hasattr(libc, "malloc_trim"):
            return None
        libc.malloc_trim.argtypes = [c_size_t]
        libc.malloc_trim.restype = int
        return libc
    except (OSError, AttributeError):
        # Not Linux, or glibc not available (e.g., Alpine/musl)
        return None


@lru_cache(maxsize=1)
def get_allocator_type() -> str:
    """Detect which memory allocator is in use at runtime.

    Returns:
        "jemalloc": jemalloc is loaded (recommended for PyTorch)
        "glibc": Standard glibc malloc is in use
        "unknown": Could not detect allocator type

    Note:
        - jemalloc is typically loaded via LD_PRELOAD
        - Detection is cached after first call
        - Safe to call frequently (no overhead after first call)
    """
    try:
        # Load the C library that the process is using
        libc = ctypes.CDLL(None)

        # jemalloc has mallctl function
        if hasattr(libc, "mallctl"):
            return "jemalloc"

        # glibc has malloc_trim
        if hasattr(libc, "malloc_trim"):
            return "glibc"
    except Exception:
        pass

    return "unknown"


def try_malloc_trim() -> Optional[int]:
    """Attempt to release free memory back to the OS (glibc only).

    This calls glibc's malloc_trim(0) to return free heap pages to the OS,
    which helps reduce RSS (Resident Set Size) after memory-intensive operations.

    Returns:
        1 if memory was released, 0 if not, None if malloc_trim is not available.

    Note:
        - Only works on Linux with glibc (not musl/Alpine)
        - Safe no-op on other platforms
        - Very low overhead, safe to call frequently
    """
    libc = _get_glibc()
    if libc is None:
        return None
    try:
        return int(libc.malloc_trim(0))
    except Exception as e:
        logger.debug(f"malloc_trim failed: {e}")
        return None


def cleanup_memory(torch_cuda_empty_cache: bool = False) -> None:
    """Perform allocator-aware memory cleanup to reduce RSS.

    This function:
    1. Runs Python garbage collection (gc.collect)
    2. For glibc: Releases free heap pages to OS (malloc_trim)
       For jemalloc: Relies on auto-decay (no manual action needed)
    3. Optionally clears PyTorch CUDA cache

    Args:
        torch_cuda_empty_cache: If True, also call torch.cuda.empty_cache().
            Only applicable to PyTorch GPU clients.

    Note:
        Call this at the end of each FL round (client) or every N rounds (server).
        The function automatically detects the allocator type and applies
        the appropriate cleanup strategy.
    """
    # Step 1: Python garbage collection (always)
    gc.collect()

    # Step 2: Allocator-specific cleanup
    allocator = get_allocator_type()
    if allocator == "glibc":
        # glibc: manually return freed pages to OS
        result = try_malloc_trim()
        if result is not None:
            logger.debug(f"malloc_trim returned {result}")
    elif allocator == "jemalloc":
        # jemalloc: auto-decay handles memory return, no manual action needed
        # Memory is returned based on MALLOC_CONF settings (dirty_decay_ms, muzzy_decay_ms)
        logger.debug("jemalloc detected, relying on auto-decay for memory management")
    # unknown: gc.collect() is the only safe action

    # Step 3: Clear PyTorch CUDA cache if requested
    if torch_cuda_empty_cache:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("torch.cuda.empty_cache() called")
        except ImportError:
            pass  # PyTorch not installed
        except Exception as e:
            logger.debug(f"cuda.empty_cache failed: {e}")
