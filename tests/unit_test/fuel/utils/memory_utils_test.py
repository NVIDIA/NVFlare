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

import gc
from unittest.mock import patch


class TestMemoryUtils:
    """Tests for nvflare.fuel.utils.memory_utils module."""

    def test_try_malloc_trim_returns_int_or_none(self):
        """Test that try_malloc_trim returns int on Linux/glibc or None otherwise."""
        from nvflare.fuel.utils.memory_utils import try_malloc_trim

        result = try_malloc_trim()
        # On Linux with glibc, returns 0 or 1; on other systems, returns None
        assert result is None or isinstance(result, int)

    def test_cleanup_memory_calls_gc_collect(self):
        """Test that cleanup_memory calls gc.collect."""
        from nvflare.fuel.utils.memory_utils import cleanup_memory

        with patch.object(gc, "collect", return_value=0) as mock_gc:
            with patch("nvflare.fuel.utils.memory_utils.try_malloc_trim"):
                cleanup_memory()
                mock_gc.assert_called_once()

    def test_cleanup_memory_calls_try_malloc_trim_for_glibc(self):
        """Test that cleanup_memory calls try_malloc_trim for glibc allocator."""
        from nvflare.fuel.utils.memory_utils import cleanup_memory

        with patch("nvflare.fuel.utils.memory_utils.get_allocator_type", return_value="glibc"):
            with patch("nvflare.fuel.utils.memory_utils.try_malloc_trim") as mock_trim:
                cleanup_memory()
                mock_trim.assert_called_once()

    def test_cleanup_memory_skips_malloc_trim_for_jemalloc(self):
        """Test that cleanup_memory skips try_malloc_trim for jemalloc allocator."""
        from nvflare.fuel.utils.memory_utils import cleanup_memory

        with patch("nvflare.fuel.utils.memory_utils.get_allocator_type", return_value="jemalloc"):
            with patch("nvflare.fuel.utils.memory_utils.try_malloc_trim") as mock_trim:
                cleanup_memory()
                mock_trim.assert_not_called()

    def test_cleanup_memory_cuda_empty_cache_false(self):
        """Test that cleanup_memory with cuda_empty_cache=False does not call torch."""
        from nvflare.fuel.utils.memory_utils import cleanup_memory

        # This should not raise and should not try to import torch
        cleanup_memory(cuda_empty_cache=False)

    def test_cleanup_memory_cuda_empty_cache_true(self):
        """Test that cleanup_memory handles cuda_empty_cache=True gracefully."""
        from nvflare.fuel.utils.memory_utils import cleanup_memory

        # This should not raise even if torch is not installed or CUDA unavailable
        cleanup_memory(cuda_empty_cache=True)

    def test_get_glibc_caching(self):
        """Test that _get_glibc is cached (only loads once)."""
        from nvflare.fuel.utils.memory_utils import _get_glibc

        # Clear the cache first
        _get_glibc.cache_clear()

        result1 = _get_glibc()
        result2 = _get_glibc()

        # Should return the same object (cached)
        assert result1 is result2

        # Check cache info
        cache_info = _get_glibc.cache_info()
        assert cache_info.hits >= 1  # Second call should be a cache hit

    def test_get_allocator_type_returns_valid_string(self):
        """Test that get_allocator_type returns a valid allocator type."""
        from nvflare.fuel.utils.memory_utils import get_allocator_type

        # Clear cache first
        get_allocator_type.cache_clear()

        result = get_allocator_type()
        assert result in ("glibc", "jemalloc", "unknown")

    def test_get_allocator_type_caching(self):
        """Test that get_allocator_type is cached (only detects once)."""
        from nvflare.fuel.utils.memory_utils import get_allocator_type

        # Clear the cache first
        get_allocator_type.cache_clear()

        result1 = get_allocator_type()
        result2 = get_allocator_type()

        # Should return the same result
        assert result1 == result2

        # Check cache info - second call should be a cache hit
        cache_info = get_allocator_type.cache_info()
        assert cache_info.hits >= 1

    def test_cleanup_memory_allocator_aware(self):
        """Test that cleanup_memory adapts behavior based on allocator type."""
        from nvflare.fuel.utils.memory_utils import cleanup_memory, get_allocator_type

        # This should work regardless of allocator type
        get_allocator_type.cache_clear()
        cleanup_memory()

    # -----------------------------------------------------------------
    # M8 fix: gc.collect() result logged at INFO when non-zero
    # -----------------------------------------------------------------

    def test_gc_collect_nonzero_logs_info(self):
        """M8 fix: cleanup_memory must log at INFO when gc.collect() returns non-zero.

        Before M8 fix, gc.collect() result was silently discarded.  Non-zero means
        reference cycles were broken — actionable signal worth surfacing in logs.
        """
        from nvflare.fuel.utils.memory_utils import cleanup_memory

        with patch("nvflare.fuel.utils.memory_utils.logger") as mock_logger:
            with patch.object(gc, "collect", return_value=5):
                cleanup_memory()
            mock_logger.info.assert_called_once()
            msg = mock_logger.info.call_args[0][0]
            assert "5" in msg or "freed" in msg.lower(), f"INFO log must mention the freed count. Got: {msg}"

    def test_gc_collect_zero_does_not_log_info(self):
        """M8 fix: cleanup_memory must NOT log INFO when gc.collect() returns 0.

        Zero freed objects is expected in steady state; logging it every round
        would be noisy with no actionable signal.
        """
        from nvflare.fuel.utils.memory_utils import cleanup_memory

        with patch("nvflare.fuel.utils.memory_utils.logger") as mock_logger:
            with patch.object(gc, "collect", return_value=0):
                cleanup_memory()
            # logger.info must not be called for gc result when freed=0
            # (other info calls from allocator-specific paths are fine, but
            # those are at DEBUG level — so info should not be called at all here)
            for call in mock_logger.info.call_args_list:
                msg = call[0][0]
                assert "freed" not in msg.lower(), f"Must not log 'freed' when gc.collect()=0. Got: {msg}"

        # Verify it completed without error - allocator-specific logic handled internally
