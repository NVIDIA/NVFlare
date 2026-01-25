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

        with patch.object(gc, "collect") as mock_gc:
            with patch("nvflare.fuel.utils.memory_utils.try_malloc_trim"):
                cleanup_memory()
                mock_gc.assert_called_once()

    def test_cleanup_memory_calls_try_malloc_trim(self):
        """Test that cleanup_memory calls try_malloc_trim."""
        from nvflare.fuel.utils.memory_utils import cleanup_memory

        with patch("nvflare.fuel.utils.memory_utils.try_malloc_trim") as mock_trim:
            cleanup_memory()
            mock_trim.assert_called_once()

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
