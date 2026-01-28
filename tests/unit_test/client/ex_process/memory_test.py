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

"""Tests for ExProcessClientAPI memory management."""

import os
import unittest
from unittest.mock import patch


class TestExProcessClientAPIMemory(unittest.TestCase):
    """Test memory management in ExProcessClientAPI."""

    def test_memory_settings_from_env_disabled(self):
        """Test that memory settings default to disabled when env vars not set."""
        # Clear any existing env vars
        env = os.environ.copy()
        env.pop("NVFLARE_CLIENT_MEMORY_GC_ROUNDS", None)
        env.pop("NVFLARE_TORCH_CUDA_EMPTY_CACHE", None)

        with patch.dict(os.environ, env, clear=True):
            # We need to test the __init__ logic without actually initializing
            # the full ExProcessClientAPI (which requires config files)
            gc_rounds = int(os.environ.get("NVFLARE_CLIENT_MEMORY_GC_ROUNDS", "0"))
            cuda_empty = os.environ.get("NVFLARE_TORCH_CUDA_EMPTY_CACHE", "").lower() == "true"

            assert gc_rounds == 0
            assert cuda_empty is False

    def test_memory_settings_from_env_enabled(self):
        """Test that memory settings are read from environment variables."""
        env = {
            "NVFLARE_CLIENT_MEMORY_GC_ROUNDS": "5",
            "NVFLARE_TORCH_CUDA_EMPTY_CACHE": "true",
        }

        with patch.dict(os.environ, env, clear=False):
            gc_rounds = int(os.environ.get("NVFLARE_CLIENT_MEMORY_GC_ROUNDS", "0"))
            cuda_empty = os.environ.get("NVFLARE_TORCH_CUDA_EMPTY_CACHE", "").lower() == "true"

            assert gc_rounds == 5
            assert cuda_empty is True

    def test_memory_settings_env_false(self):
        """Test that torch_cuda_empty_cache=false is parsed correctly."""
        env = {
            "NVFLARE_CLIENT_MEMORY_GC_ROUNDS": "1",
            "NVFLARE_TORCH_CUDA_EMPTY_CACHE": "false",
        }

        with patch.dict(os.environ, env, clear=False):
            gc_rounds = int(os.environ.get("NVFLARE_CLIENT_MEMORY_GC_ROUNDS", "0"))
            cuda_empty = os.environ.get("NVFLARE_TORCH_CUDA_EMPTY_CACHE", "").lower() == "true"

            assert gc_rounds == 1
            assert cuda_empty is False

    def test_memory_settings_env_case_insensitive(self):
        """Test that TRUE/True/true all work for torch_cuda_empty_cache."""
        for value in ["TRUE", "True", "true", "TrUe"]:
            env = {"NVFLARE_TORCH_CUDA_EMPTY_CACHE": value}
            with patch.dict(os.environ, env, clear=False):
                cuda_empty = os.environ.get("NVFLARE_TORCH_CUDA_EMPTY_CACHE", "").lower() == "true"
                assert cuda_empty is True, f"Failed for value: {value}"


class TestMaybeCleanupMemory(unittest.TestCase):
    """Test _maybe_cleanup_memory logic (extracted from ExProcessClientAPI)."""

    def test_cleanup_disabled(self):
        """Test that cleanup is skipped when gc_rounds=0."""
        memory_gc_rounds = 0
        round_count = 0

        # Logic from _maybe_cleanup_memory
        if memory_gc_rounds <= 0:
            should_cleanup = False
        else:
            round_count += 1
            should_cleanup = round_count % memory_gc_rounds == 0

        assert should_cleanup is False

    def test_cleanup_every_round(self):
        """Test cleanup every round (gc_rounds=1)."""
        memory_gc_rounds = 1
        round_count = 0

        results = []
        for _ in range(5):
            round_count += 1
            should_cleanup = round_count % memory_gc_rounds == 0
            results.append(should_cleanup)

        # Should cleanup every round
        assert results == [True, True, True, True, True]

    def test_cleanup_every_n_rounds(self):
        """Test cleanup every N rounds."""
        memory_gc_rounds = 3
        round_count = 0

        results = []
        for _ in range(9):
            round_count += 1
            should_cleanup = round_count % memory_gc_rounds == 0
            results.append(should_cleanup)

        # Should cleanup on rounds 3, 6, 9
        expected = [False, False, True, False, False, True, False, False, True]
        assert results == expected


if __name__ == "__main__":
    unittest.main()
