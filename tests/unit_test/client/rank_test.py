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

import pytest

from nvflare.client import rank as rank_utils


class _FakeDist:
    def __init__(self, rank=1, world_size=2, initialized=True):
        self.rank = rank
        self.world_size = world_size
        self.initialized = initialized

    @staticmethod
    def is_available():
        return True

    def is_initialized(self):
        return self.initialized

    def get_world_size(self):
        return self.world_size

    def get_rank(self):
        return self.rank


def _clear_rank_environment(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    for name in rank_utils.MULTIRANK_SIZE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_initialized_torch_rank_is_discovered_without_import(monkeypatch):
    monkeypatch.setitem(rank_utils.sys.modules, "torch.distributed", _FakeDist(rank=3, world_size=4))

    assert rank_utils.get_initialized_torch_distributed_rank() == 3


def test_missing_or_single_process_torch_group_has_no_distributed_rank(monkeypatch):
    monkeypatch.delitem(rank_utils.sys.modules, "torch.distributed", raising=False)
    assert rank_utils.get_initialized_torch_distributed_rank() is None

    monkeypatch.setitem(rank_utils.sys.modules, "torch.distributed", _FakeDist(rank=0, world_size=1))
    assert rank_utils.get_initialized_torch_distributed_rank() is None


@pytest.mark.parametrize("size_marker", rank_utils.MULTIRANK_SIZE_ENV_VARS)
def test_environment_declares_multirank_for_supported_launchers(monkeypatch, size_marker):
    _clear_rank_environment(monkeypatch)
    monkeypatch.setenv(size_marker, "2")

    assert rank_utils.environment_declares_multirank()


@pytest.mark.parametrize(("raw_rank", "expected_rank"), ((0, "0"), ("+0", "0"), ("00", "0"), ("01", "1")))
def test_normalize_process_rank_returns_canonical_decimal(raw_rank, expected_rank):
    assert rank_utils.normalize_process_rank(raw_rank) == expected_rank


@pytest.mark.parametrize("invalid_rank", (-1, "-1", "rank-zero", True, 1.5))
def test_normalize_process_rank_rejects_invalid_values(invalid_rank):
    with pytest.raises(ValueError, match="rank must be"):
        rank_utils.normalize_process_rank(invalid_rank)


def test_resolve_process_rank_prefers_explicit_rank(monkeypatch):
    _clear_rank_environment(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(rank_utils, "get_initialized_torch_distributed_rank", lambda: 1)

    assert rank_utils.resolve_process_rank("00") == "0"


def test_resolve_process_rank_uses_initialized_torch_rank(monkeypatch):
    _clear_rank_environment(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(rank_utils, "get_initialized_torch_distributed_rank", lambda: 1)

    assert rank_utils.resolve_process_rank() == "1"


def test_resolve_process_rank_uses_environment_rank(monkeypatch):
    _clear_rank_environment(monkeypatch)
    monkeypatch.setenv("RANK", "+0")
    monkeypatch.setattr(rank_utils, "get_initialized_torch_distributed_rank", lambda: None)

    assert rank_utils.resolve_process_rank() == "0"


@pytest.mark.parametrize("size_marker", rank_utils.MULTIRANK_SIZE_ENV_VARS)
def test_resolve_process_rank_rejects_unresolved_multirank(monkeypatch, size_marker):
    _clear_rank_environment(monkeypatch)
    monkeypatch.setenv(size_marker, "2")
    monkeypatch.setattr(rank_utils, "get_initialized_torch_distributed_rank", lambda: None)

    with pytest.raises(RuntimeError, match="global RANK is unavailable"):
        rank_utils.resolve_process_rank()


def test_resolve_process_rank_defaults_single_process_to_zero(monkeypatch):
    _clear_rank_environment(monkeypatch)
    monkeypatch.setattr(rank_utils, "get_initialized_torch_distributed_rank", lambda: None)

    assert rank_utils.resolve_process_rank() == "0"
