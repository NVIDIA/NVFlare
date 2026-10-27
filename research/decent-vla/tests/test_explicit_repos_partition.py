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

"""partition_for bypasses episode partitioning under explicit_repos."""
from __future__ import annotations

import pytest

from decent_vla.configs.schemas import BackendConfig, DataConfig, FLConfig, RunConfig, StrategyConfig
from decent_vla.runtime.builders import partition_for


class _Cls:
    """backend_cls stub: explicit_repos must NOT call list_episodes."""
    @staticmethod
    def list_episodes(data_cfg):
        raise AssertionError("list_episodes must not be called for explicit_repos")


def _cfg(repos):
    return RunConfig(
        name="t",
        backend=BackendConfig(name="mock"),
        data=DataConfig(name="lerobot", partitioning="explicit_repos",
                        client_repo_ids=repos, num_clients=len(repos) if repos else 1),
        fl=FLConfig(strategy=StrategyConfig(name="fedavg"), num_rounds=1, local_steps=1),
    )


def test_explicit_repos_returns_client_index_sentinels():
    chunks, manifest = partition_for(_cfg([["a/x", "a/y"], ["b/z"]]), _Cls)
    assert chunks == [[0], [1]]
    assert manifest.scheme == "explicit_repos"
    assert manifest.num_clients == 2
    assert manifest.clients["c0"]["repo_ids"] == ["a/x", "a/y"]
    assert manifest.clients["c1"]["repo_ids"] == ["b/z"]


def test_explicit_repos_requires_client_repo_ids():
    cfg = _cfg(None)
    with pytest.raises(ValueError, match="client_repo_ids"):
        partition_for(cfg, _Cls)
