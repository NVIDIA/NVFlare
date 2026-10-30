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

"""Aggregated stats over the union of client repos."""
from __future__ import annotations

import sys
import types

import numpy as np


def _fake_meta_module(monkeypatch, stats_by_repo):
    class _MetaCls:
        def __init__(self, repo_id, root=None):
            self.repo_id = repo_id
            self.stats = stats_by_repo[repo_id]
            self.total_episodes = 1
            self.episodes = {0: {"tasks": ["t"]}}
    try:
        # With lerobot installed, patch the class on the real module: replacing the whole module breaks the
        # lerobot.datasets package import, which also needs LeRobotDataset from it.
        import lerobot.datasets.lerobot_dataset as mod
    except ImportError:
        mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
        monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", mod)
    monkeypatch.setattr(mod, "LeRobotDatasetMetadata", _MetaCls, raising=False)


def test_aggregated_stats_union(monkeypatch):
    from decent_vla.adapters.lerobot.backend import aggregated_dataset_stats
    stats = {
        "r/a": {"action": {"mean": np.array([0.0]), "std": np.array([1.0]),
                            "count": np.array([100]), "min": np.array([-1.0]),
                            "max": np.array([1.0])}},
        "r/b": {"action": {"mean": np.array([10.0]), "std": np.array([1.0]),
                            "count": np.array([100]), "min": np.array([9.0]),
                            "max": np.array([11.0])}},
    }
    _fake_meta_module(monkeypatch, stats)
    agg = aggregated_dataset_stats([["r/a"], ["r/b"]])
    assert np.isclose(agg["action"]["mean"][0], 5.0)
    assert agg["action"]["min"][0] == -1.0 and agg["action"]["max"][0] == 11.0


def test_aggregated_stats_dedups_repeated_repo(monkeypatch):
    from decent_vla.adapters.lerobot.backend import aggregated_dataset_stats
    stats = {"r/a": {"action": {"mean": np.array([2.0]), "std": np.array([1.0]),
                                "count": np.array([50]), "min": np.array([0.0]),
                                "max": np.array([4.0])}}}
    _fake_meta_module(monkeypatch, stats)
    # same repo appears twice across clients -> counted ONCE
    agg = aggregated_dataset_stats([["r/a"], ["r/a"]])
    assert np.isclose(agg["action"]["mean"][0], 2.0)
