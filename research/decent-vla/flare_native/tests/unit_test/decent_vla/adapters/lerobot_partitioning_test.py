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

"""Tests for episode-level partitioning helpers. No lerobot needed.

`list_episodes` / `partition_lerobot_dataset` tests inject a fake
`lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata` into
`sys.modules` (same pattern as tests/adapters/test_lerobot_backend.py).
"""

from __future__ import annotations

import sys
import types

import pytest

from decent_vla.adapters.lerobot.backend import LeRobotBackend, partition_lerobot_dataset
from decent_vla.configs.schemas import DataConfig
from decent_vla.core.partitioner import EpisodeMeta

# 8 episodes, 4 tasks, 2 episodes per task: episode i -> task i // 2.
_N_EPISODES = 8
_N_TASKS = 4


def _install_fake_metadata(monkeypatch, *, with_tasks_mapping: bool):
    """Inject a fake LeRobotDatasetMetadata. Returns the fake class."""

    class _MetaCls:
        def __init__(self, repo_id, root=None):
            self.repo_id = repo_id
            self.root = root
            self.total_episodes = _N_EPISODES
            self.total_frames = 64
            self.episodes = {i: {"tasks": [f"task-{i // 2}"]} for i in range(_N_EPISODES)}
            if with_tasks_mapping:
                # v2-style: {task_index: task_name}
                self.tasks = {t: f"task-{t}" for t in range(_N_TASKS)}

    dl_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dl_mod.LeRobotDatasetMetadata = _MetaCls
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", types.ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dl_mod)
    return _MetaCls


@pytest.fixture
def fake_metadata(monkeypatch):
    return _install_fake_metadata(monkeypatch, with_tasks_mapping=True)


@pytest.fixture
def fake_metadata_no_tasks(monkeypatch):
    return _install_fake_metadata(monkeypatch, with_tasks_mapping=False)


# ---------------------------------------------------------------------------
# Optional Backend method: LeRobotBackend.list_episodes
# ---------------------------------------------------------------------------


def test_list_episodes_matches_metadata(fake_metadata):
    data_cfg = DataConfig(name="lerobot", repo_id="fake/repo", num_clients=4)
    eps = LeRobotBackend.list_episodes(data_cfg)
    assert len(eps) == _N_EPISODES
    assert all(isinstance(e, EpisodeMeta) for e in eps)
    for i, e in enumerate(eps):
        assert e.episode_index == i
        assert e.task_index == i // 2  # from the fake's v2-style tasks mapping
        assert e.task_name == f"task-{i // 2}"
    assert eps == LeRobotBackend.list_episodes(data_cfg)  # deterministic


def test_list_episodes_without_tasks_mapping_derives_indices(fake_metadata_no_tasks):
    """When metadata has no `tasks` attr, indices come from sorted unique names."""
    data_cfg = DataConfig(name="lerobot", repo_id="fake/repo")
    eps = LeRobotBackend.list_episodes(data_cfg)
    assert len(eps) == _N_EPISODES
    # sorted names task-0..task-3 -> indices 0..3
    for i, e in enumerate(eps):
        assert e.task_index == i // 2
        assert e.task_name == f"task-{i // 2}"


def test_list_episodes_requires_repo_id():
    with pytest.raises(ValueError, match="repo_id"):
        LeRobotBackend.list_episodes(DataConfig(name="lerobot"))


def test_list_episodes_repo_id_from_extra(fake_metadata):
    eps = LeRobotBackend.list_episodes(DataConfig(name="lerobot", extra={"repo_id": "fake/repo"}))
    assert len(eps) == _N_EPISODES


# ---------------------------------------------------------------------------
# Fix 2: episode with tasks=None in metadata -> task_name "unknown", valid index
# ---------------------------------------------------------------------------


def test_list_episodes_tasks_none_yields_unknown(monkeypatch):
    """An episode whose info dict has no 'tasks' key (tasks=None) must produce
    task_name='unknown' and a valid unique task_index — no exception."""

    class _MetaWithNullTask:
        def __init__(self, repo_id, root=None):
            self.repo_id = repo_id
            self.root = root
            self.total_episodes = 3
            self.total_frames = 24
            # episode 0: normal task, episode 1: explicit tasks=None (missing key),
            # episode 2: another normal task
            self.episodes = {
                0: {"tasks": ["pick_cube"]},
                1: {},  # no "tasks" key → tasks is None
                2: {"tasks": ["place_cube"]},
            }
            # No `tasks` attribute → fallback to sorted-unique enumeration

    dl_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dl_mod.LeRobotDatasetMetadata = _MetaWithNullTask
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", types.ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dl_mod)

    data_cfg = DataConfig(name="lerobot", repo_id="fake/repo")
    eps = LeRobotBackend.list_episodes(data_cfg)

    assert len(eps) == 3
    # All episodes must have valid unique task indices
    task_indices = [e.task_index for e in eps]
    assert len(set(task_indices)) == len(task_indices), "task_index must be unique per task_name"
    # Episode 1 (missing tasks key) must resolve to "unknown"
    ep1 = next(e for e in eps if e.episode_index == 1)
    assert ep1.task_name == "unknown"
    assert isinstance(ep1.task_index, int)
    # Episode 0 and 2 keep their proper names
    ep0 = next(e for e in eps if e.episode_index == 0)
    ep2 = next(e for e in eps if e.episode_index == 2)
    assert ep0.task_name == "pick_cube"
    assert ep2.task_name == "place_cube"


# ---------------------------------------------------------------------------
# Fix 3: v3 pandas-DataFrame branch of _task_name_to_index without pandas
# ---------------------------------------------------------------------------


def test_task_name_to_index_v3_dataframe_branch(monkeypatch):
    """Exercise the v3 pandas-DataFrame branch of _task_name_to_index using a
    fake DataFrame-like object — no pandas import required."""
    from decent_vla.adapters.lerobot.backend import _task_name_to_index

    # Simulate a pandas DataFrame with a 'task_index' column whose .items()
    # yields (task_name, task_index) pairs.
    class _FakeColumn:
        def __init__(self, pairs):
            self._pairs = pairs

        def items(self):
            return iter(self._pairs)

    class _FakeDataFrame:
        def __init__(self, mapping):
            # mapping: {task_name: task_index}
            self._mapping = mapping
            self.columns = list(mapping.keys()) + ["task_index"]

        def __contains__(self, item):
            return item in self.columns

        def __getitem__(self, key):
            if key == "task_index":
                return _FakeColumn(list(self._mapping.items()))
            raise KeyError(key)

    # Build a fake meta with v3-style tasks attribute
    fake_df = _FakeDataFrame({"pick_cube": 0, "place_cube": 1, "stack_blocks": 2})

    class _FakeMeta:
        tasks = fake_df

    result = _task_name_to_index(_FakeMeta(), iter([]))
    assert result == {"pick_cube": 0, "place_cube": 1, "stack_blocks": 2}


def test_list_episodes_v3_dataframe_mapping(monkeypatch):
    """list_episodes correctly resolves task indices from a v3 DataFrame tasks attr."""

    class _FakeColumn:
        def __init__(self, pairs):
            self._pairs = pairs

        def items(self):
            return iter(self._pairs)

    class _FakeDataFrame:
        def __init__(self, mapping):
            self._mapping = mapping
            self.columns = ["task_index"]

        def __getitem__(self, key):
            if key == "task_index":
                return _FakeColumn(list(self._mapping.items()))
            raise KeyError(key)

    class _MetaV3:
        def __init__(self, repo_id, root=None):
            self.repo_id = repo_id
            self.root = root
            self.total_episodes = 4
            self.total_frames = 32
            self.episodes = {
                0: {"tasks": ["pick_cube"]},
                1: {"tasks": ["pick_cube"]},
                2: {"tasks": ["place_cube"]},
                3: {"tasks": ["place_cube"]},
            }
            self.tasks = _FakeDataFrame({"pick_cube": 10, "place_cube": 20})

    dl_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dl_mod.LeRobotDatasetMetadata = _MetaV3
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", types.ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dl_mod)

    data_cfg = DataConfig(name="lerobot", repo_id="fake/repo")
    eps = LeRobotBackend.list_episodes(data_cfg)

    assert len(eps) == 4
    for e in eps:
        if e.task_name == "pick_cube":
            assert e.task_index == 10
        else:
            assert e.task_name == "place_cube"
            assert e.task_index == 20


# ---------------------------------------------------------------------------
# v3 episodes metadata: columnar datasets.Dataset instead of a dict
# ---------------------------------------------------------------------------


class _FakeEpisodesDataset:
    """Mirrors the real lerobot v3 `meta.episodes` interface: a HuggingFace
    `datasets.Dataset` with `column_names` and columnar `__getitem__` —
    crucially, no `.items()` method."""

    def __init__(self, columns: dict[str, list]):
        self._columns = columns
        self.column_names = list(columns.keys())

    def __getitem__(self, key):
        return self._columns[key]

    def __len__(self):
        return len(next(iter(self._columns.values())))


class _FakeTasksColumn:
    def __init__(self, pairs):
        self._pairs = pairs

    def items(self):
        return iter(self._pairs)


class _FakeTasksDataFrame:
    """Mirrors the real v3 `meta.tasks`: pandas DataFrame indexed by task
    name with a `task_index` column."""

    def __init__(self, mapping):
        self._mapping = mapping
        self.columns = ["task_index"]

    def __getitem__(self, key):
        if key == "task_index":
            return _FakeTasksColumn(list(self._mapping.items()))
        raise KeyError(key)


def test_list_episodes_v3_columnar_episodes(monkeypatch):
    """list_episodes handles v3 columnar episodes (datasets.Dataset shape)."""

    class _MetaV3Columnar:
        def __init__(self, repo_id, root=None):
            self.repo_id = repo_id
            self.root = root
            self.total_episodes = _N_EPISODES
            self.total_frames = 64
            self.episodes = _FakeEpisodesDataset(
                {
                    "episode_index": list(range(_N_EPISODES)),
                    "tasks": [[f"task-{i // 2}"] for i in range(_N_EPISODES)],
                    "length": [8] * _N_EPISODES,
                }
            )
            self.tasks = _FakeTasksDataFrame({f"task-{t}": t for t in range(_N_TASKS)})

    dl_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dl_mod.LeRobotDatasetMetadata = _MetaV3Columnar
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", types.ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dl_mod)

    data_cfg = DataConfig(name="lerobot", repo_id="fake/repo")
    eps = LeRobotBackend.list_episodes(data_cfg)

    assert len(eps) == _N_EPISODES
    assert all(isinstance(e, EpisodeMeta) for e in eps)
    for i, e in enumerate(eps):
        assert e.episode_index == i
        assert e.task_index == i // 2  # resolved via the DataFrame-style tasks
        assert e.task_name == f"task-{i // 2}"
    assert eps == LeRobotBackend.list_episodes(data_cfg)  # deterministic


def test_list_episodes_v3_columnar_empty_tasks_yields_unknown(monkeypatch):
    """A v3 row whose tasks entry is None falls back to 'unknown'."""

    class _MetaV3NullTask:
        def __init__(self, repo_id, root=None):
            self.repo_id = repo_id
            self.root = root
            self.total_episodes = 2
            self.total_frames = 16
            self.episodes = _FakeEpisodesDataset(
                {
                    "episode_index": [0, 1],
                    "tasks": [["pick_cube"], None],
                    "length": [8, 8],
                }
            )

    dl_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dl_mod.LeRobotDatasetMetadata = _MetaV3NullTask
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", types.ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dl_mod)

    eps = LeRobotBackend.list_episodes(DataConfig(name="lerobot", repo_id="fake/repo"))
    assert [e.task_name for e in eps] == ["pick_cube", "unknown"]


def test_list_episodes_unexpected_episodes_type_raises(monkeypatch):
    """An episodes attr that is neither dict-like nor columnar raises TypeError."""

    class _MetaBogus:
        def __init__(self, repo_id, root=None):
            self.episodes = 42

    dl_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dl_mod.LeRobotDatasetMetadata = _MetaBogus
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", types.ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dl_mod)

    with pytest.raises(TypeError, match="meta.episodes"):
        LeRobotBackend.list_episodes(DataConfig(name="lerobot", repo_id="fake/repo"))


# ---------------------------------------------------------------------------
# partition_lerobot_dataset (now a thin wrapper over list_episodes +
# core.partitioner.partition_episodes) — existing call surface preserved
# ---------------------------------------------------------------------------


def test_partition_lerobot_dataset_task(fake_metadata):
    chunks = partition_lerobot_dataset("fake/repo", num_clients=4, partitioning="task", seed=0)
    assert len(chunks) == 4
    assert sorted(i for c in chunks for i in c) == list(range(_N_EPISODES))
    # Tasks never split: each client's 2 episodes belong to one task (i // 2).
    for c in chunks:
        assert len({i // 2 for i in c}) == 1


@pytest.mark.parametrize("mode", ["iid_episode", "iid"])
def test_partition_lerobot_dataset_iid(fake_metadata, mode):
    chunks = partition_lerobot_dataset("fake/repo", num_clients=2, partitioning=mode, seed=7)
    assert len(chunks) == 2
    assert sorted(i for c in chunks for i in c) == list(range(_N_EPISODES))
    again = partition_lerobot_dataset("fake/repo", num_clients=2, partitioning=mode, seed=7)
    assert chunks == again


def test_partition_lerobot_dataset_unknown_mode(fake_metadata):
    with pytest.raises(ValueError, match="[Uu]nknown"):
        partition_lerobot_dataset("fake/repo", num_clients=2, partitioning="bogus")
