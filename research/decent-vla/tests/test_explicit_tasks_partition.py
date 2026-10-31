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

"""explicit_tasks partitioning: config surface + split semantics + guards."""
from __future__ import annotations

import textwrap

import pytest

from decent_vla.configs import load_run_config
from decent_vla.configs.schemas import DataConfig
from decent_vla.core.partitioner import EpisodeMeta, HeldOutSpec, HeldOutTask, partition_episodes


def _episodes():
    """4 tasks x 3 episodes, episode_index globally unique (as lerobot ships)."""
    return [
        EpisodeMeta(episode_index=t * 3 + k, task_index=t, task_name=f"task{t}")
        for t in range(4)
        for k in range(3)
    ]


def _write(tmp_path, body: str) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body))
    return str(p)


# ---------------------------------------------------------------- config surface
def test_dataconfig_has_client_task_indices_default_none():
    assert DataConfig().client_task_indices is None


def test_explicit_tasks_parses_client_task_indices(tmp_path):
    cfg = load_run_config(_write(tmp_path, """
        name: t
        backend: {name: mock}
        data:
          name: lerobot
          repo_id: HuggingFaceVLA/libero
          partitioning: {scheme: explicit_tasks, num_clients: 2}
          client_task_indices: [[2, 11], [15, 19]]
        fl: {strategy: {name: fedavg}, rounds: 1, local_steps: 1, min_clients: 2}
    """))
    assert cfg.data.partitioning == "explicit_tasks"
    assert cfg.data.client_task_indices == [[2, 11], [15, 19]]
    assert cfg.data.num_clients == 2


# ---------------------------------------------------------------- split semantics
def test_explicit_tasks_splits_by_named_tasks():
    chunks, man = partition_episodes(
        _episodes(), scheme="explicit_tasks", num_clients=2,
        client_task_indices=[[0, 2], [1]],
    )
    assert chunks == [[0, 1, 2, 6, 7, 8], [3, 4, 5]]
    assert man.scheme == "explicit_tasks"
    assert man.clients["c0"]["task_names"] == ["task0", "task2"]
    assert man.clients["c1"]["task_names"] == ["task1"]
    # task3 is simply unused — an explicit list is allowed to leave tasks out
    assert man.total_episodes == 9
    assert man.extra["client_task_indices"] == [[0, 2], [1]]


def test_explicit_tasks_is_seed_independent():
    a, _ = partition_episodes(_episodes(), scheme="explicit_tasks", num_clients=2,
                              seed=1, client_task_indices=[[0], [1]])
    b, _ = partition_episodes(_episodes(), scheme="explicit_tasks", num_clients=2,
                              seed=99, client_task_indices=[[0], [1]])
    assert a == b


def test_explicit_tasks_one_task_per_client():
    chunks, man = partition_episodes(
        _episodes(), scheme="explicit_tasks", num_clients=4,
        client_task_indices=[[0], [1], [2], [3]],
    )
    assert [len(c) for c in chunks] == [3, 3, 3, 3]
    assert man.num_clients == 4


# ---------------------------------------------------------------- guards
def test_explicit_tasks_requires_the_list():
    with pytest.raises(ValueError, match="client_task_indices is required"):
        partition_episodes(_episodes(), scheme="explicit_tasks", num_clients=2)


def test_explicit_tasks_rejects_client_count_mismatch():
    with pytest.raises(ValueError, match="!= num_clients"):
        partition_episodes(_episodes(), scheme="explicit_tasks", num_clients=3,
                           client_task_indices=[[0], [1]])


def test_explicit_tasks_rejects_empty_client():
    with pytest.raises(ValueError, match="empty task list"):
        partition_episodes(_episodes(), scheme="explicit_tasks", num_clients=2,
                           client_task_indices=[[0], []])


def test_explicit_tasks_rejects_unknown_task():
    with pytest.raises(ValueError, match="absent from the dataset"):
        partition_episodes(_episodes(), scheme="explicit_tasks", num_clients=2,
                           client_task_indices=[[0], [77]])


def test_explicit_tasks_rejects_overlap():
    with pytest.raises(ValueError, match="must be disjoint"):
        partition_episodes(_episodes(), scheme="explicit_tasks", num_clients=2,
                           client_task_indices=[[0, 1], [1]])


def test_explicit_tasks_rejects_held_out_task():
    held = HeldOutSpec(tasks=(HeldOutTask("libero_goal", 2, 0, "task2"),))
    with pytest.raises(ValueError, match="held-out spec removes"):
        partition_episodes(_episodes(), scheme="explicit_tasks", num_clients=2,
                           held_out=held, client_task_indices=[[0], [2]])
