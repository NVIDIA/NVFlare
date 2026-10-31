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

"""Each site reconstructs a deterministic, disjoint share of the common data."""

import pytest

from src.support.partitioning import EpisodeMeta, HeldOutSpec, HeldOutTask, partition_episodes


@pytest.mark.parametrize("scheme", ["task_round_robin", "iid_episode"])
@pytest.mark.parametrize("clients", [2, 3])
def test_partition_is_disjoint_complete_and_reproducible(scheme, clients):
    episodes = [
        EpisodeMeta(episode_index=3 * t + k, task_index=t, task_name=f"task{t}") for t in range(4) for k in range(3)
    ]
    held = HeldOutSpec(tasks=(HeldOutTask("libero_goal", 3, 0, "task3"),))
    chunks, _ = partition_episodes(episodes, scheme, clients, seed=42, held_out=held)
    again, _ = partition_episodes(list(reversed(episodes)), scheme, clients, seed=42, held_out=held)
    assert chunks == again
    assigned = [index for chunk in chunks for index in chunk]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(range(9))  # all training episodes; held-out task 3 is absent
    assert all(chunks)


def test_explicit_task_overlap_is_rejected():
    episodes = [EpisodeMeta(episode_index=t, task_index=t, task_name=f"task{t}") for t in range(3)]
    with pytest.raises(ValueError, match="must be disjoint"):
        partition_episodes(episodes, "explicit_tasks", 2, client_task_indices=[[0, 1], [1, 2]])
