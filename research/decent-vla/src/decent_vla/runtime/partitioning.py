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

"""Dataset partitioning helpers used to give each FL client its slice.

These are pure-Python helpers; the actual dataloader instantiation is
done by the backend in `Backend.build_dataloader(data_cfg, partition_indices)`.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from decent_vla.core.partitioner import partition_episodes  # noqa: F401


def iid_partition(
    num_samples: int, num_clients: int, seed: int = 42
) -> list[list[int]]:
    """Random shuffle, then chunk into `num_clients` near-equal lists of indices."""
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if num_samples < num_clients:
        raise ValueError(f"num_samples ({num_samples}) < num_clients ({num_clients})")
    rng = random.Random(seed)
    idx = list(range(num_samples))
    rng.shuffle(idx)
    chunks = [idx[i::num_clients] for i in range(num_clients)]
    return chunks


def task_partition(
    task_ids: Sequence[int], num_clients: int, seed: int = 42
) -> list[list[int]]:
    """Group samples by their task_id, then assign whole tasks to clients.

    Used for the non-IID setting (which the FL papers call "task-partitioned"
    or "feature-skewed"). Each client sees only a subset of tasks; tasks
    do not cross client boundaries.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    by_task: dict[int, list[int]] = {}
    for i, t in enumerate(task_ids):
        by_task.setdefault(int(t), []).append(i)

    unique_tasks = sorted(by_task.keys())
    if len(unique_tasks) < num_clients:
        raise ValueError(
            f"Cannot task-partition: only {len(unique_tasks)} unique tasks "
            f"but {num_clients} clients requested."
        )

    rng = random.Random(seed)
    rng.shuffle(unique_tasks)
    task_chunks = [unique_tasks[i::num_clients] for i in range(num_clients)]
    return [sorted(idx for t in chunk for idx in by_task[t]) for chunk in task_chunks]
