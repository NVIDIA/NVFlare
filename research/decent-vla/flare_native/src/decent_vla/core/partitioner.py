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

"""Partitioning types: episode metadata, held-out spec, audit manifest.

The held-out JSON keeps BOTH index spaces (spec §5): `dataset_task_index`
filters training; `(suite, eval_task_id)` tags eval results. Example:
configs/libero_heldout_8.json.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from decent_vla.core.results import config_hash


@dataclass(frozen=True)
class EpisodeMeta:
    """One episode of an episodic dataset, as the partitioners see it."""

    episode_index: int
    task_index: int
    task_name: str


@dataclass(frozen=True)
class HeldOutTask:
    suite: str
    dataset_task_index: int
    eval_task_id: int
    task_name: str


@dataclass(frozen=True)
class HeldOutSpec:
    tasks: tuple[HeldOutTask, ...]
    description: str = ""

    @property
    def dataset_task_indices(self) -> set[int]:
        return {t.dataset_task_index for t in self.tasks}

    def eval_ids_for_suite(self, suite: str) -> set[int]:
        return {t.eval_task_id for t in self.tasks if t.suite == suite}


def load_heldout_json(path: str | Path) -> HeldOutSpec:
    with Path(path).open() as f:
        raw = json.load(f)
    tasks = tuple(
        HeldOutTask(
            suite=t["suite"],
            dataset_task_index=int(t["dataset_task_index"]),
            eval_task_id=int(t["eval_task_id"]),
            task_name=t["task_name"],
        )
        for t in raw["tasks"]
    )
    return HeldOutSpec(tasks=tasks, description=raw.get("description", ""))


@dataclass
class PartitionManifest:
    """Audit artifact written before training step 1 (spec §5).

    Values in ``clients`` (and all nested dicts) must be JSON-native with
    string keys only — non-string keys won't roundtrip through write/read.
    """

    scheme: str
    num_clients: int
    seed: int
    dataset_repo_id: str | None
    dataset_revision: str | None
    held_out_task_indices: list[int]
    held_out_task_names: list[str]
    clients: dict[str, dict[str, Any]]  # client_id -> {task_names, episode_indices}
    total_episodes: int
    extra: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        return config_hash(asdict(self))

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def read(cls, path: str | Path) -> PartitionManifest:
        with Path(path).open() as f:
            return cls(**json.load(f))


_SCHEME_ALIASES = {"task": "task_round_robin", "iid": "iid_episode"}


def partition_episodes(
    episodes: Sequence[EpisodeMeta],
    scheme: str,
    num_clients: int,
    seed: int = 42,
    held_out: HeldOutSpec | None = None,
    dataset_repo_id: str | None = None,
    dataset_revision: str | None = None,
    client_task_indices: Sequence[Sequence[int]] | None = None,
) -> tuple[list[list[int]], PartitionManifest]:
    """Partition episodes into per-client episode-index lists + audit manifest.

    Held-out tasks are dropped from the pool BEFORE partitioning (spec §5),
    so FL (num_clients=N) and centralised (num_clients=1) train on the
    identical reduced pool.

    `client_task_indices` is required by — and only read by — the
    ``explicit_tasks`` scheme: one list of dataset task indices per client,
    spelled out in the config instead of derived from a seed.
    """
    scheme = _SCHEME_ALIASES.get(scheme, scheme)
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")

    held_idx = held_out.dataset_task_indices if held_out else set()
    missing = held_idx - {e.task_index for e in episodes}
    if missing:
        raise ValueError(f"held_out spec references task indices absent from the dataset: {sorted(missing)}")
    held_names = sorted({t.task_name for t in held_out.tasks}) if held_out else []
    pool = [e for e in episodes if e.task_index not in held_idx]

    if scheme == "task_round_robin":
        by_task: dict[int, list[EpisodeMeta]] = {}
        for e in pool:
            by_task.setdefault(e.task_index, []).append(e)
        tasks = sorted(by_task)
        if len(tasks) < num_clients:
            raise ValueError(f"task_round_robin: {len(tasks)} tasks < {num_clients} clients")
        rng = random.Random(seed)
        rng.shuffle(tasks)
        task_chunks = [tasks[i::num_clients] for i in range(num_clients)]
        chunks = [sorted(e.episode_index for t in tc for e in by_task[t]) for tc in task_chunks]
        client_tasks = [sorted(by_task[t][0].task_name for t in tc) for tc in task_chunks]
    elif scheme == "explicit_tasks":
        # Spelled-out, seed-free counterpart of task_round_robin: the config
        # names exactly which dataset task indices each client holds. Used when
        # WHICH tasks sit together is the experimental variable (e.g. a
        # deliberately maximally-heterogeneous client set) rather than a
        # nuisance the seed should randomise away.
        if not client_task_indices:
            raise ValueError(
                "explicit_tasks: data.client_task_indices is required " "(one list of dataset task indices per client)"
            )
        if len(client_task_indices) != num_clients:
            raise ValueError(
                f"explicit_tasks: {len(client_task_indices)} client task lists " f"!= num_clients={num_clients}"
            )
        by_task = {}
        for e in pool:
            by_task.setdefault(e.task_index, []).append(e)
        owner: dict[int, int] = {}
        chunks, client_tasks = [], []
        for i, tids in enumerate(client_task_indices):
            tids = [int(t) for t in tids]
            if not tids:
                raise ValueError(f"explicit_tasks: client {i} has an empty task list")
            for t in tids:
                if t in held_idx:
                    raise ValueError(
                        f"explicit_tasks: client {i} asks for task {t}, which the "
                        f"held-out spec removes from training"
                    )
                if t not in by_task:
                    raise ValueError(
                        f"explicit_tasks: client {i} asks for task index {t}, absent "
                        f"from the dataset (available: {min(by_task)}..{max(by_task)})"
                    )
                # Disjointness is not decoration: a task listed twice would be
                # counted twice by every size-weighted aggregation downstream.
                if t in owner:
                    raise ValueError(
                        f"explicit_tasks: task {t} assigned to both client "
                        f"{owner[t]} and client {i}; client task lists must be disjoint"
                    )
                owner[t] = i
            chunks.append(sorted(e.episode_index for t in tids for e in by_task[t]))
            client_tasks.append(sorted({by_task[t][0].task_name for t in tids}))
    elif scheme == "iid_episode":
        rng = random.Random(seed)
        idx = sorted(e.episode_index for e in pool)
        rng.shuffle(idx)
        chunks = [sorted(idx[i::num_clients]) for i in range(num_clients)]
        name_by_ep = {e.episode_index: e.task_name for e in pool}
        client_tasks = [sorted({name_by_ep[i] for i in c}) for c in chunks]
    else:
        raise ValueError(f"Unknown partitioning scheme: {scheme!r}")

    manifest = PartitionManifest(
        scheme=scheme,
        num_clients=num_clients,
        seed=seed,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        held_out_task_indices=sorted(held_idx),
        held_out_task_names=held_names,
        clients={f"c{i}": {"task_names": client_tasks[i], "episode_indices": chunks[i]} for i in range(num_clients)},
        total_episodes=sum(len(c) for c in chunks),
        extra=(
            {"client_task_indices": [[int(t) for t in c] for c in client_task_indices]}
            if scheme == "explicit_tasks"
            else {}
        ),
    )
    return chunks, manifest
